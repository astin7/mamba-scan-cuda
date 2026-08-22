/*
 * Stage 2: Parallel Associative Scan for Mamba Selective Scan (CUDA)
 * =========================================================================
 * INTERVIEW PREP NOTE: read this whole comment block before an interview.
 *
 * THE PROBLEM WE'RE SOLVING
 * --------------------------
 * The recurrence h[t] = Abar[t] * h[t-1] + Bbar_u[t] is sequential: you
 * cannot compute h[5] without first computing h[4], h[3], ... h[0]. A naive
 * GPU implementation would do this as a single-threaded loop, wasting
 * ~10,000 idle CUDA cores while one thread crawls through the sequence.
 *
 * THE TRICK: PARALLEL ASSOCIATIVE SCAN
 * --------------------------------------
 * Define a pair (a, b) meaning "multiply state by a, then add b".
 * Composing two such pairs is ASSOCIATIVE:
 *     (a2,b2) . (a1,b1) = (a2*a1, a2*b1 + b2)
 * so we can compute the whole sequence's cumulative composition with a
 * parallel scan instead of a serial loop - same algorithm family as
 * GPU parallel prefix-sum.
 *
 * TWO KERNELS IN THIS FILE
 * --------------------------
 * 1. warp_scan_forward   - handles seq_len <= 32 (single warp, Hillis-Steele
 *                           scan using __shfl_up_sync register communication)
 * 2. block_scan_forward  - handles seq_len <= 1024 (up to 32 warps/block),
 *                           using a two-level scan: each warp scans its own
 *                           32-timestep chunk, then a second pass combines
 *                           per-warp totals so every warp knows its correct
 *                           "carry-in" from all earlier warps.
 *
 * PADDING (block_scan_forward only)
 * ------------------------------------
 * __shfl_up_sync requires all 32 threads in a warp to participate. If some
 * threads exited early (t >= seq_len), that's undefined behavior. Fix:
 * pad the sequence to a multiple of 32 with IDENTITY values (Abar=1 "no
 * change", Bbar_u=0 "add nothing") so every thread always does valid,
 * harmless work. Padding is added before the kernel launch and stripped
 * off the output afterward - see block_scan_forward below.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MAX_WARPS_PER_BLOCK 32  // -> max 1024 threads/timesteps per block

/*
 * Composes two (a, b) pairs: applying transform 2 AFTER transform 1.
 *   a_out = a2 * a1
 *   b_out = a2 * b1 + b2
 */
__device__ __forceinline__ void compose(
    float a1, float b1,
    float a2, float b2,
    float &a_out, float &b_out
) {
    a_out = a2 * a1;
    b_out = a2 * b1 + b2;
}

// ============================================================
// Kernel 1: single-warp scan (seq_len <= 32)
// ============================================================
__global__ void warp_scan_kernel(
    const float* __restrict__ Abar,
    const float* __restrict__ Bbar_u,
    float* __restrict__ h_out,
    int seq_len
) {
    int row = blockIdx.x;
    int t = threadIdx.x;

    if (t >= seq_len) return;  // safe here since blockDim.x == seq_len <= 32,
                                 // i.e. every launched thread is a real timestep

    int idx = row * seq_len + t;

    float a = Abar[idx];
    float b = Bbar_u[idx];

    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float a_prev = __shfl_up_sync(0xFFFFFFFF, a, offset);
        float b_prev = __shfl_up_sync(0xFFFFFFFF, b, offset);
        if (t >= offset) {
            float a_new, b_new;
            compose(a_prev, b_prev, a, b, a_new, b_new);
            a = a_new;
            b = b_new;
        }
    }

    h_out[idx] = b;
}

torch::Tensor warp_scan_forward(torch::Tensor Abar, torch::Tensor Bbar_u) {
    TORCH_CHECK(Abar.is_cuda(), "Abar must be a CUDA tensor");
    TORCH_CHECK(Bbar_u.is_cuda(), "Bbar_u must be a CUDA tensor");
    TORCH_CHECK(Abar.sizes() == Bbar_u.sizes(), "Abar and Bbar_u must have the same shape");
    TORCH_CHECK(Abar.dim() == 2, "expected flattened 2D input (rows, seq_len)");

    int num_rows = Abar.size(0);
    int seq_len = Abar.size(1);
    TORCH_CHECK(seq_len <= WARP_SIZE, "warp_scan_forward only supports seq_len <= 32; use block_scan_forward for longer sequences");

    auto Abar_c = Abar.contiguous();
    auto Bbar_u_c = Bbar_u.contiguous();
    auto h_out = torch::empty_like(Abar_c);

    dim3 grid(num_rows);
    dim3 block(WARP_SIZE);

    warp_scan_kernel<<<grid, block>>>(
        Abar_c.data_ptr<float>(),
        Bbar_u_c.data_ptr<float>(),
        h_out.data_ptr<float>(),
        seq_len
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return h_out;
}

// ============================================================
// Kernel 2: multi-warp scan (seq_len <= 1024)
// ============================================================
__global__ void block_scan_kernel(
    const float* __restrict__ Abar,
    const float* __restrict__ Bbar_u,
    float* __restrict__ h_out,
    int padded_seq_len
) {
    __shared__ float warp_a[MAX_WARPS_PER_BLOCK];
    __shared__ float warp_b[MAX_WARPS_PER_BLOCK];
    __shared__ float carry_a[MAX_WARPS_PER_BLOCK];
    __shared__ float carry_b[MAX_WARPS_PER_BLOCK];

    int row = blockIdx.x;
    int t = threadIdx.x;
    int lane = t % WARP_SIZE;
    int warp_id = t / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    int idx = row * padded_seq_len + t;

    float a = Abar[idx];
    float b = Bbar_u[idx];

    // ---- Level 1: intra-warp inclusive scan ----
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float a_prev = __shfl_up_sync(0xFFFFFFFF, a, offset);
        float b_prev = __shfl_up_sync(0xFFFFFFFF, b, offset);
        if (lane >= offset) {
            float a_new, b_new;
            compose(a_prev, b_prev, a, b, a_new, b_new);
            a = a_new;
            b = b_new;
        }
    }

    // Last lane in each warp holds that warp's total - stash it in shared mem
    if (lane == WARP_SIZE - 1) {
        warp_a[warp_id] = a;
        warp_b[warp_id] = b;
    }
    __syncthreads();

    // ---- Level 2: exclusive scan over per-warp totals (small, done by
    // one thread; num_warps <= 32 so this sequential pass is cheap) ----
    if (t == 0) {
        float running_a = 1.0f;
        float running_b = 0.0f;
        for (int w = 0; w < num_warps; w++) {
            carry_a[w] = running_a;
            carry_b[w] = running_b;
            float new_a, new_b;
            compose(running_a, running_b, warp_a[w], warp_b[w], new_a, new_b);
            running_a = new_a;
            running_b = new_b;
        }
    }
    __syncthreads();

    // ---- Final: fold this warp's carry-in into the local result ----
    float final_a, final_b;
    compose(carry_a[warp_id], carry_b[warp_id], a, b, final_a, final_b);

    h_out[idx] = final_b;
}

torch::Tensor block_scan_forward(torch::Tensor Abar, torch::Tensor Bbar_u) {
    TORCH_CHECK(Abar.is_cuda(), "Abar must be a CUDA tensor");
    TORCH_CHECK(Bbar_u.is_cuda(), "Bbar_u must be a CUDA tensor");
    TORCH_CHECK(Abar.sizes() == Bbar_u.sizes(), "Abar and Bbar_u must have the same shape");
    TORCH_CHECK(Abar.dim() == 2, "expected flattened 2D input (rows, seq_len)");

    int num_rows = Abar.size(0);
    int seq_len = Abar.size(1);
    TORCH_CHECK(seq_len <= 1024, "block_scan_forward supports seq_len <= 1024 (32 warps/block max)");

    int padded_seq_len = ((seq_len + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    torch::Tensor Abar_padded, Bbar_u_padded;
    if (padded_seq_len != seq_len) {
        int pad_amount = padded_seq_len - seq_len;
        auto pad_a = torch::ones({num_rows, pad_amount}, Abar.options());
        auto pad_b = torch::zeros({num_rows, pad_amount}, Bbar_u.options());
        Abar_padded = torch::cat({Abar, pad_a}, 1).contiguous();
        Bbar_u_padded = torch::cat({Bbar_u, pad_b}, 1).contiguous();
    } else {
        Abar_padded = Abar.contiguous();
        Bbar_u_padded = Bbar_u.contiguous();
    }

    auto h_out_padded = torch::empty_like(Abar_padded);

    dim3 grid(num_rows);
    dim3 block(padded_seq_len);

    block_scan_kernel<<<grid, block>>>(
        Abar_padded.data_ptr<float>(),
        Bbar_u_padded.data_ptr<float>(),
        h_out_padded.data_ptr<float>(),
        padded_seq_len
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return h_out_padded.slice(1, 0, seq_len);
}

// ============================================================
// Kernel 3 (Stage 3): Fully fused discretization + scan + output projection
// ============================================================
/*
 * INTERVIEW PREP NOTE:
 * Unlike warp_scan_forward/block_scan_forward (which take PRE-COMPUTED
 * Abar/Bbar_u - meaning discretization already happened as separate,
 * separate-memory-traffic PyTorch ops), this kernel takes the RAW Mamba
 * inputs (u, delta, A, B, C, D) directly and does EVERYTHING in one
 * kernel launch:
 *   1. Loads delta[b,d,:] and u[b,d,:] into shared memory ONCE per
 *      (batch, d_inner) block - eliminating d_state-many redundant
 *      re-reads of the same data from global memory.
 *   2. For each state dimension n, computes Abar/Bbar_u on the fly in
 *      registers (never written to global memory), runs the same
 *      two-level warp+block scan as before, and accumulates the output
 *      projection y[t] += C[n,t] * h[t] directly.
 *   3. Only the final y (and optionally the skip connection D*u) is
 *      written to global memory - h, Abar, and Bbar_u never touch VRAM
 *      at all.
 *
 * This is the real "SRAM tiling" optimization Stage 3 is about: load
 * once, reuse across multiple computations, minimize HBM round trips.
 */

__global__ void fused_scan_kernel(
    const float* __restrict__ u,      // (batch, d_inner, seq_len)
    const float* __restrict__ delta,  // (batch, d_inner, seq_len)
    const float* __restrict__ A,      // (d_inner, d_state)
    const float* __restrict__ B,      // (batch, d_state, seq_len)
    const float* __restrict__ C,      // (batch, d_state, seq_len)
    const float* __restrict__ D,      // (d_inner,) or nullptr
    float* __restrict__ y_out,        // (batch, d_inner, seq_len)
    int d_inner, int d_state, int seq_len, int padded_seq_len,
    bool has_D
) {
    extern __shared__ float smem[];
    float* delta_s = smem;                        // [padded_seq_len]
    float* u_s     = smem + padded_seq_len;        // [padded_seq_len]
    float* warp_a  = u_s + padded_seq_len;          // [MAX_WARPS_PER_BLOCK]
    float* warp_b  = warp_a + MAX_WARPS_PER_BLOCK;
    float* carry_a = warp_b + MAX_WARPS_PER_BLOCK;
    float* carry_b = carry_a + MAX_WARPS_PER_BLOCK;

    int bd_idx = blockIdx.x;   // flattened (batch, d_inner) index
    int b = bd_idx / d_inner;
    int d = bd_idx % d_inner;

    int t = threadIdx.x;
    int lane = t % WARP_SIZE;
    int warp_id = t / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // ---- Load delta and u into shared memory ONCE for this (b,d) ----
    if (t < seq_len) {
        delta_s[t] = delta[(b * d_inner + d) * seq_len + t];
        u_s[t]     = u[(b * d_inner + d) * seq_len + t];
    } else {
        // Padding: delta=0 -> Abar=exp(0)=1 (identity), harmless
        delta_s[t] = 0.0f;
        u_s[t] = 0.0f;
    }
    __syncthreads();

    float y_acc = 0.0f;  // accumulates C.h across all state dimensions

    for (int n = 0; n < d_state; n++) {
        float A_dn = A[d * d_state + n];  // scalar for this (d,n), all t

        float a, bb;
        if (t < seq_len) {
            float delta_t = delta_s[t];
            float B_val = B[(b * d_state + n) * seq_len + t];
            a  = expf(delta_t * A_dn);            // Abar, computed on the fly
            bb = delta_t * B_val * u_s[t];         // Bbar_u, computed on the fly
        } else {
            a = 1.0f;
            bb = 0.0f;
        }

        // ---- Level 1: intra-warp scan (same as block_scan_kernel) ----
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            float a_prev = __shfl_up_sync(0xFFFFFFFF, a, offset);
            float b_prev = __shfl_up_sync(0xFFFFFFFF, bb, offset);
            if (lane >= offset) {
                float a_new, b_new;
                compose(a_prev, b_prev, a, bb, a_new, b_new);
                a = a_new;
                bb = b_new;
            }
        }

        if (lane == WARP_SIZE - 1) {
            warp_a[warp_id] = a;
            warp_b[warp_id] = bb;
        }
        __syncthreads();

        // ---- Level 2: carry across warps ----
        if (t == 0) {
            float running_a = 1.0f, running_b = 0.0f;
            for (int w = 0; w < num_warps; w++) {
                carry_a[w] = running_a;
                carry_b[w] = running_b;
                float new_a, new_b;
                compose(running_a, running_b, warp_a[w], warp_b[w], new_a, new_b);
                running_a = new_a;
                running_b = new_b;
            }
        }
        __syncthreads();

        float final_a, final_b;
        compose(carry_a[warp_id], carry_b[warp_id], a, bb, final_a, final_b);
        // final_b == h[t] for this state dimension n - lives only in a register

        if (t < seq_len) {
            float C_val = C[(b * d_state + n) * seq_len + t];
            y_acc += C_val * final_b;  // fused output projection, accumulated in-register
        }

        __syncthreads();  // ensure all threads finish reading shared warp/carry
                            // arrays before the next n iteration overwrites them
    }

    // ---- Skip connection + final write (the ONLY global memory write) ----
    if (t < seq_len) {
        if (has_D) {
            y_acc += D[d] * u_s[t];
        }
        y_out[(b * d_inner + d) * seq_len + t] = y_acc;
    }
}

torch::Tensor fused_scan_forward(
    torch::Tensor u, torch::Tensor delta, torch::Tensor A,
    torch::Tensor B, torch::Tensor C,
    c10::optional<torch::Tensor> D_opt
) {
    TORCH_CHECK(u.is_cuda() && delta.is_cuda() && A.is_cuda() && B.is_cuda() && C.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(u.dim() == 3, "u must be (batch, d_inner, seq_len)");
    TORCH_CHECK(A.dim() == 2, "A must be (d_inner, d_state)");
    TORCH_CHECK(B.dim() == 3 && C.dim() == 3, "B and C must be (batch, d_state, seq_len)");

    int batch = u.size(0);
    int d_inner = u.size(1);
    int seq_len = u.size(2);
    int d_state = A.size(1);

    TORCH_CHECK(A.size(0) == d_inner, "A's first dim must match d_inner");
    TORCH_CHECK(seq_len <= 1024, "fused_scan_forward supports seq_len <= 1024");

    int padded_seq_len = ((seq_len + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    auto u_c = u.contiguous();
    auto delta_c = delta.contiguous();
    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    auto C_c = C.contiguous();

    bool has_D = D_opt.has_value();
    torch::Tensor D_c;
    if (has_D) {
        D_c = D_opt.value().contiguous();
        TORCH_CHECK(D_c.size(0) == d_inner, "D must have shape (d_inner,)");
    }

    auto y_out = torch::empty({batch, d_inner, seq_len}, u.options());

    dim3 grid(batch * d_inner);
    dim3 block(padded_seq_len);

    size_t smem_bytes = (2 * padded_seq_len + 4 * MAX_WARPS_PER_BLOCK) * sizeof(float);

    fused_scan_kernel<<<grid, block, smem_bytes>>>(
        u_c.data_ptr<float>(),
        delta_c.data_ptr<float>(),
        A_c.data_ptr<float>(),
        B_c.data_ptr<float>(),
        C_c.data_ptr<float>(),
        has_D ? D_c.data_ptr<float>() : nullptr,
        y_out.data_ptr<float>(),
        d_inner, d_state, seq_len, padded_seq_len, has_D
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return y_out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warp_scan_forward", &warp_scan_forward, "Warp-level parallel associative scan (CUDA), seq_len<=32");
    m.def("block_scan_forward", &block_scan_forward, "Multi-warp parallel associative scan (CUDA), seq_len<=1024");
    m.def("fused_scan_forward", &fused_scan_forward, "Fully fused discretization+scan+output-projection (CUDA), seq_len<=1024",
          py::arg("u"), py::arg("delta"), py::arg("A"), py::arg("B"), py::arg("C"), py::arg("D") = py::none());
}
