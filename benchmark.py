"""
Stage 4: Benchmarking across implementations and sequence lengths.

Compares:
  1. Naive sequential Python loop (the true "no GPU parallelism at all"
     baseline - everything computed one timestep at a time, no vectorized
     precompute even).
  2. Stage 1 reference (vectorized precompute of Abar/Bbar_u, but still a
     sequential Python-level loop for the recurrence itself).
  3. Stage 2 CUDA (block_scan_forward - parallel associative scan, but
     Abar/Bbar_u precomputed as separate PyTorch ops beforehand).
  4. Stage 3 CUDA (fused_scan_forward - fully fused, single kernel launch,
     raw inputs to output).

Produces a latency-vs-sequence-length plot (log scale) - this is the
"publication-grade" chart the project plan calls for.
"""

import time
import torch
import matplotlib.pyplot as plt
import mamba_scan_cuda

from stage1_reference import selective_scan_simple


def naive_python_loop(u, delta, A, B, C, D):
    """
    True naive baseline: no vectorized precompute at all, everything
    recomputed timestep by timestep in pure Python/PyTorch, on GPU tensors
    but with Python-level loop overhead for EVERY operation, not just the
    recurrence. This is deliberately the slowest possible correct
    implementation, for a fair "before" comparison.
    """
    batch, d_inner, seq_len = u.shape
    d_state = A.shape[1]
    h = torch.zeros(batch, d_inner, d_state, device=u.device)
    ys = []
    for t in range(seq_len):
        Abar_t = torch.exp(delta[:, :, t].unsqueeze(-1) * A.unsqueeze(0))
        Bbar_u_t = (delta[:, :, t].unsqueeze(-1) * B[:, :, t].unsqueeze(1) * u[:, :, t].unsqueeze(-1))
        h = Abar_t * h + Bbar_u_t
        y_t = torch.einsum('bdn,bn->bd', h, C[:, :, t])
        ys.append(y_t)
    y = torch.stack(ys, dim=2)
    if D is not None:
        y = y + u * D.unsqueeze(-1)
    return y


def time_fn(fn, *args, warmup=3, iters=10):
    """GPU timing done correctly: warmup iterations (to avoid measuring
    one-time CUDA context/kernel-compilation overhead), then torch.cuda
    Events for accurate GPU-side timing (wall-clock time.time() around
    async CUDA calls would be misleading without a sync)."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # milliseconds per call


def make_test_case(batch, d_inner, d_state, seq_len, seed=0):
    torch.manual_seed(seed)
    u = torch.randn(batch, d_inner, seq_len, device="cuda")
    delta = torch.nn.functional.softplus(torch.randn(batch, d_inner, seq_len, device="cuda"))
    A = -torch.exp(torch.randn(d_inner, d_state, device="cuda"))
    B = torch.randn(batch, d_state, seq_len, device="cuda")
    C = torch.randn(batch, d_state, seq_len, device="cuda")
    D = torch.randn(d_inner, device="cuda")
    return u, delta, A, B, C, D


def stage2_wrapper(u, delta, A, B, C, D):
    """Stage 2 takes precomputed Abar/Bbar_u - compute those here (as
    separate ops, representing the un-fused pipeline) then call the kernel."""
    batch, d_inner, seq_len = u.shape
    d_state = A.shape[1]
    Abar = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    Bbar_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    # Flatten (batch, d_inner, d_state) -> rows for the kernel's expected shape
    Abar_flat = Abar.permute(0, 1, 3, 2).reshape(batch * d_inner * d_state, seq_len)
    Bbar_u_flat = Bbar_u.permute(0, 1, 3, 2).reshape(batch * d_inner * d_state, seq_len)
    h = mamba_scan_cuda.block_scan_forward(Abar_flat, Bbar_u_flat)
    h = h.reshape(batch, d_inner, d_state, seq_len)
    y = torch.einsum('bdnl,bnl->bdl', h, C)
    if D is not None:
        y = y + u * D.unsqueeze(-1)
    return y


def run_benchmark():
    seq_lengths = [16, 32, 64, 128, 256, 512, 1000]
    batch, d_inner, d_state = 4, 16, 8

    results = {"naive_loop": [], "stage1_ref": [], "stage2_cuda": [], "stage3_fused": []}

    for seq_len in seq_lengths:
        print(f"Benchmarking seq_len={seq_len}...")
        u, delta, A, B, C, D = make_test_case(batch, d_inner, d_state, seq_len)

        t_naive = time_fn(naive_python_loop, u, delta, A, B, C, D, iters=5 if seq_len <= 128 else 2)
        t_stage1 = time_fn(selective_scan_simple, u, delta, A, B, C, D)
        t_stage2 = time_fn(stage2_wrapper, u, delta, A, B, C, D)
        t_stage3 = time_fn(mamba_scan_cuda.fused_scan_forward, u, delta, A, B, C, D)

        results["naive_loop"].append(t_naive)
        results["stage1_ref"].append(t_stage1)
        results["stage2_cuda"].append(t_stage2)
        results["stage3_fused"].append(t_stage3)

        print(f"  naive_loop={t_naive:.3f}ms  stage1_ref={t_stage1:.3f}ms  "
              f"stage2_cuda={t_stage2:.3f}ms  stage3_fused={t_stage3:.3f}ms")

    # ---- Plot ----
    plt.figure(figsize=(9, 6))
    plt.plot(seq_lengths, results["naive_loop"], marker='o', label="Naive Python loop")
    plt.plot(seq_lengths, results["stage1_ref"], marker='o', label="Stage 1: PyTorch reference")
    plt.plot(seq_lengths, results["stage2_cuda"], marker='o', label="Stage 2: CUDA parallel scan")
    plt.plot(seq_lengths, results["stage3_fused"], marker='o', label="Stage 3: Fused CUDA kernel")
    plt.xlabel("Sequence Length")
    plt.ylabel("Latency (ms, log scale)")
    plt.yscale("log")
    plt.title(f"Mamba Selective Scan: Latency vs Sequence Length\n(batch={batch}, d_inner={d_inner}, d_state={d_state}, RTX 3090)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_latency.png", dpi=150)
    print("\nSaved plot to benchmark_latency.png")

    # ---- Speedup summary ----
    print("\n--- Speedup Summary (Stage 3 fused vs. others, at largest seq_len) ---")
    idx = -1
    print(f"seq_len = {seq_lengths[idx]}")
    print(f"  Stage 3 vs naive loop:  {results['naive_loop'][idx] / results['stage3_fused'][idx]:.1f}x faster")
    print(f"  Stage 3 vs Stage 1:     {results['stage1_ref'][idx] / results['stage3_fused'][idx]:.1f}x faster")
    print(f"  Stage 3 vs Stage 2:     {results['stage2_cuda'][idx] / results['stage3_fused'][idx]:.1f}x faster")

    return results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available.")
    else:
        run_benchmark()
