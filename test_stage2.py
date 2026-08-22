"""
Stage 2 Verification: compare the CUDA warp-scan kernel's output against the
Stage 1 PyTorch reference implementation, using a numerical tolerance check
(NOT exact equality) - see the project plan notes on why exact bit-for-bit
matching is the wrong verification method for parallel floating-point ops.

This test uses a SHORT sequence (<=32 timesteps) since Stage 2's first
version only supports warp-level (single-warp) scans. Multi-warp chaining
for longer sequences comes in the next step.
"""

import torch
import mamba_scan_cuda  # the compiled extension


def test_warp_scan_correctness():
    torch.manual_seed(0)

    # Small test case that fits in a single warp: seq_len <= 32.
    # We work directly with flattened (rows, seq_len) tensors here, since
    # the kernel itself is shape-agnostic about what "row" means (it could
    # be a (batch, channel, state) triple - flattening is the caller's job).
    num_rows = 100   # e.g. batch * d_inner * d_state, flattened
    seq_len = 16

    # Abar must represent a real discretized state-transition scalar,
    # exp(delta * A) with A < 0, so Abar is in (0, 1) - mimics real values.
    delta = torch.rand(num_rows, seq_len, device="cuda") * 0.1
    A = -torch.rand(num_rows, 1, device="cuda")
    Abar = torch.exp(delta * A)  # (num_rows, seq_len), values in (0,1)

    Bbar_u = torch.randn(num_rows, seq_len, device="cuda") * 0.5

    # ---- Reference: plain sequential PyTorch loop (same math, different
    # implementation) - this mirrors exactly what Stage 1's loop does, just
    # scoped to this flattened (row, seq_len) shape instead of the full
    # Mamba tensor shapes, so we can compare apples to apples against the
    # kernel's flattened row format.
    h_ref = torch.zeros(num_rows, device="cuda")
    h_ref_all = torch.zeros(num_rows, seq_len, device="cuda")
    for t in range(seq_len):
        h_ref = Abar[:, t] * h_ref + Bbar_u[:, t]
        h_ref_all[:, t] = h_ref

    # ---- CUDA kernel result ----
    h_cuda = mamba_scan_cuda.warp_scan_forward(Abar, Bbar_u)

    # ---- Compare with tolerance, not exact equality ----
    match = torch.allclose(h_ref_all, h_cuda, atol=1e-4, rtol=1e-4)
    max_diff = (h_ref_all - h_cuda).abs().max().item()

    print(f"Shapes -> ref: {h_ref_all.shape}, cuda: {h_cuda.shape}")
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"torch.allclose (atol=1e-4, rtol=1e-4): {match}")

    if match:
        print("PASS: CUDA kernel matches PyTorch reference within tolerance.")
    else:
        print("FAIL: outputs diverge beyond tolerance - see max_diff above.")

    return match


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available - cannot run this test.")
    else:
        test_warp_scan_correctness()
