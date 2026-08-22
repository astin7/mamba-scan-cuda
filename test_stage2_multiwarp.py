"""
Stage 2 Multi-Warp Verification: tests block_scan_forward against a plain
sequential PyTorch loop, using sequence lengths that exceed a single warp
(32 timesteps) - this is what actually gets exercised: the two-level
warp+block scan logic, padding, and carry propagation across warps.
"""

import torch
import mamba_scan_cuda


def run_test(num_rows, seq_len, label):
    torch.manual_seed(0)

    delta = torch.rand(num_rows, seq_len, device="cuda") * 0.1
    A = -torch.rand(num_rows, 1, device="cuda")
    Abar = torch.exp(delta * A)
    Bbar_u = torch.randn(num_rows, seq_len, device="cuda") * 0.5

    # Sequential reference
    h_ref = torch.zeros(num_rows, device="cuda")
    h_ref_all = torch.zeros(num_rows, seq_len, device="cuda")
    for t in range(seq_len):
        h_ref = Abar[:, t] * h_ref + Bbar_u[:, t]
        h_ref_all[:, t] = h_ref

    h_cuda = mamba_scan_cuda.block_scan_forward(Abar, Bbar_u)

    match = torch.allclose(h_ref_all, h_cuda, atol=1e-3, rtol=1e-3)
    max_diff = (h_ref_all - h_cuda).abs().max().item()

    status = "PASS" if match else "FAIL"
    print(f"[{status}] {label} (rows={num_rows}, seq_len={seq_len}) "
          f"max_diff={max_diff:.8f}")
    return match


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(1)

    results = []
    # Exactly one warp (sanity check against the simpler kernel's territory)
    results.append(run_test(50, 32, "exactly one warp"))
    # Multiple full warps
    results.append(run_test(50, 128, "4 full warps"))
    # Not a multiple of 32 - exercises the padding logic
    results.append(run_test(50, 100, "non-multiple-of-32 (padding test)"))
    # Near the max supported size
    results.append(run_test(20, 1000, "near max (1000, 32 warps)"))
    # Odd/small edge case
    results.append(run_test(10, 1, "single timestep edge case"))

    print()
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - see above")
