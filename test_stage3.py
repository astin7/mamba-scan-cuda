"""
Stage 3 Verification: tests fused_scan_forward against the REAL Stage 1
saved output (stage1_test_case.pt) - this is the most meaningful test in
the whole project, since it validates the entire chain end-to-end: raw
Mamba inputs -> fully fused CUDA kernel -> output that matches the
original pure-PyTorch mathematical reference.
"""

import torch
import mamba_scan_cuda


def test_against_stage1_reference():
    data = torch.load("stage1_test_case.pt")

    u = data["u"].cuda()
    delta = data["delta"].cuda()
    A = data["A"].cuda()
    B = data["B"].cuda()
    C = data["C"].cuda()
    D = data["D"].cuda()
    expected_out = data["out"].cuda()

    print(f"Loaded stage1_test_case.pt: batch={u.shape[0]}, d_inner={u.shape[1]}, "
          f"seq_len={u.shape[2]}, d_state={A.shape[1]}")

    y = mamba_scan_cuda.fused_scan_forward(u, delta, A, B, C, D)

    match = torch.allclose(y, expected_out, atol=1e-3, rtol=1e-3)
    max_diff = (y - expected_out).abs().max().item()

    print(f"Max absolute difference vs Stage 1 reference: {max_diff:.8f}")
    print(f"torch.allclose (atol=1e-3, rtol=1e-3): {match}")

    if match:
        print("PASS: fully fused CUDA kernel matches the ORIGINAL Stage 1 reference.")
    else:
        print("FAIL: divergence beyond tolerance.")

    return match


def test_larger_multiwarp_case():
    """A bigger, randomly generated case to exercise the multi-warp path
    inside the fused kernel specifically (stage1_test_case.pt uses a short
    seq_len that only exercises a single warp)."""
    torch.manual_seed(42)

    batch, d_inner, d_state, seq_len = 3, 6, 8, 200

    u = torch.randn(batch, d_inner, seq_len, device="cuda")
    delta = torch.nn.functional.softplus(torch.randn(batch, d_inner, seq_len, device="cuda"))
    A = -torch.exp(torch.randn(d_inner, d_state, device="cuda"))
    B = torch.randn(batch, d_state, seq_len, device="cuda")
    C = torch.randn(batch, d_state, seq_len, device="cuda")
    D = torch.randn(d_inner, device="cuda")

    # Reference: same sequential-loop math as Stage 1's selective_scan_simple
    Abar = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    Bbar_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    h = torch.zeros(batch, d_inner, d_state, device="cuda")
    ys = []
    for t in range(seq_len):
        h = Abar[:, :, t, :] * h + Bbar_u[:, :, t, :]
        y_t = torch.einsum('bdn,bn->bd', h, C[:, :, t])
        ys.append(y_t)
    y_ref = torch.stack(ys, dim=2) + u * D.unsqueeze(-1)

    y_cuda = mamba_scan_cuda.fused_scan_forward(u, delta, A, B, C, D)

    match = torch.allclose(y_ref, y_cuda, atol=1e-3, rtol=1e-3)
    max_diff = (y_ref - y_cuda).abs().max().item()

    print(f"\nLarger multi-warp case: batch={batch}, d_inner={d_inner}, "
          f"d_state={d_state}, seq_len={seq_len}")
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"torch.allclose: {match}")
    print("PASS" if match else "FAIL")

    return match


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit(1)

    r1 = test_against_stage1_reference()
    r2 = test_larger_multiwarp_case()

    print()
    print("ALL STAGE 3 TESTS PASSED" if (r1 and r2) else "SOME STAGE 3 TESTS FAILED")
