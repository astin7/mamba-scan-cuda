"""
Stage 1: Mamba Selective Scan - Pure PyTorch Reference Implementation
========================================================================
Purpose: establish a mathematically correct, ground-truth baseline that
the Stage 2 CUDA kernel must reproduce (within numerical tolerance).

This is intentionally the "slow" version - a straightforward sequential
loop, no parallelism tricks. Stage 2 will replace the loop with a
CUDA-based parallel associative scan, but must produce the same outputs.

INTERVIEW PREP NOTE: read every comment block below before an interview.
Each one explains *why*, not just *what*.
"""

import torch
import torch.nn.functional as F


def selective_scan_simple(u, delta, A, B, C, D=None):
    """
    Core Mamba recurrence:
        h[t] = Abar[t] * h[t-1] + Bbar_u[t]
        y[t] = C[t] . h[t]

    where:
        Abar = exp(delta * A)          <- exact ZOH discretization of A
        Bbar ~= delta * B              <- first-order (simplified) approx
                                            of the ZOH discretization of B
                                            (this is what the *real* Mamba
                                            implementation uses too - see
                                            selective_scan_ref in the
                                            official repo)

    Shapes:
        u:     (batch, d_inner, seq_len)   input sequence
        delta: (batch, d_inner, seq_len)   input-dependent step size (>0)
        A:     (d_inner, d_state)          fixed, should be negative
                                            (negative = stable/decaying
                                            state; positive would blow up)
        B:     (batch, d_state, seq_len)   input-dependent
        C:     (batch, d_state, seq_len)   input-dependent
        D:     (d_inner,) or None          optional skip/feedthrough term

    Returns:
        out:   (batch, d_inner, seq_len)
    """
    batch, d_inner, seq_len = u.shape
    d_state = A.shape[1]

    # ---- Step 1: discretize A for every timestep at once (vectorized) ----
    # einsum meaning: for each (batch b, channel d, time l), multiply
    # delta[b,d,l] by every entry of A[d,n], producing a (b,d,l,n) tensor.
    # Then exp() turns it into Abar. This matches deltaA in the official
    # selective_scan_ref.
    Abar = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))  # (batch, d_inner, seq_len, d_state)

    # ---- Step 2: precompute Bbar * u for every timestep at once ----
    # Using the simplified approximation Bbar ~= delta * B (not the exact
    # ZOH inverse formula - this is the standard practical choice, and
    # matches deltaB_u in the official reference).
    Bbar_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)  # (batch, d_inner, seq_len, d_state)

    # ---- Step 3: the sequential recurrence (the actual bottleneck) ----
    # This loop is *why* Mamba is sequential and *why* Stage 2 exists:
    # h[t] depends on h[t-1], so timesteps cannot be computed independently
    # without a smarter algorithm (parallel associative scan - Stage 2).
    h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(seq_len):
        h = Abar[:, :, t, :] * h + Bbar_u[:, :, t, :]     # state update
        y_t = torch.einsum('bdn,bn->bd', h, C[:, :, t])   # read out output
        ys.append(y_t)

    y = torch.stack(ys, dim=2)  # (batch, d_inner, seq_len)

    # ---- Step 4: optional skip connection ----
    # D provides a direct input->output path, bypassing the state entirely.
    # Real Mamba always uses this; it's a learned per-channel scalar.
    if D is not None:
        y = y + u * D.unsqueeze(-1)

    return y


def generate_test_case(batch=2, d_inner=8, d_state=4, seq_len=16, seed=0):
    """
    Generates a reproducible random test case and saves it to disk.
    This becomes the ground-truth reference Stage 2's CUDA kernel must match.
    """
    torch.manual_seed(seed)

    u = torch.randn(batch, d_inner, seq_len)

    # delta must be positive (it's a timestep size) - softplus guarantees
    # this, exactly like the real Mamba implementation does.
    delta_raw = torch.randn(batch, d_inner, seq_len)
    delta = F.softplus(delta_raw)

    # A must be negative for a stable (decaying, not exploding) state.
    # Real Mamba stores A in log-space and negates: A = -exp(A_log).
    A_log = torch.randn(d_inner, d_state)
    A = -torch.exp(A_log)

    B = torch.randn(batch, d_state, seq_len)
    C = torch.randn(batch, d_state, seq_len)
    D = torch.randn(d_inner)

    out = selective_scan_simple(u, delta, A, B, C, D)

    torch.save(
        {"u": u, "delta": delta, "A": A, "B": B, "C": C, "D": D, "out": out},
        "stage1_test_case.pt",
    )

    print("Test case generated and saved to stage1_test_case.pt")
    print(f"  u:     {u.shape}")
    print(f"  delta: {delta.shape}")
    print(f"  A:     {A.shape}")
    print(f"  B:     {B.shape}")
    print(f"  C:     {C.shape}")
    print(f"  D:     {D.shape}")
    print(f"  out:   {out.shape}")
    print(f"  out sample (first 5 values, batch 0, channel 0):")
    print(f"    {out[0, 0, :5]}")
    print(f"  out contains NaN: {torch.isnan(out).any().item()}")

    return u, delta, A, B, C, D, out


if __name__ == "__main__":
    generate_test_case()