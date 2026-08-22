"""
Minimal script for Nsight Compute to profile. Deliberately simple - a
single, isolated call to the fused kernel, so the profiler's output is
focused on just this kernel's execution, not benchmark loop overhead.
"""

import torch
import mamba_scan_cuda

import sys

torch.manual_seed(0)
# Default: small size (matches the first profiling run). Pass "large" as an
# arg to use a more production-realistic size instead, to compare occupancy.
if len(sys.argv) > 1 and sys.argv[1] == "large":
    batch, d_inner, d_state, seq_len = 32, 512, 16, 512
else:
    batch, d_inner, d_state, seq_len = 4, 16, 8, 512

u = torch.randn(batch, d_inner, seq_len, device="cuda")
delta = torch.nn.functional.softplus(torch.randn(batch, d_inner, seq_len, device="cuda"))
A = -torch.exp(torch.randn(d_inner, d_state, device="cuda"))
B = torch.randn(batch, d_state, seq_len, device="cuda")
C = torch.randn(batch, d_state, seq_len, device="cuda")
D = torch.randn(d_inner, device="cuda")

# Warmup (so we're not profiling one-time CUDA context setup)
for _ in range(3):
    y = mamba_scan_cuda.fused_scan_forward(u, delta, A, B, C, D)
torch.cuda.synchronize()

# The actual call Nsight Compute will capture
y = mamba_scan_cuda.fused_scan_forward(u, delta, A, B, C, D)
torch.cuda.synchronize()

print("Kernel executed, output shape:", y.shape)
