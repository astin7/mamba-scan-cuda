"""
Build script for the mamba_scan_cuda extension.

Usage:
    python3 setup.py build_ext --inplace

This compiles mamba_scan_kernel.cu with nvcc and produces a Python-importable
.so extension module (mamba_scan_cuda), using PyTorch's built-in CUDAExtension
helper - this handles the PyBind11 + torch tensor integration boilerplate for
us (same underlying mechanism as Apex-LOB's PyBind11 bridge, just using
PyTorch's convenience wrapper around it since we're binding torch::Tensor
types specifically).
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mamba_scan_cuda",
    ext_modules=[
        CUDAExtension(
            name="mamba_scan_cuda",
            sources=["mamba_scan_kernel.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
