"""
Kernel dispatch and optimized kernel implementations for EinJAX.

Provides a KernelRegistry that dispatches einsum patterns to the best
available kernel: Pallas (GPU Triton / TPU Mosaic) when available,
or generic JAX (jnp.einsum + segment_sum) as fallback.
"""

from .registry import KernelRegistry, KernelInfo
from .pallas_matmul import block_sparse_matmul, block_sparse_matmul_generic
from .pallas_gather import coordinate_join_sorted, coordinate_join_hash

__all__ = [
    "KernelRegistry",
    "KernelInfo",
    "block_sparse_matmul",
    "block_sparse_matmul_generic",
    "coordinate_join_sorted",
    "coordinate_join_hash",
]
