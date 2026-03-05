"""
Kernel dispatch registry for EinJAX.

Per PRD Section 7.3, the KernelRegistry dispatches einsum patterns to
the best available kernel implementation:

Lookup order:
1. Pallas kernel for current backend (GPU Triton or TPU Mosaic)
2. Generic JAX fallback (jnp.einsum + segment_sum)

Patterns are matched by: operation type, sparsity, backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import jax

from einjax.tensor.sparse import SparseTensorRelation
from einjax.execution.sparse_dispatch import execute_sparse
from einjax.execution.dense_kernels import execute_dense_einsum
from einjax.kernels.pallas_matmul import block_sparse_matmul_generic


@dataclass
class KernelInfo:
    """Metadata for a registered kernel.

    Attributes:
        name: Human-readable kernel name.
        pattern: Einsum pattern this kernel handles (e.g., "ij,jk->ik").
        backend: Backend requirement ("generic", "gpu", "tpu", "gpu:triton", etc.).
        sparsity: Sparsity mode ("dense", "sparse", "any").
        priority: Higher priority kernels are preferred. Pallas kernels
            should have higher priority than generic fallbacks.
        kernel_fn: The callable that implements the kernel.
    """
    name: str
    pattern: str
    backend: str
    sparsity: str
    priority: int
    kernel_fn: Callable[..., Any]


class KernelRegistry:
    """Dispatch einsum patterns to optimized kernel implementations.

    Lookup order:
    1. Pallas kernel for current backend (GPU Triton or TPU Mosaic)
    2. Generic JAX fallback (jnp.einsum + segment_sum)

    Patterns are matched by: operation type, sparsity, backend.

    Usage:
        registry = KernelRegistry()
        kernel_fn = registry.lookup("ij,jk->ik", backend="gpu", sparsity="sparse")
        result = kernel_fn(lhs, rhs, ...)
    """

    def __init__(self) -> None:
        self._kernels: list[KernelInfo] = []
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register the built-in generic kernels."""
        # Generic sparse matmul fallback
        self.register(KernelInfo(
            name="generic_sparse_matmul",
            pattern="ij,jk->ik",
            backend="generic",
            sparsity="sparse",
            priority=0,
            kernel_fn=block_sparse_matmul_generic,
        ))

        # Generic sparse execution for any pattern
        self.register(KernelInfo(
            name="generic_sparse_einsum",
            pattern="*",
            backend="generic",
            sparsity="sparse",
            priority=-1,
            kernel_fn=execute_sparse,
        ))

        # Generic dense execution for any pattern
        self.register(KernelInfo(
            name="generic_dense_einsum",
            pattern="*",
            backend="generic",
            sparsity="dense",
            priority=0,
            kernel_fn=execute_dense_einsum,
        ))

    def register(self, kernel: KernelInfo) -> None:
        """Register a kernel implementation.

        Args:
            kernel: KernelInfo with pattern, backend, and callable.
        """
        self._kernels.append(kernel)

    def lookup(
        self,
        pattern: str,
        backend: str | None = None,
        sparsity: str = "dense",
    ) -> Callable[..., Any]:
        """Find best kernel for the given einsum pattern and backend.

        Args:
            pattern: Einsum notation (e.g., "ij,jk->ik").
            backend: Backend identifier ("gpu", "tpu", "cpu", "generic").
                If None, auto-detected from current JAX backend.
            sparsity: "dense" or "sparse".

        Returns:
            Callable kernel function.

        Raises:
            ValueError: If no matching kernel is found.
        """
        if backend is None:
            backend = _detect_backend()

        candidates = self._find_candidates(pattern, backend, sparsity)

        if not candidates:
            raise ValueError(
                f"No kernel found for pattern={pattern!r}, "
                f"backend={backend!r}, sparsity={sparsity!r}"
            )

        # Sort by priority descending, return highest
        candidates.sort(key=lambda k: k.priority, reverse=True)
        return candidates[0].kernel_fn

    def lookup_info(
        self,
        pattern: str,
        backend: str | None = None,
        sparsity: str = "dense",
    ) -> KernelInfo:
        """Find best kernel info for the given pattern and backend.

        Like lookup() but returns the full KernelInfo instead of just
        the callable.

        Args:
            pattern: Einsum notation.
            backend: Backend identifier.
            sparsity: "dense" or "sparse".

        Returns:
            KernelInfo for the best matching kernel.

        Raises:
            ValueError: If no matching kernel is found.
        """
        if backend is None:
            backend = _detect_backend()

        candidates = self._find_candidates(pattern, backend, sparsity)

        if not candidates:
            raise ValueError(
                f"No kernel found for pattern={pattern!r}, "
                f"backend={backend!r}, sparsity={sparsity!r}"
            )

        candidates.sort(key=lambda k: k.priority, reverse=True)
        return candidates[0]

    def list_kernels(
        self,
        pattern: str | None = None,
        backend: str | None = None,
        sparsity: str | None = None,
    ) -> list[KernelInfo]:
        """List all registered kernels, optionally filtered.

        Args:
            pattern: Filter by pattern (exact match or "*" for wildcard).
            backend: Filter by backend.
            sparsity: Filter by sparsity mode.

        Returns:
            List of matching KernelInfo objects.
        """
        result = list(self._kernels)

        if pattern is not None:
            result = [k for k in result if k.pattern == pattern or k.pattern == "*"]

        if backend is not None:
            result = [k for k in result if _backend_matches(k.backend, backend)]

        if sparsity is not None:
            result = [k for k in result if k.sparsity == sparsity or k.sparsity == "any"]

        return result

    def _find_candidates(
        self,
        pattern: str,
        backend: str,
        sparsity: str,
    ) -> list[KernelInfo]:
        """Find all kernels matching the given criteria."""
        candidates = []
        for k in self._kernels:
            if not _pattern_matches(k.pattern, pattern):
                continue
            if not _backend_matches(k.backend, backend):
                continue
            if not _sparsity_matches(k.sparsity, sparsity):
                continue
            candidates.append(k)
        return candidates


def _detect_backend() -> str:
    """Detect the current JAX backend.

    Returns:
        "gpu", "tpu", or "cpu".
    """
    try:
        devices = jax.devices()
        if devices:
            platform = devices[0].platform
            if platform == "gpu":
                return "gpu"
            elif platform == "tpu":
                return "tpu"
    except Exception:
        pass
    return "cpu"


def _pattern_matches(registered: str, query: str) -> bool:
    """Check if a registered pattern matches a query pattern.

    "*" matches any pattern. Otherwise, exact string match.
    """
    if registered == "*":
        return True
    return registered == query


def _backend_matches(registered: str, query: str) -> bool:
    """Check if a registered backend matches a query backend.

    "generic" matches any backend. "gpu" matches "gpu:triton", etc.
    """
    if registered == "generic":
        return True
    if registered == query:
        return True
    if query.startswith(registered + ":"):
        return True
    return False


def _sparsity_matches(registered: str, query: str) -> bool:
    """Check if a registered sparsity mode matches a query."""
    if registered == "any":
        return True
    return registered == query
