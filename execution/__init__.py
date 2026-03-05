"""
Execution engine for EinJAX.

Provides dense kernel execution for single-device einsum operations,
and the ExecutionEngine for multi-stage DAG execution with topological
sort and resharding.
"""

from .dense_kernels import execute_dense_einsum, execute_sharded_einsum
from .engine import ExecutionEngine, build_dependency_graph, topological_sort
from .sparse_dispatch import (
    execute_sparse,
    execute_sharded_sparse,
    coordinate_join,
    kernel_einsum,
    segment_sum,
)

__all__ = [
    "execute_dense_einsum",
    "execute_sharded_einsum",
    "ExecutionEngine",
    "build_dependency_graph",
    "topological_sort",
    "execute_sparse",
    "execute_sharded_sparse",
    "coordinate_join",
    "kernel_einsum",
    "segment_sum",
]
