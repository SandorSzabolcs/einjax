"""
Tensor representations for EinJAX.

Provides BaseTensor ABC, DenseTensor, SparseTensor, and
SparseTensorRelation for tensor-relational sparse representation.
"""

from .base import BaseTensor, IndexedTerm
from .dense import DenseTensor
from .sparse import SparseTensor, SparseTensorRelation
from .stats import (
    compute_sparsity_stats_coo,
    compute_sparsity_stats_dense,
    sparsity_ratio,
    update_scheme_sparsity,
)
from .tiling import (
    prune_infeasible_schemes,
    rank_schemes_by_sparsity,
    select_best_sparse_tiling,
    get_sparse_partition_spec,
    compute_tile_memory,
    compute_relation_memory,
    filter_schemes_by_sharding,
    filter_schemes_by_device_count,
    prepare_sparse_tiling,
)

__all__ = [
    "BaseTensor",
    "IndexedTerm",
    "DenseTensor",
    "SparseTensor",
    "SparseTensorRelation",
    "compute_sparsity_stats_coo",
    "compute_sparsity_stats_dense",
    "sparsity_ratio",
    "update_scheme_sparsity",
    "prune_infeasible_schemes",
    "rank_schemes_by_sparsity",
    "select_best_sparse_tiling",
    "get_sparse_partition_spec",
    "compute_tile_memory",
    "compute_relation_memory",
    "filter_schemes_by_sharding",
    "filter_schemes_by_device_count",
    "prepare_sparse_tiling",
]
