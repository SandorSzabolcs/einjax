"""
EinJAX: Upper-Case-Lower-Case EinSum on JAX

Native JAX implementation of the upper-case-lower-case EinSum system
from the VLDB 2026 paper "Automated Tensor-Relational Decomposition
for Large-Scale Sparse Tensor Computation".
"""

from .core.notation import (
    index_to_subscript,
    find_all_factors,
    get_label_dimensions,
    normalize_notation,
    validate_inputs,
)
from .core.types import (
    Expr,
    UnaryOp,
    BinaryOp,
    Constant,
    TilingScheme,
    CaseAssignment,
)
from .tensor.base import BaseTensor, IndexedTerm
from .tensor.dense import DenseTensor
from .tensor.sparse import SparseTensor, SparseTensorRelation
from .tensor.stats import (
    compute_sparsity_stats_coo,
    compute_sparsity_stats_dense,
    sparsity_ratio,
    update_scheme_sparsity,
)
from .tensor.tiling import (
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
from .optimizer.cost_model import (
    CostModelConfig,
    detect_device_config,
    calibrate,
    compute_join_cost,
)
from .optimizer.dp import DPOptimizer, ReductionInfo, infer_reduction_info
from .optimizer.contraction_path import (
    ContractionStep,
    ContractionPlan,
    get_contraction_order,
    plan_contraction,
)
from .sharding.partition import tile_shape_to_partition_spec, derive_partition_specs
from .sharding.mesh import create_mesh, infer_mesh_shape
from .sharding.reshard import (
    needs_reshard,
    estimate_reshard_bytes,
    estimate_reshard_cost,
    reshard_dense,
    plan_reshard_sequence,
)
from .execution.dense_kernels import execute_dense_einsum, execute_sharded_einsum
from .execution.engine import ExecutionEngine, build_dependency_graph, topological_sort
from .execution.sparse_dispatch import (
    execute_sparse,
    execute_sharded_sparse,
    coordinate_join,
    kernel_einsum,
    segment_sum,
)
from .kernels.registry import KernelRegistry, KernelInfo
from .kernels.pallas_matmul import block_sparse_matmul, block_sparse_matmul_generic
from .kernels.pallas_gather import coordinate_join_sorted, coordinate_join_hash
from .autodiff.custom_vjp import sparse_einsum, sparse_einsum_raw
from .config import (
    get_config,
    set_config,
    reset_config,
    list_device_types,
    get_hardware_profile,
)
from .api import einsum, analyze, AnalysisResult, with_mesh

__version__ = "0.1.0"
__all__ = [
    # Utility functions
    "index_to_subscript",
    "find_all_factors",
    "get_label_dimensions",
    "normalize_notation",
    "validate_inputs",
    # Expression classes
    "Expr",
    "UnaryOp",
    "BinaryOp",
    "Constant",
    # Tiling
    "TilingScheme",
    "CaseAssignment",
    # Tensor classes
    "BaseTensor",
    "IndexedTerm",
    "DenseTensor",
    "SparseTensor",
    "SparseTensorRelation",
    # Sparsity statistics
    "compute_sparsity_stats_coo",
    "compute_sparsity_stats_dense",
    "sparsity_ratio",
    "update_scheme_sparsity",
    # Sparse tiling
    "prune_infeasible_schemes",
    "rank_schemes_by_sparsity",
    "select_best_sparse_tiling",
    "get_sparse_partition_spec",
    "compute_tile_memory",
    "compute_relation_memory",
    "filter_schemes_by_sharding",
    "filter_schemes_by_device_count",
    "prepare_sparse_tiling",
    # Cost model
    "CostModelConfig",
    "detect_device_config",
    "calibrate",
    "compute_join_cost",
    # DP Optimizer
    "DPOptimizer",
    "ReductionInfo",
    "infer_reduction_info",
    # Contraction path
    "ContractionStep",
    "ContractionPlan",
    "get_contraction_order",
    "plan_contraction",
    # Sharding
    "tile_shape_to_partition_spec",
    "derive_partition_specs",
    "create_mesh",
    "infer_mesh_shape",
    # Resharding
    "needs_reshard",
    "estimate_reshard_bytes",
    "estimate_reshard_cost",
    "reshard_dense",
    "plan_reshard_sequence",
    # Execution
    "execute_dense_einsum",
    "execute_sharded_einsum",
    "ExecutionEngine",
    "build_dependency_graph",
    "topological_sort",
    # Sparse execution
    "execute_sparse",
    "execute_sharded_sparse",
    "coordinate_join",
    "kernel_einsum",
    "segment_sum",
    # Kernels
    "KernelRegistry",
    "KernelInfo",
    "block_sparse_matmul",
    "block_sparse_matmul_generic",
    "coordinate_join_sorted",
    "coordinate_join_hash",
    # Autodiff
    "sparse_einsum",
    "sparse_einsum_raw",
    # Config
    "get_config",
    "set_config",
    "reset_config",
    "list_device_types",
    "get_hardware_profile",
    # API
    "einsum",
    "analyze",
    "AnalysisResult",
    "with_mesh",
]
