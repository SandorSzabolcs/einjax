"""Rigorous additional tests for einjax library.

Covers edge cases, boundary conditions, integration scenarios,
and numerical correctness across all modules.
"""

import unittest
from math import prod
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import jax
import jax.numpy as jnp
import scipy.sparse as sp

from einjax.core.notation import (
    index_to_subscript,
    find_all_factors,
    get_label_dimensions,
    normalize_notation,
    validate_inputs,
)
from einjax.core.types import (
    Expr,
    UnaryOp,
    BinaryOp,
    Constant,
    CaseAssignment,
    TilingScheme,
)
from einjax.tensor.base import BaseTensor, IndexedTerm
from einjax.tensor.dense import DenseTensor
from einjax.tensor.sparse import SparseTensor, SparseTensorRelation, _extract_coo
from einjax.tensor.stats import (
    compute_sparsity_stats_coo,
    compute_sparsity_stats_dense,
    sparsity_ratio,
    update_scheme_sparsity,
)
from einjax.tensor.tiling import (
    prune_infeasible_schemes,
    rank_schemes_by_sparsity,
    select_best_sparse_tiling,
    compute_tile_memory,
    compute_relation_memory,
    filter_schemes_by_sharding,
    filter_schemes_by_device_count,
    prepare_sparse_tiling,
)
from einjax.optimizer.cost_model import (
    CostModelConfig,
    compute_join_cost,
    detect_device_config,
)
from einjax.optimizer.dp import (
    DPOptimizer,
    ReductionInfo,
    infer_reduction_info,
)
from einjax.optimizer.contraction_path import (
    ContractionStep,
    ContractionPlan,
    get_contraction_order,
    plan_contraction,
)
from einjax.sharding.partition import tile_shape_to_partition_spec, derive_partition_specs
from einjax.sharding.mesh import infer_mesh_shape, create_mesh
from einjax.sharding.reshard import (
    needs_reshard,
    estimate_reshard_bytes,
    estimate_reshard_cost,
    reshard_dense,
    plan_reshard_sequence,
)
from einjax.execution.dense_kernels import execute_dense_einsum
from einjax.execution.engine import (
    build_dependency_graph,
    topological_sort,
    ExecutionEngine,
)
from einjax.execution.sparse_dispatch import (
    coordinate_join,
    kernel_einsum,
    segment_sum,
    execute_sparse,
    execute_sharded_sparse,
    _add_batch_dim,
    _infer_output_tile_shape,
    _compute_output_coords,
    _partition_matched_pairs,
)
from einjax.autodiff.custom_vjp import (
    _reverse_einsum_string,
    sparse_einsum,
    sparse_einsum_raw,
)
from einjax.kernels.registry import KernelRegistry, KernelInfo
from einjax.kernels.pallas_matmul import block_sparse_matmul, block_sparse_matmul_generic
from einjax.kernels.pallas_gather import coordinate_join_hash, coordinate_join_sorted
from einjax.config import get_config, set_config, reset_config
from einjax.api import einsum, analyze, with_mesh, AnalysisResult


# =============================================================================
# Notation: Edge Cases
# =============================================================================


class TestIndexToSubscriptEdgeCases(unittest.TestCase):
    """Edge cases for index_to_subscript."""

    def test_boundary_zero(self):
        self.assertEqual(index_to_subscript(0), "i")

    def test_boundary_max(self):
        # 'z' - 'i' = 17, so max valid index is 17
        self.assertEqual(index_to_subscript(17), "z")

    def test_one_past_max_raises(self):
        with self.assertRaises(ValueError):
            index_to_subscript(18)

    def test_negative_wraps_or_raises(self):
        # Negative index may produce unexpected chars; verify behavior
        # chr(ord('i') + (-1)) = chr(104) = 'h' which is < 'i' but <= 'z'
        # This is not explicitly guarded, document it
        result = index_to_subscript(-1)
        self.assertEqual(result, "h")


class TestFindAllFactorsEdgeCases(unittest.TestCase):
    """Edge cases for find_all_factors."""

    def test_one(self):
        self.assertEqual(find_all_factors(1), [1])

    def test_prime(self):
        self.assertEqual(find_all_factors(7), [1, 7])

    def test_large_prime(self):
        self.assertEqual(find_all_factors(997), [1, 997])

    def test_perfect_square(self):
        factors = find_all_factors(16)
        self.assertEqual(factors, [1, 2, 4, 8, 16])

    def test_large_number(self):
        factors = find_all_factors(1024)
        self.assertIn(512, factors)
        self.assertIn(1024, factors)
        self.assertEqual(factors[0], 1)
        self.assertEqual(factors[-1], 1024)

    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            find_all_factors(0)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            find_all_factors(-5)


class TestNormalizeNotationEdgeCases(unittest.TestCase):
    """Edge cases for normalize_notation."""

    def test_single_tensor_identity(self):
        self.assertEqual(normalize_notation("ij"), "ij->ij")

    def test_single_tensor_trace(self):
        self.assertEqual(normalize_notation("ii"), "ii->")

    def test_three_tensor_all_contracted(self):
        # All indices repeated → scalar output
        self.assertEqual(normalize_notation("ij,jk,ki"), "ij,jk,ki->")

    def test_explicit_already_normalized(self):
        self.assertEqual(normalize_notation("ij,jk->ik"), "ij,jk->ik")

    def test_empty_output(self):
        self.assertEqual(normalize_notation("i,i"), "i,i->")

    def test_single_index(self):
        self.assertEqual(normalize_notation("i"), "i->i")


class TestValidateInputsEdgeCases(unittest.TestCase):
    """Edge cases for validate_inputs."""

    def test_scalar_output_valid(self):
        # "ii->" with (3,3) → valid trace
        validate_inputs("ii->", [(3, 3)])

    def test_repeated_labels_in_single_tensor(self):
        validate_inputs("iij->j", [(3, 3, 4)])

    def test_uppercase_label_raises(self):
        with self.assertRaises(ValueError):
            validate_inputs("Ij,jk->Ik", [(3, 4), (4, 5)])

    def test_digit_label_raises(self):
        with self.assertRaises(ValueError):
            validate_inputs("1j,jk->1k", [(3, 4), (4, 5)])

    def test_empty_tensor_shapes(self):
        # "->" with no tensors: input_part="" splits to [""] which has length 1,
        # but we have 0 tensors, so this should raise ValueError
        with self.assertRaises(ValueError):
            validate_inputs("->", [])

    def test_custom_names_in_error(self):
        with self.assertRaises(ValueError, msg="A"):
            validate_inputs("ij,jk->ik", [(3, 4), (5, 6)], names=["A", "B"])


class TestGetLabelDimensionsEdgeCases(unittest.TestCase):
    """Edge cases for get_label_dimensions."""

    def test_3d_tensor(self):
        dims = get_label_dimensions("ijk->ij", [(2, 3, 4)])
        self.assertEqual(dims, {"i": 2, "j": 3, "k": 4})

    def test_shared_label_consistent(self):
        dims = get_label_dimensions("ij,jk->ik", [(3, 4), (4, 5)])
        self.assertEqual(dims["j"], 4)

    def test_shared_label_inconsistent_raises(self):
        with self.assertRaises(ValueError):
            get_label_dimensions("ij,jk->ik", [(3, 4), (5, 6)])


# =============================================================================
# Types: Edge Cases
# =============================================================================


class TestExprHierarchy(unittest.TestCase):
    """Deep nesting and edge cases for Expr AST."""

    def test_deeply_nested_expr(self):
        expr = Constant(1.0)
        for _ in range(100):
            expr = UnaryOp("neg", expr)
        self.assertEqual(expr.flops(), 100.0)

    def test_binary_tree(self):
        a = Constant(0.0)
        b = Constant(0.0)
        c = BinaryOp("*", a, b)
        d = BinaryOp("+", c, c)
        # d.flops = 1 + c.flops + c.flops = 1 + 1 + 1 = 3
        self.assertEqual(d.flops(), 3.0)


class TestTilingSchemeEdgeCases(unittest.TestCase):
    """Edge cases for TilingScheme."""

    def test_1d_scheme(self):
        s = TilingScheme(node=None, shape=(10,), tile_shape=(5,))
        self.assertEqual(s.tile_size, 5)
        self.assertEqual(s.num_tuples, 2)
        self.assertEqual(s.value_count, (2,))

    def test_identity_tiling(self):
        s = TilingScheme(node=None, shape=(8, 8), tile_shape=(8, 8))
        self.assertEqual(s.num_tuples, 1)
        self.assertEqual(s.tile_size, 64)
        cases = s.get_case_assignments()
        self.assertTrue(all(c == CaseAssignment.LOWER for c in cases))

    def test_fully_sharded(self):
        s = TilingScheme(node=None, shape=(8, 8), tile_shape=(1, 1))
        self.assertEqual(s.num_tuples, 64)
        cases = s.get_case_assignments()
        self.assertTrue(all(c == CaseAssignment.UPPER for c in cases))

    def test_mixed_case_assignment(self):
        s = TilingScheme(node=None, shape=(8, 8), tile_shape=(8, 4))
        cases = s.get_case_assignments()
        self.assertEqual(cases[0], CaseAssignment.LOWER)
        self.assertEqual(cases[1], CaseAssignment.UPPER)

    def test_3d_tiling(self):
        s = TilingScheme(node=None, shape=(12, 8, 6), tile_shape=(3, 4, 2))
        self.assertEqual(s.tile_size, 24)
        self.assertEqual(s.num_tuples, 4 * 2 * 3)
        self.assertEqual(s.value_count, (4, 2, 3))

    def test_hash_same_node_same_tile(self):
        # Same node string "A" → same id(node) because Python interns short strings
        # So hash(s1) == hash(s2) when node is same interned string
        s1 = TilingScheme(node="A", shape=(8, 8), tile_shape=(4, 4))
        s2 = TilingScheme(node="A", shape=(8, 8), tile_shape=(4, 4))
        self.assertEqual(hash(s1), hash(s2))

    def test_hash_different_nodes(self):
        # Distinct non-interned objects → different id → different hash
        n1 = type("N", (), {})()
        n2 = type("N", (), {})()
        s1 = TilingScheme(node=n1, shape=(8, 8), tile_shape=(4, 4))
        s2 = TilingScheme(node=n2, shape=(8, 8), tile_shape=(4, 4))
        self.assertNotEqual(hash(s1), hash(s2))

    def test_scheme_cost_defaults(self):
        s = TilingScheme(node=None, shape=(4,), tile_shape=(2,))
        self.assertEqual(s.cost, 0.0)
        self.assertEqual(s.comm, 0.0)
        self.assertEqual(s.flops, 0.0)
        self.assertEqual(s.accumulated_cost, 0.0)


# =============================================================================
# Tensor Base: Edge Cases
# =============================================================================


class TestBaseTensorEdgeCases(unittest.TestCase):
    """Edge cases for BaseTensor."""

    def test_1d_tensor_schemes(self):
        data = np.ones((6,))
        t = DenseTensor("t", data)
        # factors of 6 are [1, 2, 3, 6]
        self.assertEqual(len(t.tile_shapes), 4)

    def test_prime_dimension(self):
        data = np.ones((7, 7))
        t = DenseTensor("t", data)
        # 7 is prime: only factors are 1 and 7
        # tile_shapes: (1,1), (1,7), (7,1), (7,7) = 4
        self.assertEqual(len(t.tile_shapes), 4)

    def test_operator_overloads_return_binaryop(self):
        a = DenseTensor("a", np.ones((2, 2)))
        b = DenseTensor("b", np.ones((2, 2)))
        self.assertIsInstance(a + b, BinaryOp)
        self.assertIsInstance(a - b, BinaryOp)
        self.assertIsInstance(a / b, BinaryOp)

    def test_getitem_returns_indexed_term(self):
        t = DenseTensor("t", np.ones((3, 4)))
        it = t["ij"]
        self.assertIsInstance(it, IndexedTerm)
        self.assertEqual(it.indices, "ij")
        self.assertIs(it.tensor, t)

    def test_ndim_property(self):
        t = DenseTensor("t", np.ones((2, 3, 4)))
        self.assertEqual(t.ndim, 3)


# =============================================================================
# Dense Tensor: Edge Cases
# =============================================================================


class TestDenseTensorEdgeCases(unittest.TestCase):
    """Edge cases for DenseTensor."""

    def test_jax_array_input(self):
        jax_data = jnp.ones((3, 4))
        t = DenseTensor("t", jax_data)
        self.assertEqual(t.shape, (3, 4))
        # Original data reference preserved
        self.assertIs(t.data, jax_data)

    def test_single_element_tensor(self):
        data = np.array([[5.0]])
        t = DenseTensor("t", data)
        self.assertEqual(t.shape, (1, 1))
        self.assertEqual(len(t.tile_shapes), 1)  # only (1,1)

    def test_large_sparse_tensor(self):
        data = np.zeros((100, 100))
        data[0, 0] = 1.0
        t = DenseTensor("t", data)
        # Element tiling (1,1) should have num_tuples=1
        self.assertEqual(t.schemes[(1, 1)].num_tuples, 1)
        # Full tiling should have num_tuples=1 (one block with nonzero)
        self.assertEqual(t.schemes[(100, 100)].num_tuples, 1)

    def test_all_nonzero(self):
        data = np.ones((4, 4))
        t = DenseTensor("t", data)
        # Element tiling: all 16 tiles are non-empty
        self.assertEqual(t.schemes[(1, 1)].num_tuples, 16)

    def test_sparsity_varies_by_tiling(self):
        # Diagonal: only diagonal blocks are non-empty
        data = np.eye(4)
        t = DenseTensor("t", data)
        # Element tiling (1,1): 4 non-empty tiles
        self.assertEqual(t.schemes[(1, 1)].num_tuples, 4)
        # Row tiling (1,4): all 4 rows have nonzero = 4 tiles
        self.assertEqual(t.schemes[(1, 4)].num_tuples, 4)
        # Column tiling (4,1): all 4 cols have nonzero = 4 tiles
        self.assertEqual(t.schemes[(4, 1)].num_tuples, 4)
        # 2x2 block tiling: 2 diagonal blocks
        self.assertEqual(t.schemes[(2, 2)].num_tuples, 2)


# =============================================================================
# Sparse Tensor: Edge Cases
# =============================================================================


class TestSparseTensorEdgeCases(unittest.TestCase):
    """Edge cases for SparseTensor."""

    def test_from_scipy_csr(self):
        mat = sp.csr_matrix(np.eye(4))
        t = SparseTensor("t", mat)
        self.assertEqual(t.shape, (4, 4))
        self.assertEqual(t.nnz, 4)

    def test_from_scipy_csc(self):
        mat = sp.csc_matrix(np.eye(3))
        t = SparseTensor("t", mat)
        self.assertEqual(t.shape, (3, 3))
        self.assertEqual(t.nnz, 3)

    def test_3d_sparse(self):
        data = np.zeros((4, 4, 4))
        data[0, 0, 0] = 1.0
        data[3, 3, 3] = 2.0
        t = SparseTensor("t", data)
        self.assertEqual(t.nnz, 2)
        self.assertEqual(t.shape, (4, 4, 4))

    def test_to_dense_array_roundtrip(self):
        original = np.zeros((8, 8))
        original[1, 3] = 5.0
        original[6, 2] = -3.0
        t = SparseTensor("t", original)
        recovered = t.to_dense_array()
        npt.assert_array_equal(recovered, original)

    def test_to_relation_roundtrip(self):
        original = np.eye(4) * np.array([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("t", original)
        rel = t.to_relation((2, 2))
        recovered = rel.to_dense()
        npt.assert_array_almost_equal(recovered, original)

    def test_to_relation_non_divisible_raises(self):
        t = SparseTensor("t", np.eye(5))
        with self.assertRaises(ValueError):
            t.to_relation((2, 2))

    def test_to_relation_element_tiling(self):
        data = np.eye(3)
        t = SparseTensor("t", data)
        rel = t.to_relation((1, 1))
        self.assertEqual(rel.num_tuples, 3)
        self.assertEqual(rel.tile_shape, (1, 1))

    def test_empty_sparse_tensor(self):
        data = np.zeros((4, 4))
        t = SparseTensor("t", data)
        self.assertEqual(t.nnz, 0)
        for scheme in t.schemes.values():
            self.assertEqual(scheme.num_tuples, 0)


class TestSparseTensorRelationEdgeCases(unittest.TestCase):
    """Edge cases for SparseTensorRelation."""

    def test_density_empty(self):
        r = SparseTensorRelation(
            coords=np.zeros((0, 2), dtype=np.int32),
            values=np.zeros((0, 2, 2)),
            shape=(4, 4),
            tile_shape=(2, 2),
        )
        self.assertEqual(r.density, 0.0)
        self.assertEqual(r.nnz, 0)
        self.assertEqual(r.num_tuples, 0)

    def test_density_full(self):
        coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
        values = np.ones((4, 2, 2))
        r = SparseTensorRelation(
            coords=coords, values=values,
            shape=(4, 4), tile_shape=(2, 2),
        )
        self.assertAlmostEqual(r.density, 1.0)

    def test_to_dense_3d(self):
        coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
        values = np.ones((2, 2, 2, 2))
        r = SparseTensorRelation(
            coords=coords, values=values,
            shape=(4, 4, 4), tile_shape=(2, 2, 2),
        )
        dense = r.to_dense()
        self.assertEqual(dense.shape, (4, 4, 4))
        # First block
        npt.assert_array_equal(dense[:2, :2, :2], np.ones((2, 2, 2)))
        # Second block
        npt.assert_array_equal(dense[2:4, 2:4, 2:4], np.ones((2, 2, 2)))

    def test_tile_size(self):
        r = SparseTensorRelation(
            coords=np.zeros((1, 3), dtype=np.int32),
            values=np.zeros((1, 2, 3, 4)),
            shape=(2, 3, 4), tile_shape=(2, 3, 4),
        )
        self.assertEqual(r.tile_size, 24)


class TestExtractCOOEdgeCases(unittest.TestCase):
    """Edge cases for _extract_coo."""

    def test_1d_array(self):
        arr = np.array([0.0, 1.0, 0.0, 2.0])
        indices, values, shape = _extract_coo(arr)
        self.assertEqual(shape, (4,))
        self.assertEqual(len(values), 2)
        npt.assert_array_equal(indices[:, 0], [1, 3])

    def test_all_zeros(self):
        arr = np.zeros((3, 3))
        indices, values, shape = _extract_coo(arr)
        self.assertEqual(len(values), 0)
        self.assertEqual(indices.shape, (0, 2))

    def test_negative_values(self):
        arr = np.array([[-1.0, 0.0], [0.0, -2.0]])
        indices, values, shape = _extract_coo(arr)
        self.assertEqual(len(values), 2)
        npt.assert_array_equal(values, [-1.0, -2.0])


# =============================================================================
# Stats: Edge Cases
# =============================================================================


class TestSparsityStatsEdgeCases(unittest.TestCase):
    """Edge cases for sparsity statistics."""

    def test_coo_dense_consistency(self):
        """COO and dense stats should agree."""
        data = np.random.RandomState(42).rand(8, 8) * (np.random.RandomState(42).rand(8, 8) > 0.7)
        tile_shapes = [(1, 1), (2, 2), (4, 4), (8, 8)]
        dense_stats = compute_sparsity_stats_dense(data, tile_shapes)
        nz = np.nonzero(data)
        indices = np.stack(nz, axis=1).astype(np.int32)
        values = data[nz]
        coo_stats = compute_sparsity_stats_coo(indices, values, data.shape, tile_shapes)
        for ts in tile_shapes:
            self.assertEqual(dense_stats[ts], coo_stats[ts])

    def test_sparsity_ratio_full(self):
        r = sparsity_ratio(16, (4, 4), (1, 1))
        self.assertAlmostEqual(r, 1.0)

    def test_sparsity_ratio_empty(self):
        r = sparsity_ratio(0, (4, 4), (1, 1))
        self.assertAlmostEqual(r, 0.0)

    def test_sparsity_ratio_half(self):
        r = sparsity_ratio(8, (4, 4), (1, 1))
        self.assertAlmostEqual(r, 0.5)

    def test_update_scheme_sparsity(self):
        s = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        update_scheme_sparsity(s, 3, (2, 2))
        self.assertEqual(s.num_tuples, 3)
        self.assertEqual(s.value_count, (2, 2))


# =============================================================================
# Tiling: Edge Cases
# =============================================================================


class TestTilingEdgeCases(unittest.TestCase):
    """Edge cases for tiling functions."""

    def test_compute_tile_memory_with_coords(self):
        # 2D tile (4,4) with float32, include_coords=True
        mem = compute_tile_memory((4, 4), include_coords=True)
        # values: 16*4 = 64 bytes, coords: 2*4 = 8 bytes
        self.assertEqual(mem, 72)

    def test_compute_tile_memory_without_coords(self):
        mem = compute_tile_memory((4, 4), include_coords=False)
        self.assertEqual(mem, 64)

    def test_compute_relation_memory(self):
        s = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        # Override num_tuples for test
        s.num_tuples = 10
        mem = compute_relation_memory(s)
        # 10 tiles * (16*4 values + 2*4 coords) = 10 * 72 = 720
        self.assertEqual(mem, 720)

    def test_prepare_sparse_tiling_returns_scheme(self):
        data = np.eye(8)
        t = SparseTensor("t", data)
        scheme = prepare_sparse_tiling(t)
        self.assertIsNotNone(scheme)
        self.assertIsInstance(scheme, TilingScheme)


# =============================================================================
# Cost Model: Edge Cases
# =============================================================================


class TestCostModelEdgeCases(unittest.TestCase):
    """Edge cases for cost model."""

    def test_single_device_no_overhead(self):
        config = CostModelConfig.from_device_type("cpu", 1)
        self.assertEqual(config.all_reduce_cost(1e9), 0.0)
        self.assertEqual(config.parallelism_overhead(), 0.0)

    def test_zero_flops(self):
        config = CostModelConfig.from_device_type("cpu", 1)
        self.assertEqual(config.kernel_cost(0.0), 0.0)

    def test_zero_bytes(self):
        config = CostModelConfig.from_device_type("cpu", 1)
        self.assertEqual(config.transfer_cost(0.0), 0.0)
        self.assertEqual(config.reshard_cost(0.0), 0.0)

    def test_total_cost_consistency(self):
        config = CostModelConfig.from_device_type("gpu:a100", 4)
        total = config.total_cost(1e9, 1e12, 100)
        expected = (
            config.transfer_cost(1e9)
            + config.kernel_cost(1e12)
            + config.fixed_cost(100)
        )
        self.assertAlmostEqual(total, expected)

    def test_compute_join_cost_with_agg_keys(self):
        config = CostModelConfig.from_device_type("gpu:a100", 1)
        lhs = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        rhs = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        total, comm, flops, fixed = compute_join_cost(
            lhs, rhs,
            join_key_dims=[1],
            agg_key_dims=[0],
            output_tile_shape=(4, 4),
            config=config,
        )
        self.assertGreater(total, 0)
        self.assertGreater(flops, 0)

    def test_compute_join_cost_no_join_keys(self):
        config = CostModelConfig.from_device_type("cpu", 1)
        lhs = TilingScheme(node=None, shape=(4,), tile_shape=(2,))
        rhs = TilingScheme(node=None, shape=(4,), tile_shape=(2,))
        total, comm, flops, fixed = compute_join_cost(
            lhs, rhs,
            join_key_dims=[],
            agg_key_dims=[],
            output_tile_shape=(2, 2),
            config=config,
        )
        self.assertGreater(total, 0)

    def test_all_reduce_scales_with_devices(self):
        config_2 = CostModelConfig.from_device_type("gpu:a100", 2)
        config_8 = CostModelConfig.from_device_type("gpu:a100", 8)
        cost_2 = config_2.all_reduce_cost(1e9)
        cost_8 = config_8.all_reduce_cost(1e9)
        self.assertGreater(cost_8, cost_2)

    def test_detect_device_config_returns_valid(self):
        config = detect_device_config()
        self.assertIsInstance(config, CostModelConfig)
        self.assertGreater(config.peak_flops, 0)
        self.assertGreater(config.num_devices, 0)


# =============================================================================
# DP Optimizer: Edge Cases
# =============================================================================


class TestDPOptimizerEdgeCases(unittest.TestCase):
    """Edge cases for DP optimizer."""

    def test_infer_reduction_info_matmul(self):
        lhs = DenseTensor("A", np.ones((3, 4)))
        rhs = DenseTensor("B", np.ones((4, 5)))
        lhs_term = IndexedTerm(lhs, "ij")
        rhs_term = IndexedTerm(rhs, "jk")
        info = infer_reduction_info("ik", lhs_term, rhs_term)
        self.assertEqual(info.output_shape, (3, 5))
        # j is a join key
        self.assertEqual(len(info.join_keys), 1)

    def test_infer_reduction_info_outer_product(self):
        lhs = DenseTensor("A", np.ones((3,)))
        rhs = DenseTensor("B", np.ones((4,)))
        lhs_term = IndexedTerm(lhs, "i")
        rhs_term = IndexedTerm(rhs, "j")
        info = infer_reduction_info("ij", lhs_term, rhs_term)
        self.assertEqual(info.output_shape, (3, 4))
        # No join keys for outer product
        self.assertEqual(len(info.join_keys), 0)

    def test_optimize_small_matmul(self):
        lhs = DenseTensor("A", np.random.rand(4, 4))
        rhs = DenseTensor("B", np.random.rand(4, 4))
        lhs_term = IndexedTerm(lhs, "ij")
        rhs_term = IndexedTerm(rhs, "jk")
        config = CostModelConfig.from_device_type("cpu", 1)
        info = infer_reduction_info("ik", lhs_term, rhs_term)
        # Create output tensor with shape (4,4)
        output = DenseTensor("C", np.zeros((4, 4)))
        opt = DPOptimizer(config)
        opt.optimize_reduction(output, info)
        best = opt.get_best_scheme(output)
        self.assertIsNotNone(best)
        self.assertGreater(best.accumulated_cost, 0)

    def test_plan_has_correct_source_chain(self):
        lhs = DenseTensor("A", np.random.rand(4, 4))
        rhs = DenseTensor("B", np.random.rand(4, 4))
        lhs_term = IndexedTerm(lhs, "ij")
        rhs_term = IndexedTerm(rhs, "jk")
        config = CostModelConfig.from_device_type("cpu", 1)
        info = infer_reduction_info("ik", lhs_term, rhs_term)
        output = DenseTensor("C", np.zeros((4, 4)))
        opt = DPOptimizer(config)
        opt.optimize_reduction(output, info)
        plan = opt.get_optimal_plan(output)
        # plan should include lhs, rhs, and output schemes
        self.assertGreaterEqual(len(plan), 2)


# =============================================================================
# Contraction Path: Edge Cases
# =============================================================================


class TestContractionPathEdgeCases(unittest.TestCase):
    """Edge cases for contraction path planning."""

    def test_two_tensor_single_step(self):
        steps = get_contraction_order(
            "ij,jk->ik", [(4, 4), (4, 4)]
        )
        self.assertEqual(len(steps), 1)

    def test_three_tensor_two_steps(self):
        steps = get_contraction_order(
            "ij,jk,kl->il", [(4, 4), (4, 4), (4, 4)]
        )
        self.assertEqual(len(steps), 2)

    def test_plan_contraction_matmul(self):
        shapes = [(4, 4), (4, 4)]
        tensors = [DenseTensor(f"T{i}", np.random.rand(*s)) for i, s in enumerate(shapes)]
        config = CostModelConfig.from_device_type("cpu", 1)
        plan = plan_contraction("ij,jk->ik", tensors, config)
        self.assertIsInstance(plan, ContractionPlan)
        self.assertEqual(len(plan.steps), 1)

    def test_plan_contraction_three_tensor(self):
        shapes = [(4, 4), (4, 4), (4, 4)]
        tensors = [DenseTensor(f"T{i}", np.random.rand(*s)) for i, s in enumerate(shapes)]
        config = CostModelConfig.from_device_type("cpu", 1)
        plan = plan_contraction("ij,jk,kl->il", tensors, config)
        self.assertEqual(len(plan.steps), 2)

    def test_fewer_than_two_tensors_raises(self):
        t = DenseTensor("T", np.random.rand(4, 4))
        config = CostModelConfig.from_device_type("cpu", 1)
        with self.assertRaises(ValueError):
            plan_contraction("ij->ij", [t], config)


# =============================================================================
# Sharding: Edge Cases
# =============================================================================


class TestPartitionSpecEdgeCases(unittest.TestCase):
    """Edge cases for partition spec derivation."""

    def test_scalar_tensor_empty_spec(self):
        # 0-D: shape=(), tile_shape=()
        spec = tile_shape_to_partition_spec((), ())
        self.assertEqual(spec, ())

    def test_5d_all_sharded(self):
        shape = (10, 10, 10, 10, 10)
        tile = (5, 5, 5, 5, 5)
        # Only 4 axis names by default, this should raise
        with self.assertRaises(ValueError):
            tile_shape_to_partition_spec(shape, tile)

    def test_custom_axis_names(self):
        spec = tile_shape_to_partition_spec(
            (8, 8), (4, 4), mesh_axis_names=("a", "b")
        )
        self.assertEqual(spec, ("a", "b"))

    def test_derive_partition_specs_stores_on_scheme(self):
        s = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 8))
        spec = derive_partition_specs(s)
        self.assertEqual(spec, ("x", None))
        self.assertEqual(s.partition_spec, ("x", None))


class TestMeshEdgeCases(unittest.TestCase):
    """Edge cases for mesh creation."""

    def test_infer_mesh_1d(self):
        shape = infer_mesh_shape(1, 1)
        self.assertEqual(shape, (1,))

    def test_infer_mesh_2d_4_devices(self):
        shape = infer_mesh_shape(4, 2)
        self.assertEqual(prod(shape), 4)
        self.assertEqual(len(shape), 2)

    def test_infer_mesh_invalid_zero(self):
        with self.assertRaises(ValueError):
            infer_mesh_shape(0, 1)

    def test_create_mesh_single_device(self):
        mesh = create_mesh(("x",), num_devices=1)
        self.assertEqual(mesh.shape["x"], 1)


class TestReshardEdgeCases(unittest.TestCase):
    """Edge cases for resharding."""

    def test_same_scheme_no_reshard(self):
        s = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        self.assertFalse(needs_reshard(s, s))

    def test_different_scheme_needs_reshard(self):
        s1 = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        s2 = TilingScheme(node=None, shape=(8, 8), tile_shape=(8, 8))
        self.assertTrue(needs_reshard(s1, s2))

    def test_reshard_bytes_different_shapes_raises(self):
        s1 = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        s2 = TilingScheme(node=None, shape=(4, 4), tile_shape=(2, 2))
        with self.assertRaises(ValueError):
            estimate_reshard_bytes(s1, s2)

    def test_reshard_cost_proportional_to_bytes(self):
        s1 = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        s2 = TilingScheme(node=None, shape=(8, 8), tile_shape=(8, 8))
        cost = estimate_reshard_cost(s1, s2, interconnect_bandwidth=100e9)
        bytes_est = estimate_reshard_bytes(s1, s2)
        self.assertAlmostEqual(cost, bytes_est / 100e9)

    def test_reshard_dense_no_mesh(self):
        arr = jnp.ones((4, 4))
        s1 = TilingScheme(node=None, shape=(4, 4), tile_shape=(2, 2))
        s2 = TilingScheme(node=None, shape=(4, 4), tile_shape=(4, 4))
        result = reshard_dense(arr, s1, s2, mesh=None)
        npt.assert_array_equal(np.asarray(result), np.ones((4, 4)))

    def test_plan_reshard_sequence_no_changes(self):
        s = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        pairs = plan_reshard_sequence([s, s, s])
        self.assertEqual(pairs, [])

    def test_plan_reshard_sequence_all_changes(self):
        s1 = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        s2 = TilingScheme(node=None, shape=(8, 8), tile_shape=(8, 8))
        s3 = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 2))
        pairs = plan_reshard_sequence([s1, s2, s3])
        self.assertEqual(pairs, [(0, 1), (1, 2)])


# =============================================================================
# Execution: Sparse Dispatch Edge Cases
# =============================================================================


class TestCoordinateJoinEdgeCases(unittest.TestCase):
    """Edge cases for coordinate join."""

    def test_single_tile_self_join(self):
        coords = np.array([[0, 0]], dtype=np.int32)
        values = np.ones((1, 2, 2))
        rel = SparseTensorRelation(
            coords=coords, values=values,
            shape=(2, 2), tile_shape=(2, 2),
        )
        lhs_idx, rhs_idx = coordinate_join(rel, rel, [(1, 0)])
        self.assertEqual(len(lhs_idx), 1)
        self.assertEqual(len(rhs_idx), 1)

    def test_no_join_keys_cross_product(self):
        coords_a = np.array([[0], [1]], dtype=np.int32)
        values_a = np.ones((2, 3))
        rel_a = SparseTensorRelation(
            coords=coords_a, values=values_a,
            shape=(6,), tile_shape=(3,),
        )
        coords_b = np.array([[0], [1], [2]], dtype=np.int32)
        values_b = np.ones((3, 2))
        rel_b = SparseTensorRelation(
            coords=coords_b, values=values_b,
            shape=(6,), tile_shape=(2,),
        )
        lhs_idx, rhs_idx = coordinate_join(rel_a, rel_b, [])
        # Cross product: 2 * 3 = 6
        self.assertEqual(len(lhs_idx), 6)

    def test_many_to_many_join(self):
        # LHS: two tiles with coord 0 on dim 0
        coords_a = np.array([[0, 0], [0, 1]], dtype=np.int32)
        values_a = np.ones((2, 2, 2))
        rel_a = SparseTensorRelation(
            coords=coords_a, values=values_a,
            shape=(2, 4), tile_shape=(2, 2),
        )
        # RHS: two tiles with coord 0 on dim 0
        coords_b = np.array([[0, 0], [0, 1]], dtype=np.int32)
        values_b = np.ones((2, 2, 2))
        rel_b = SparseTensorRelation(
            coords=coords_b, values=values_b,
            shape=(2, 4), tile_shape=(2, 2),
        )
        lhs_idx, rhs_idx = coordinate_join(rel_a, rel_b, [(0, 0)])
        # 2 lhs tiles * 2 rhs tiles (both match on coord 0) = 4
        self.assertEqual(len(lhs_idx), 4)


class TestKernelEinsumEdgeCases(unittest.TestCase):
    """Edge cases for kernel einsum."""

    def test_single_pair(self):
        lhs = np.eye(2).reshape(1, 2, 2)
        rhs = np.array([[[1, 2], [3, 4]]])
        lhs_idx = np.array([0])
        rhs_idx = np.array([0])
        result = kernel_einsum(lhs, rhs, lhs_idx, rhs_idx, "ij,jk->ik")
        npt.assert_array_almost_equal(result[0], [[1, 2], [3, 4]])

    def test_empty_pairs(self):
        lhs = np.zeros((3, 2, 2))
        rhs = np.zeros((3, 2, 2))
        lhs_idx = np.array([], dtype=np.int64)
        rhs_idx = np.array([], dtype=np.int64)
        result = kernel_einsum(lhs, rhs, lhs_idx, rhs_idx, "ij,jk->ik")
        self.assertEqual(result.shape[0], 0)


class TestSegmentSumEdgeCases(unittest.TestCase):
    """Edge cases for segment sum."""

    def test_single_entry(self):
        values = np.array([[[1, 2], [3, 4]]])
        coords = np.array([[0, 0]])
        unique_coords, agg = segment_sum(values, coords)
        self.assertEqual(agg.shape, (1, 2, 2))
        npt.assert_array_equal(agg[0], [[1, 2], [3, 4]])

    def test_three_entries_same_coord(self):
        values = np.ones((3, 2, 2))
        coords = np.array([[0, 0], [0, 0], [0, 0]])
        unique_coords, agg = segment_sum(values, coords)
        self.assertEqual(agg.shape, (1, 2, 2))
        npt.assert_array_almost_equal(agg[0], np.ones((2, 2)) * 3)


class TestAddBatchDim(unittest.TestCase):
    """Edge cases for _add_batch_dim."""

    def test_trace(self):
        result = _add_batch_dim("ii->", "z")
        self.assertEqual(result, "zii->z")

    def test_three_input(self):
        result = _add_batch_dim("ij,jk,kl->il", "z")
        self.assertEqual(result, "zij,zjk,zkl->zil")

    def test_no_arrow_raises(self):
        with self.assertRaises(ValueError):
            _add_batch_dim("ij", "z")


class TestInferOutputTileShape(unittest.TestCase):
    """Edge cases for _infer_output_tile_shape."""

    def test_scalar_output(self):
        shape = _infer_output_tile_shape("ij,ji->", (2, 3), (3, 2))
        self.assertEqual(shape, ())

    def test_vector_output(self):
        shape = _infer_output_tile_shape("ij,j->i", (4, 3), (3,))
        self.assertEqual(shape, (4,))

    def test_no_arrow_raises(self):
        with self.assertRaises(ValueError):
            _infer_output_tile_shape("ij", (2, 3), (3, 2))


class TestPartitionMatchedPairs(unittest.TestCase):
    """Edge cases for _partition_matched_pairs."""

    def test_one_match_many_devices(self):
        parts = _partition_matched_pairs(1, 8)
        # Only first device gets the match
        total = sum(end - start for start, end in parts)
        self.assertEqual(total, 1)

    def test_exact_division(self):
        parts = _partition_matched_pairs(16, 4)
        for start, end in parts:
            self.assertEqual(end - start, 4)


# =============================================================================
# Execution: Dense Kernels
# =============================================================================


class TestDenseKernelNumerics(unittest.TestCase):
    """Numerical correctness tests for dense einsum."""

    def test_matmul_numeric(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = execute_dense_einsum("ij,jk->ik", A, B)
        expected = jnp.array([[19.0, 22.0], [43.0, 50.0]])
        npt.assert_array_almost_equal(result, expected)

    def test_trace_numeric(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = execute_dense_einsum("ii->", A)
        self.assertAlmostEqual(float(result), 5.0)

    def test_outer_product_numeric(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0])
        result = execute_dense_einsum("i,j->ij", a, b)
        expected = jnp.array([[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]])
        npt.assert_array_almost_equal(result, expected)

    def test_batch_matmul_numeric(self):
        A = jnp.ones((2, 3, 4))
        B = jnp.ones((2, 4, 5))
        result = execute_dense_einsum("bij,bjk->bik", A, B)
        self.assertEqual(result.shape, (2, 3, 5))
        # Each element should be sum of 4 ones = 4.0
        npt.assert_array_almost_equal(result, 4.0 * np.ones((2, 3, 5)))

    def test_diagonal_extraction(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = execute_dense_einsum("ii->i", A)
        npt.assert_array_almost_equal(result, [1.0, 4.0])


# =============================================================================
# Execution: Sparse Pipeline End-to-End
# =============================================================================


class TestExecuteSparseNumerics(unittest.TestCase):
    """Numerical correctness for sparse execution."""

    def test_sparse_matmul_vs_dense(self):
        """Sparse matmul should match dense matmul."""
        A = np.eye(4)
        B = np.array([
            [1, 2, 0, 0],
            [0, 3, 4, 0],
            [0, 0, 5, 6],
            [0, 0, 0, 7],
        ], dtype=np.float64)
        # A * B = B (since A is identity)
        st_a = SparseTensor("A", A)
        st_b = SparseTensor("B", B)
        rel_a = st_a.to_relation((2, 2))
        rel_b = st_b.to_relation((2, 2))
        result = execute_sparse(
            rel_a, rel_b,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
        )
        dense_result = result.to_dense()
        npt.assert_array_almost_equal(dense_result, B)

    def test_sparse_matmul_with_aggregation(self):
        """Non-identity matmul requires aggregation."""
        np.random.seed(42)
        A = np.zeros((4, 4))
        A[0, 0] = 1; A[0, 2] = 2
        A[1, 1] = 3; A[1, 3] = 4
        B = np.zeros((4, 4))
        B[0, 0] = 5; B[2, 0] = 6
        B[1, 1] = 7; B[3, 1] = 8
        expected = A @ B
        st_a = SparseTensor("A", A)
        st_b = SparseTensor("B", B)
        rel_a = st_a.to_relation((2, 2))
        rel_b = st_b.to_relation((2, 2))
        result = execute_sparse(
            rel_a, rel_b,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
        )
        dense_result = result.to_dense()
        npt.assert_array_almost_equal(dense_result, expected)

    def test_sparse_elementwise(self):
        """Element-wise product (Hadamard)."""
        A = np.array([[1, 0], [0, 2]], dtype=np.float64)
        B = np.array([[3, 0], [0, 4]], dtype=np.float64)
        st_a = SparseTensor("A", A)
        st_b = SparseTensor("B", B)
        rel_a = st_a.to_relation((1, 1))
        rel_b = st_b.to_relation((1, 1))
        result = execute_sparse(
            rel_a, rel_b,
            join_keys=[(0, 0), (1, 1)],
            kernel_string="ij,ij->ij",
            agg_keys=[(0, 0), (0, 1)],
        )
        dense_result = result.to_dense()
        npt.assert_array_almost_equal(dense_result, A * B)


# =============================================================================
# Autodiff: Edge Cases
# =============================================================================


class TestAutodiffEdgeCases(unittest.TestCase):
    """Edge cases for autodiff."""

    def test_reverse_einsum_outer_product(self):
        grad_lhs, grad_rhs = _reverse_einsum_string("i,j->ij")
        self.assertEqual(grad_lhs, "ij,j->i")
        self.assertEqual(grad_rhs, "i,ij->j")

    def test_reverse_einsum_dot_product(self):
        grad_lhs, grad_rhs = _reverse_einsum_string("i,i->")
        self.assertEqual(grad_lhs, ",i->i")
        self.assertEqual(grad_rhs, "i,->i")

    def test_sparse_einsum_forward_correctness(self):
        """sparse_einsum forward should match dense matmul."""
        A = np.eye(4)
        B = np.eye(4) * 2.0
        st_a = SparseTensor("A", A)
        st_b = SparseTensor("B", B)
        rel_a = st_a.to_relation((2, 2))
        rel_b = st_b.to_relation((2, 2))
        result = sparse_einsum(
            rel_a, rel_b,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
        )
        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_array_almost_equal(dense_result, expected)

    def test_sparse_einsum_raw_grad_shapes(self):
        """Gradient shapes from sparse_einsum_raw should match input shapes."""
        lhs_values = jnp.ones((2, 2, 2))
        rhs_values = jnp.ones((2, 2, 2))
        lhs_coords = np.array([[0, 0], [1, 1]], dtype=np.int32)
        rhs_coords = np.array([[0, 0], [1, 1]], dtype=np.int32)
        lhs_indices = np.array([0, 1], dtype=np.int64)
        rhs_indices = np.array([0, 1], dtype=np.int64)

        def loss_fn(lhs_v, rhs_v):
            result = sparse_einsum_raw(
                lhs_v, rhs_v,
                lhs_coords, rhs_coords,
                lhs_indices, rhs_indices,
                "ij,jk->ik",
            )
            return jnp.sum(result)

        grad_lhs, grad_rhs = jax.grad(loss_fn, argnums=(0, 1))(lhs_values, rhs_values)
        self.assertEqual(grad_lhs.shape, lhs_values.shape)
        self.assertEqual(grad_rhs.shape, rhs_values.shape)


# =============================================================================
# Kernels: Registry
# =============================================================================


class TestKernelRegistryEdgeCases(unittest.TestCase):
    """Edge cases for kernel registry."""

    def test_wildcard_matches_any_pattern(self):
        reg = KernelRegistry()
        # Default registry has a wildcard "*" pattern for dense, which matches anything
        fn = reg.lookup("nonexistent_pattern_xyz", "gpu", sparsity="dense")
        self.assertIsNotNone(fn)

    def test_builtin_sparse_matmul(self):
        reg = KernelRegistry()
        fn = reg.lookup("ij,jk->ik", "cpu", sparsity="sparse")
        self.assertIsNotNone(fn)

    def test_register_and_retrieve(self):
        reg = KernelRegistry()
        fn = lambda: "test_result"
        reg.register(KernelInfo(
            name="test_op", pattern="ab,bc->ac", backend="generic",
            sparsity="dense", priority=100, kernel_fn=fn,
        ))
        result = reg.lookup("ab,bc->ac", "gpu", sparsity="dense")
        self.assertIs(result, fn)

    def test_priority_ordering(self):
        reg = KernelRegistry()
        low_fn = lambda: "low"
        high_fn = lambda: "high"
        reg.register(KernelInfo(
            name="low", pattern="ab->ba", backend="generic",
            sparsity="dense", priority=0, kernel_fn=low_fn,
        ))
        reg.register(KernelInfo(
            name="high", pattern="ab->ba", backend="generic",
            sparsity="dense", priority=100, kernel_fn=high_fn,
        ))
        result = reg.lookup("ab->ba", "generic", sparsity="dense")
        self.assertIs(result, high_fn)


# =============================================================================
# Pallas Kernels: Edge Cases
# =============================================================================


class TestBlockSparseMatmulEdgeCases(unittest.TestCase):
    """Edge cases for block sparse matmul."""

    def test_identity_matmul(self):
        # Build SparseTensorRelation for identity matrix
        coords = np.array([[0, 0]], dtype=np.int32)
        lhs_values = np.eye(4).reshape(1, 4, 4)
        rhs_values = np.ones((1, 4, 4))
        lhs_rel = SparseTensorRelation(
            coords=coords, values=lhs_values,
            shape=(4, 4), tile_shape=(4, 4),
        )
        rhs_rel = SparseTensorRelation(
            coords=coords, values=rhs_values,
            shape=(4, 4), tile_shape=(4, 4),
        )
        result = block_sparse_matmul_generic(lhs_rel, rhs_rel)
        npt.assert_array_almost_equal(result.to_dense(), np.ones((4, 4)))

    def test_block_sparse_matmul_api(self):
        coords = np.array([[0, 0]], dtype=np.int32)
        lhs_values = np.eye(2, dtype=np.float64).reshape(1, 2, 2)
        rhs_values = np.array([[[3.0, 4.0], [5.0, 6.0]]])
        result = block_sparse_matmul(
            coords, lhs_values, coords, rhs_values,
            block_shape=(2, 2, 2),
            lhs_shape=(2, 2), rhs_shape=(2, 2),
        )
        npt.assert_array_almost_equal(
            result.to_dense(), [[3.0, 4.0], [5.0, 6.0]]
        )


class TestCoordinateJoinKernelEdgeCases(unittest.TestCase):
    """Edge cases for coordinate join kernels."""

    def test_hash_and_sorted_consistency(self):
        """Hash and sorted join should return same matches."""
        coords_a = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32)
        coords_b = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int32)
        join_dims = (1, 0)  # a.dim1 == b.dim0

        hash_l, hash_r = coordinate_join_hash(coords_a, coords_b, [join_dims])
        sort_l, sort_r = coordinate_join_sorted(coords_a, coords_b, [join_dims])

        # Same number of matches
        self.assertEqual(len(hash_l), len(sort_l))
        # Same pairs (order may differ)
        hash_pairs = set(zip(hash_l.tolist(), hash_r.tolist()))
        sort_pairs = set(zip(sort_l.tolist(), sort_r.tolist()))
        self.assertEqual(hash_pairs, sort_pairs)

    def test_empty_inputs(self):
        coords_a = np.zeros((0, 2), dtype=np.int32)
        coords_b = np.zeros((0, 2), dtype=np.int32)
        l, r = coordinate_join_hash(coords_a, coords_b, [(0, 0)])
        self.assertEqual(len(l), 0)


# =============================================================================
# API: Edge Cases
# =============================================================================


class TestEinsumAPIEdgeCases(unittest.TestCase):
    """Edge cases for the user-facing einsum API."""

    def test_single_tensor_ops(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        # Trace
        result = einsum("ii->", A)
        self.assertAlmostEqual(float(result), 5.0)
        # Transpose
        result = einsum("ij->ji", A)
        npt.assert_array_almost_equal(result, A.T)
        # Diagonal
        result = einsum("ii->i", A)
        npt.assert_array_almost_equal(result, [1.0, 4.0])

    def test_implicit_notation(self):
        A = jnp.ones((3, 4))
        B = jnp.ones((4, 5))
        result = einsum("ij,jk", A, B)
        self.assertEqual(result.shape, (3, 5))

    def test_three_tensor(self):
        A = jnp.ones((2, 3))
        B = jnp.ones((3, 4))
        C = jnp.ones((4, 5))
        result = einsum("ij,jk,kl->il", A, B, C)
        self.assertEqual(result.shape, (2, 5))
        npt.assert_array_almost_equal(result, 12.0 * np.ones((2, 5)))

    def test_num_devices_1_single_device(self):
        A = jnp.ones((4, 4))
        B = jnp.ones((4, 4))
        result = einsum("ij,jk->ik", A, B, num_devices=1)
        self.assertEqual(result.shape, (4, 4))

    def test_validation_error(self):
        A = jnp.ones((3, 4))
        B = jnp.ones((5, 6))
        with self.assertRaises(ValueError):
            einsum("ij,jk->ik", A, B)


class TestAnalyzeEdgeCases(unittest.TestCase):
    """Edge cases for analyze function."""

    def test_analyze_trace(self):
        A = np.zeros((3, 3))
        result = analyze("ii->", A)
        self.assertEqual(result.output_shape, ())
        self.assertIn("i", result.contracted_indices)

    def test_analyze_transpose(self):
        A = np.zeros((3, 4))
        result = analyze("ij->ji", A)
        self.assertEqual(result.output_shape, (4, 3))
        self.assertEqual(result.contracted_indices, [])

    def test_analyze_repr(self):
        A = np.zeros((3, 4))
        B = np.zeros((4, 5))
        result = analyze("ij,jk->ik", A, B)
        text = repr(result)
        self.assertIn("ij,jk->ik", text)
        self.assertIn("(3, 5)", text)

    def test_analyze_three_tensor(self):
        A = np.zeros((2, 3))
        B = np.zeros((3, 4))
        C = np.zeros((4, 5))
        result = analyze("ij,jk,kl->il", A, B, C)
        self.assertEqual(result.output_shape, (2, 5))
        self.assertIn("j", result.contracted_indices)
        self.assertIn("k", result.contracted_indices)


class TestWithMeshEdgeCases(unittest.TestCase):
    """Edge cases for with_mesh context manager."""

    def test_nested_contexts(self):
        mesh1 = create_mesh(("x",), num_devices=1)
        mesh2 = create_mesh(("x",), num_devices=1)
        with with_mesh(mesh1) as m1:
            self.assertIs(m1, mesh1)
            with with_mesh(mesh2) as m2:
                self.assertIs(m2, mesh2)

    def test_exception_cleans_up(self):
        mesh = create_mesh(("x",), num_devices=1)
        try:
            with with_mesh(mesh):
                raise RuntimeError("test")
        except RuntimeError:
            pass
        # Context should be cleaned up — next einsum should use single device
        result = einsum("ij,jk->ik", jnp.ones((2, 2)), jnp.ones((2, 2)))
        self.assertEqual(result.shape, (2, 2))


# =============================================================================
# Config: Edge Cases
# =============================================================================


class TestConfigEdgeCases(unittest.TestCase):
    """Edge cases for config management."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_set_invalid_type(self):
        with self.assertRaises(TypeError):
            set_config("not a config")

    def test_get_auto_detects(self):
        config = get_config()
        self.assertIsInstance(config, CostModelConfig)

    def test_reset_allows_redetection(self):
        c1 = get_config()
        reset_config()
        c2 = get_config()
        # Should both be valid configs
        self.assertIsInstance(c1, CostModelConfig)
        self.assertIsInstance(c2, CostModelConfig)


# =============================================================================
# Execution Engine: Edge Cases
# =============================================================================


class TestExecutionEngineEdgeCases(unittest.TestCase):
    """Edge cases for ExecutionEngine."""

    def test_single_matmul_no_mesh(self):
        engine = ExecutionEngine(mesh=None)
        A = jnp.ones((3, 4))
        B = jnp.ones((4, 5))
        result = engine._execute_einsum("ij,jk->ik", [A, B])
        self.assertEqual(result.shape, (3, 5))

    def test_execute_plan_matmul(self):
        """Execute a contraction plan through the engine."""
        shapes = [(4, 4), (4, 4)]
        tensors = [DenseTensor(f"T{i}", np.random.rand(*s)) for i, s in enumerate(shapes)]
        config = CostModelConfig.from_device_type("cpu", 1)
        plan = plan_contraction("ij,jk->ik", tensors, config)
        engine = ExecutionEngine(mesh=None)
        A = jnp.asarray(tensors[0].data)
        B = jnp.asarray(tensors[1].data)
        result = engine.execute_plan(plan, [A, B])
        expected = A @ B
        npt.assert_array_almost_equal(result, expected, decimal=5)

    def test_build_dependency_graph_leaf(self):
        """A leaf scheme has no sources."""
        s = TilingScheme(node=type("N", (), {"name": "leaf"})(), shape=(4, 4), tile_shape=(4, 4))
        graph = build_dependency_graph(s)
        self.assertEqual(len(graph), 1)

    def test_topological_sort_single(self):
        s = TilingScheme(node=type("N", (), {"name": "single"})(), shape=(4,), tile_shape=(4,))
        ordered = topological_sort(s)
        self.assertEqual(len(ordered), 1)


# =============================================================================
# Integration: End-to-End Correctness
# =============================================================================


class TestEndToEndCorrectness(unittest.TestCase):
    """End-to-end integration tests verifying numerical correctness."""

    def test_dense_matmul_chain(self):
        """A @ B @ C via einjax matches numpy."""
        np.random.seed(42)
        A = np.random.rand(4, 6)
        B = np.random.rand(6, 5)
        C = np.random.rand(5, 3)
        expected = A @ B @ C
        result = einsum("ij,jk,kl->il",
                        jnp.array(A), jnp.array(B), jnp.array(C))
        npt.assert_array_almost_equal(result, expected, decimal=5)

    def test_batch_trace(self):
        """Batch trace: sum of diagonal for each batch element."""
        A = jnp.array([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ], dtype=jnp.float32)
        result = einsum("bii->b", A)
        npt.assert_array_almost_equal(result, [5.0, 13.0])

    def test_hadamard_product(self):
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = einsum("ij,ij->ij", A, B)
        npt.assert_array_almost_equal(result, A * B)

    def test_vector_dot_product(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0, 6.0])
        result = einsum("i,i->", a, b)
        self.assertAlmostEqual(float(result), 32.0)

    def test_tensor_contraction_4d(self):
        A = jnp.ones((2, 3, 4, 5))
        B = jnp.ones((5, 6))
        result = einsum("ijkl,lm->ijkm", A, B)
        self.assertEqual(result.shape, (2, 3, 4, 6))
        npt.assert_array_almost_equal(result, 5.0 * np.ones((2, 3, 4, 6)))

    def test_sparse_to_dense_consistency(self):
        """SparseTensor → relation → to_dense matches original."""
        np.random.seed(42)
        data = np.zeros((8, 8))
        for _ in range(10):
            i, j = np.random.randint(0, 8, 2)
            data[i, j] = np.random.rand()
        t = SparseTensor("t", data)
        for ts in [(1, 1), (2, 2), (4, 4), (8, 8)]:
            rel = t.to_relation(ts)
            recovered = rel.to_dense()
            npt.assert_array_almost_equal(recovered, data, err_msg=f"Failed for tile_shape={ts}")

    def test_analysis_and_execution_agree(self):
        """analyze() output_shape matches actual einsum result shape."""
        A = jnp.ones((3, 4))
        B = jnp.ones((4, 5))
        analysis = analyze("ij,jk->ik", A, B)
        result = einsum("ij,jk->ik", A, B)
        self.assertEqual(result.shape, analysis.output_shape)


# =============================================================================
# Package-level Import Tests
# =============================================================================


class TestAllExports(unittest.TestCase):
    """Verify key functions are importable from top-level einjax."""

    def test_einsum_importable(self):
        from einjax import einsum
        self.assertTrue(callable(einsum))

    def test_analyze_importable(self):
        from einjax import analyze
        self.assertTrue(callable(analyze))

    def test_with_mesh_importable(self):
        from einjax import with_mesh

    def test_sparse_tensor_importable(self):
        from einjax import SparseTensor, SparseTensorRelation

    def test_dense_tensor_importable(self):
        from einjax import DenseTensor

    def test_kernel_registry_importable(self):
        from einjax import KernelRegistry

    def test_cost_model_importable(self):
        from einjax import CostModelConfig, compute_join_cost

    def test_optimizer_importable(self):
        from einjax import DPOptimizer, plan_contraction

    def test_sharding_importable(self):
        from einjax import (
            tile_shape_to_partition_spec,
            create_mesh,
            needs_reshard,
            reshard_dense,
        )

    def test_execution_importable(self):
        from einjax import (
            execute_dense_einsum,
            execute_sparse,
            ExecutionEngine,
            coordinate_join,
        )

    def test_autodiff_importable(self):
        from einjax import sparse_einsum, sparse_einsum_raw


if __name__ == "__main__":
    unittest.main()
