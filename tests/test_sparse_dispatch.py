"""
Tests for einjax sparse dispatch execution path.

Tests coordinate_join, kernel_einsum, segment_sum, and end-to-end
execute_sparse against numpy.einsum reference results.
"""

import unittest

import numpy as np
import numpy.testing as npt

from einjax.tensor.sparse import SparseTensor, SparseTensorRelation
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


# =============================================================================
# Helper: build SparseTensorRelation from dense matrix with given tile shape
# =============================================================================


def _relation_from_dense(dense: np.ndarray, tile_shape: tuple[int, ...]) -> SparseTensorRelation:
    """Build a SparseTensorRelation from a dense array."""
    st = SparseTensor("test", dense)
    return st.to_relation(tile_shape)


# =============================================================================
# Coordinate Join Tests
# =============================================================================


class TestCoordinateJoin(unittest.TestCase):
    """Tests for the coordinate join phase."""

    def test_matmul_join(self):
        """Matmul A(I,J)*B(J,K) joins on J dimension."""
        # A: 4x4 with tiles at (0,0) and (0,1)
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[0, 2] = 2.0
        lhs = _relation_from_dense(A, (2, 2))

        # B: 4x4 with tiles at (0,0) and (1,0)
        B = np.zeros((4, 4))
        B[0, 0] = 3.0
        B[2, 0] = 4.0
        rhs = _relation_from_dense(B, (2, 2))

        # Join on LHS dim 1 == RHS dim 0
        lhs_idx, rhs_idx = coordinate_join(lhs, rhs, [(1, 0)])

        # LHS tile (0,0) has J=0, LHS tile (0,1) has J=1
        # RHS tile (0,0) has J=0, RHS tile (1,0) has J=1
        # Matches: (0,0)-(0,0) on J=0, (0,1)-(1,0) on J=1
        self.assertEqual(len(lhs_idx), len(rhs_idx))
        self.assertEqual(len(lhs_idx), 2)

    def test_cross_product(self):
        """No join keys produces cross product."""
        A = np.eye(2)
        B = np.eye(2)
        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        lhs_idx, rhs_idx = coordinate_join(lhs, rhs, [])
        # 2 LHS tiles x 2 RHS tiles = 4 matches
        self.assertEqual(len(lhs_idx), 4)
        self.assertEqual(len(rhs_idx), 4)

    def test_no_matches(self):
        """Non-overlapping join keys produce no matches."""
        # LHS has tiles at J=0 only
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        lhs = _relation_from_dense(A, (2, 2))

        # RHS has tiles at J=1 only
        B = np.zeros((4, 4))
        B[2, 0] = 1.0
        rhs = _relation_from_dense(B, (2, 2))

        lhs_idx, rhs_idx = coordinate_join(lhs, rhs, [(1, 0)])
        # LHS tile (0,0) has J=0, RHS tile (1,0) has J=1 — no match
        self.assertEqual(len(lhs_idx), 0)

    def test_multiple_matches_per_key(self):
        """Multiple LHS tiles can match a single RHS tile."""
        # LHS: tiles at (0,0), (1,0) — both with J=0
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[2, 0] = 2.0
        lhs = _relation_from_dense(A, (2, 2))

        # RHS: tile at (0,0) — J=0
        B = np.zeros((4, 4))
        B[0, 0] = 3.0
        rhs = _relation_from_dense(B, (2, 2))

        lhs_idx, rhs_idx = coordinate_join(lhs, rhs, [(1, 0)])
        # Both LHS tiles match the single RHS tile on J=0
        self.assertEqual(len(lhs_idx), 2)

    def test_empty_input(self):
        """Empty inputs produce empty output."""
        lhs = SparseTensorRelation(
            coords=np.zeros((0, 2), dtype=np.int32),
            values=np.zeros((0, 2, 2), dtype=np.float64),
            shape=(4, 4),
            tile_shape=(2, 2),
        )
        rhs = SparseTensorRelation(
            coords=np.zeros((0, 2), dtype=np.int32),
            values=np.zeros((0, 2, 2), dtype=np.float64),
            shape=(4, 4),
            tile_shape=(2, 2),
        )
        lhs_idx, rhs_idx = coordinate_join(lhs, rhs, [(1, 0)])
        self.assertEqual(len(lhs_idx), 0)


# =============================================================================
# Kernel Einsum Tests
# =============================================================================


class TestKernelEinsum(unittest.TestCase):
    """Tests for the vmap(einsum) kernel phase."""

    def test_matmul_tiles(self):
        """Per-tile matmul on matched pairs."""
        # Two 2x2 LHS tiles
        lhs_vals = np.array([
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ])
        # Two 2x2 RHS tiles
        rhs_vals = np.array([
            [[3.0, 0.0], [0.0, 3.0]],
            [[4.0, 0.0], [0.0, 4.0]],
        ])
        lhs_idx = np.array([0, 1])
        rhs_idx = np.array([0, 1])

        result = kernel_einsum(lhs_vals, rhs_vals, lhs_idx, rhs_idx, "ij,jk->ik")

        self.assertEqual(result.shape, (2, 2, 2))
        # First pair: I @ 3I = 3I
        npt.assert_allclose(result[0], [[3.0, 0.0], [0.0, 3.0]], atol=1e-6)
        # Second pair: 2I @ 4I = 8I
        npt.assert_allclose(result[1], [[8.0, 0.0], [0.0, 8.0]], atol=1e-6)

    def test_empty_matches(self):
        """No matches produces empty output."""
        lhs_vals = np.zeros((2, 3, 3))
        rhs_vals = np.zeros((2, 3, 3))
        result = kernel_einsum(
            lhs_vals, rhs_vals, np.array([], dtype=np.int64),
            np.array([], dtype=np.int64), "ij,jk->ik"
        )
        self.assertEqual(result.shape[0], 0)
        self.assertEqual(result.shape[1:], (3, 3))

    def test_outer_product_tiles(self):
        """Per-tile outer product."""
        lhs_vals = np.array([[1.0, 2.0]])  # shape (1, 2)
        rhs_vals = np.array([[3.0, 4.0]])  # shape (1, 2)
        result = kernel_einsum(
            lhs_vals, rhs_vals,
            np.array([0]), np.array([0]),
            "i,j->ij"
        )
        self.assertEqual(result.shape, (1, 2, 2))
        npt.assert_allclose(result[0], [[3.0, 4.0], [6.0, 8.0]], atol=1e-6)

    def test_dot_product_tiles(self):
        """Per-tile dot product (full contraction)."""
        lhs_vals = np.array([[1.0, 2.0, 3.0]])
        rhs_vals = np.array([[4.0, 5.0, 6.0]])
        result = kernel_einsum(
            lhs_vals, rhs_vals,
            np.array([0]), np.array([0]),
            "i,i->"
        )
        self.assertEqual(result.shape, (1,))
        npt.assert_allclose(result[0], 32.0, atol=1e-6)


# =============================================================================
# Segment Sum Tests
# =============================================================================


class TestSegmentSum(unittest.TestCase):
    """Tests for the aggregation phase."""

    def test_no_aggregation_needed(self):
        """All unique coords pass through unchanged."""
        values = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        coords = np.array([[0, 0]], dtype=np.int32)
        out_coords, out_values = segment_sum(values, coords)
        self.assertEqual(out_coords.shape, (1, 2))
        npt.assert_allclose(out_values, values, atol=1e-6)

    def test_aggregation_sums_tiles(self):
        """Tiles with same output coord are summed."""
        values = np.array([
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ])
        coords = np.array([[0, 0], [0, 0]], dtype=np.int32)

        out_coords, out_values = segment_sum(values, coords)
        self.assertEqual(out_coords.shape[0], 1)  # Two inputs → one output
        npt.assert_allclose(out_values[0], [[3.0, 0.0], [0.0, 3.0]], atol=1e-6)

    def test_mixed_aggregation(self):
        """Some coords aggregate, others don't."""
        values = np.array([
            [[1.0]], [[2.0]], [[3.0]],
        ])
        coords = np.array([[0, 0], [0, 0], [1, 0]], dtype=np.int32)

        out_coords, out_values = segment_sum(values, coords)
        self.assertEqual(out_coords.shape[0], 2)
        npt.assert_allclose(out_values[0], [[3.0]], atol=1e-6)
        npt.assert_allclose(out_values[1], [[3.0]], atol=1e-6)

    def test_empty_input(self):
        """Empty values produce empty output."""
        values = np.zeros((0, 2, 2))
        coords = np.zeros((0, 2), dtype=np.int32)
        out_coords, out_values = segment_sum(values, coords)
        self.assertEqual(out_coords.shape[0], 0)
        self.assertEqual(out_values.shape[0], 0)

    def test_sorted_output(self):
        """Output coordinates are sorted."""
        values = np.array([[[1.0]], [[2.0]], [[3.0]]])
        coords = np.array([[2, 0], [0, 0], [1, 0]], dtype=np.int32)
        out_coords, out_values = segment_sum(values, coords)
        self.assertTrue(np.all(out_coords[:-1] <= out_coords[1:]))


# =============================================================================
# End-to-End execute_sparse Tests
# =============================================================================


class TestExecuteSparse(unittest.TestCase):
    """End-to-end tests for sparse tensor-relational execution."""

    def test_sparse_matmul_identity(self):
        """Sparse matmul A @ I = A for identity-like B."""
        A = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        B = np.eye(2)

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = execute_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(2, 2),
            output_tile_shape=(1, 1),
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_sparse_matmul_diagonal(self):
        """Sparse matmul of two diagonal matrices."""
        A = np.diag([1.0, 2.0, 3.0, 4.0])
        B = np.diag([5.0, 6.0, 7.0, 8.0])

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = execute_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(4, 4),
            output_tile_shape=(1, 1),
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_sparse_matmul_tiled(self):
        """Sparse matmul with 2x2 tiles."""
        A = np.array([
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ])
        B = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))

        result = execute_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(4, 4),
            output_tile_shape=(2, 2),
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_sparse_outer_product(self):
        """Outer product of two sparse vectors."""
        a = np.array([1.0, 0.0, 3.0])
        b = np.array([0.0, 2.0, 4.0])

        # Reshape to 2D for tile compatibility
        A = a.reshape(3, 1)
        B = b.reshape(1, 3)

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = execute_sparse(
            lhs, rhs,
            join_keys=[],
            kernel_string="ij,kl->ij",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(3, 3),
            output_tile_shape=(1, 1),
        )

        dense_result = result.to_dense()
        expected = np.outer(a, b)
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_sparse_elementwise(self):
        """Element-wise multiply of two sparse matrices."""
        A = np.diag([1.0, 2.0, 3.0])
        B = np.diag([4.0, 5.0, 6.0])

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        # Element-wise: join on both dims, no aggregation
        result = execute_sparse(
            lhs, rhs,
            join_keys=[(0, 0), (1, 1)],
            kernel_string="ij,ij->ij",
            agg_keys=[(0, 0), (0, 1)],
            output_shape=(3, 3),
            output_tile_shape=(1, 1),
        )

        dense_result = result.to_dense()
        expected = A * B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_empty_result(self):
        """Non-overlapping sparse tensors produce empty result."""
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        B = np.zeros((4, 4))
        B[2, 2] = 1.0

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))

        result = execute_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(4, 4),
            output_tile_shape=(2, 2),
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_inferred_output_shapes(self):
        """Output shape/tile shape inferred when not provided."""
        A = np.diag([1.0, 2.0])
        B = np.diag([3.0, 4.0])

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = execute_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestAddBatchDim(unittest.TestCase):
    """Tests for _add_batch_dim helper."""

    def test_matmul(self):
        self.assertEqual(_add_batch_dim("ij,jk->ik"), "zij,zjk->zik")

    def test_outer_product(self):
        self.assertEqual(_add_batch_dim("i,j->ij"), "zi,zj->zij")

    def test_dot_product(self):
        self.assertEqual(_add_batch_dim("i,i->"), "zi,zi->z")

    def test_no_arrow_raises(self):
        with self.assertRaises(ValueError):
            _add_batch_dim("ij,jk")


class TestInferOutputTileShape(unittest.TestCase):
    """Tests for _infer_output_tile_shape helper."""

    def test_matmul(self):
        shape = _infer_output_tile_shape("ij,jk->ik", (2, 3), (3, 4))
        self.assertEqual(shape, (2, 4))

    def test_outer_product(self):
        shape = _infer_output_tile_shape("i,j->ij", (3,), (4,))
        self.assertEqual(shape, (3, 4))

    def test_dot_product(self):
        shape = _infer_output_tile_shape("i,i->", (5,), (5,))
        self.assertEqual(shape, ())

    def test_trace(self):
        shape = _infer_output_tile_shape("ii,ij->j", (3, 3), (3, 4))
        self.assertEqual(shape, (4,))


class TestComputeOutputCoords(unittest.TestCase):
    """Tests for _compute_output_coords helper."""

    def test_matmul_output_coords(self):
        """Matmul output coord = (lhs row, rhs col)."""
        lhs = SparseTensorRelation(
            coords=np.array([[0, 0], [1, 0]], dtype=np.int32),
            values=np.zeros((2, 2, 2)),
            shape=(4, 4), tile_shape=(2, 2),
        )
        rhs = SparseTensorRelation(
            coords=np.array([[0, 0], [0, 1]], dtype=np.int32),
            values=np.zeros((2, 2, 2)),
            shape=(4, 4), tile_shape=(2, 2),
        )
        lhs_idx = np.array([0, 0, 1, 1])
        rhs_idx = np.array([0, 1, 0, 1])

        out_coords = _compute_output_coords(
            lhs, rhs, lhs_idx, rhs_idx,
            join_keys=[(1, 0)],
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(4, 4),
            output_tile_shape=(2, 2),
        )

        self.assertEqual(out_coords.shape, (4, 2))
        # Match 0: lhs(0,0) rhs(0,0) → output (0,0)
        npt.assert_array_equal(out_coords[0], [0, 0])
        # Match 1: lhs(0,0) rhs(0,1) → output (0,1)
        npt.assert_array_equal(out_coords[1], [0, 1])

    def test_empty_matches(self):
        """No matches produces empty output coords."""
        lhs = SparseTensorRelation(
            coords=np.zeros((0, 2), dtype=np.int32),
            values=np.zeros((0, 2, 2)),
            shape=(4, 4), tile_shape=(2, 2),
        )
        rhs = SparseTensorRelation(
            coords=np.zeros((0, 2), dtype=np.int32),
            values=np.zeros((0, 2, 2)),
            shape=(4, 4), tile_shape=(2, 2),
        )
        out_coords = _compute_output_coords(
            lhs, rhs, np.array([], dtype=np.int64), np.array([], dtype=np.int64),
            join_keys=[(1, 0)],
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(4, 4), output_tile_shape=(2, 2),
        )
        self.assertEqual(out_coords.shape, (0, 2))


# =============================================================================
# Partition Matched Pairs Tests
# =============================================================================


class TestPartitionMatchedPairs(unittest.TestCase):
    """Tests for _partition_matched_pairs helper."""

    def test_even_partition(self):
        """Evenly divisible matches across devices."""
        parts = _partition_matched_pairs(8, 4)
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts, [(0, 2), (2, 4), (4, 6), (6, 8)])

    def test_uneven_partition(self):
        """Non-evenly divisible — remainder distributed to first devices."""
        parts = _partition_matched_pairs(10, 4)
        self.assertEqual(len(parts), 4)
        # 10 // 4 = 2, remainder 2 → first 2 get 3, last 2 get 2
        self.assertEqual(parts[0], (0, 3))
        self.assertEqual(parts[1], (3, 6))
        self.assertEqual(parts[2], (6, 8))
        self.assertEqual(parts[3], (8, 10))

    def test_zero_matches(self):
        """Zero matches produces all-empty partitions."""
        parts = _partition_matched_pairs(0, 4)
        self.assertEqual(len(parts), 4)
        for start, end in parts:
            self.assertEqual(start, 0)
            self.assertEqual(end, 0)

    def test_fewer_matches_than_devices(self):
        """Fewer matches than devices — some devices get nothing."""
        parts = _partition_matched_pairs(3, 8)
        self.assertEqual(len(parts), 8)
        non_empty = [(s, e) for s, e in parts if s != e]
        self.assertEqual(len(non_empty), 3)

    def test_single_device(self):
        """Single device gets all matches."""
        parts = _partition_matched_pairs(10, 1)
        self.assertEqual(parts, [(0, 10)])

    def test_partitions_cover_all(self):
        """All partitions together cover all matches with no gaps or overlaps."""
        for n in [0, 1, 7, 15, 64, 100]:
            for d in [1, 2, 4, 8]:
                parts = _partition_matched_pairs(n, d)
                total = sum(e - s for s, e in parts)
                self.assertEqual(total, n)
                # Check contiguity
                for i in range(len(parts) - 1):
                    self.assertEqual(parts[i][1], parts[i + 1][0])


# =============================================================================
# Sharded Sparse Execution Tests
# =============================================================================


import jax


def _get_test_mesh(num_devices=None):
    """Create a test mesh using available devices."""
    devices = jax.local_devices()
    if num_devices is not None:
        devices = devices[:num_devices]
    return jax.sharding.Mesh(np.array(devices).reshape(-1), ("x",))


class TestExecuteShardedSparse(unittest.TestCase):
    """Tests for execute_sharded_sparse multi-device execution."""

    def test_no_mesh_falls_back(self):
        """Without mesh, falls back to single-device execute_sparse."""
        A = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        B = np.eye(2)

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = execute_sharded_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(2, 2),
            output_tile_shape=(1, 1),
            mesh=None,
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_sharded_matmul_identity(self):
        """Sharded sparse matmul A @ I = A."""
        mesh = _get_test_mesh()

        A = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        B = np.eye(2)

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = execute_sharded_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(2, 2),
            output_tile_shape=(1, 1),
            mesh=mesh,
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_sharded_matmul_diagonal(self):
        """Sharded sparse matmul of two diagonal matrices."""
        mesh = _get_test_mesh()

        A = np.diag([1.0, 2.0, 3.0, 4.0])
        B = np.diag([5.0, 6.0, 7.0, 8.0])

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = execute_sharded_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(4, 4),
            output_tile_shape=(1, 1),
            mesh=mesh,
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_sharded_matmul_tiled(self):
        """Sharded sparse matmul with 2x2 tiles."""
        mesh = _get_test_mesh()

        A = np.array([
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ])
        B = np.eye(4)

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))

        result = execute_sharded_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(4, 4),
            output_tile_shape=(2, 2),
            mesh=mesh,
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_sharded_elementwise(self):
        """Sharded element-wise multiply of two sparse matrices."""
        mesh = _get_test_mesh()

        A = np.diag([1.0, 2.0, 3.0])
        B = np.diag([4.0, 5.0, 6.0])

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = execute_sharded_sparse(
            lhs, rhs,
            join_keys=[(0, 0), (1, 1)],
            kernel_string="ij,ij->ij",
            agg_keys=[(0, 0), (0, 1)],
            output_shape=(3, 3),
            output_tile_shape=(1, 1),
            mesh=mesh,
        )

        dense_result = result.to_dense()
        expected = A * B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_sharded_empty_result(self):
        """Non-overlapping sparse tensors produce empty result (sharded)."""
        mesh = _get_test_mesh()

        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        B = np.zeros((4, 4))
        B[2, 2] = 1.0

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))

        result = execute_sharded_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(4, 4),
            output_tile_shape=(2, 2),
            mesh=mesh,
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_sharded_matches_single_device(self):
        """Sharded result matches single-device result exactly."""
        mesh = _get_test_mesh()

        np.random.seed(42)
        A = np.random.randn(8, 8)
        B = np.random.randn(8, 8)
        # Sparsify: zero out ~50% of entries
        A[A < 0] = 0.0
        B[B < 0] = 0.0

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))

        args = dict(
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(8, 8),
            output_tile_shape=(2, 2),
        )

        single_result = execute_sparse(lhs, rhs, **args)
        sharded_result = execute_sharded_sparse(lhs, rhs, **args, mesh=mesh)

        npt.assert_allclose(
            sharded_result.to_dense(), single_result.to_dense(), atol=1e-5
        )

    def test_sharded_inferred_shapes(self):
        """Output shapes are correctly inferred in sharded mode."""
        mesh = _get_test_mesh()

        A = np.diag([1.0, 2.0])
        B = np.diag([3.0, 4.0])

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = execute_sharded_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            mesh=mesh,
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_sharded_with_aggregation(self):
        """Sharded sparse matmul where tiles require aggregation."""
        mesh = _get_test_mesh()

        # Dense matmul where multiple tile pairs map to the same output tile
        A = np.ones((4, 4))
        B = np.ones((4, 4))

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))

        result = execute_sharded_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(4, 4),
            output_tile_shape=(2, 2),
            mesh=mesh,
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)

    def test_num_devices_parameter(self):
        """Explicit num_devices parameter is respected."""
        mesh = _get_test_mesh()

        A = np.diag([1.0, 2.0])
        B = np.diag([3.0, 4.0])

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = execute_sharded_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            mesh=mesh,
            num_devices=1,
        )

        dense_result = result.to_dense()
        expected = A @ B
        npt.assert_allclose(dense_result, expected, atol=1e-6)


# =============================================================================
# Package Import Tests
# =============================================================================


class TestPackageSparseDispatchImports(unittest.TestCase):
    """Tests for package-level imports of sparse dispatch functions."""

    def test_execution_package_imports(self):
        """Sparse dispatch exports are importable from execution package."""
        from einjax.execution import execute_sparse, coordinate_join
        self.assertIsNotNone(execute_sparse)
        self.assertIsNotNone(coordinate_join)

    def test_top_level_imports(self):
        """Sparse dispatch exports are importable from einjax."""
        from einjax import execute_sparse, coordinate_join
        self.assertIsNotNone(execute_sparse)
        self.assertIsNotNone(coordinate_join)

    def test_sharded_sparse_execution_package_import(self):
        """execute_sharded_sparse is importable from execution package."""
        from einjax.execution import execute_sharded_sparse
        self.assertIsNotNone(execute_sharded_sparse)

    def test_sharded_sparse_top_level_import(self):
        """execute_sharded_sparse is importable from einjax."""
        from einjax import execute_sharded_sparse
        self.assertIsNotNone(execute_sharded_sparse)


if __name__ == "__main__":
    unittest.main()
