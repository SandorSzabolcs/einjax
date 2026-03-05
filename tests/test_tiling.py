"""
Tests for einjax sparse tiling scheme generation.

Tests for tensor/tiling.py: scheme pruning, ranking, selection,
memory estimation, sharding filtering, and end-to-end preparation.
"""

import unittest

import numpy as np

from einjax.core.types import CaseAssignment, TilingScheme
from einjax.tensor.sparse import SparseTensor
from einjax.tensor.dense import DenseTensor
from einjax.tensor.tiling import (
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


# =============================================================================
# Pruning Tests
# =============================================================================


class TestPruneInfeasibleSchemes(unittest.TestCase):
    """Tests for prune_infeasible_schemes."""

    def test_no_pruning_single_device(self):
        """Single-device mode should not prune any schemes."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("s", data)
        pruned = prune_infeasible_schemes(t.schemes, num_devices=1)
        self.assertEqual(len(t.schemes), len(pruned))

    def test_prune_by_memory(self):
        """Schemes with tiles exceeding memory limit are pruned."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("s", data)
        # Only allow tiles up to 4 bytes (1 float32 element)
        pruned = prune_infeasible_schemes(
            t.schemes, num_devices=1, max_tile_bytes=4, dtype_size=4,
        )
        # Only (1,1) tile should fit (tile_size=1, 4 bytes)
        for tile_shape in pruned:
            self.assertEqual(1, pruned[tile_shape].tile_size)

    def test_prune_preserves_feasible(self):
        """Feasible schemes are always preserved."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("s", data)
        # Large memory limit — all schemes should pass
        pruned = prune_infeasible_schemes(
            t.schemes, num_devices=1, max_tile_bytes=1_000_000,
        )
        self.assertEqual(len(t.schemes), len(pruned))

    def test_prune_empty_result(self):
        """All schemes pruned when memory limit is zero."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("s", data)
        pruned = prune_infeasible_schemes(
            t.schemes, num_devices=1, max_tile_bytes=0,
        )
        self.assertEqual(0, len(pruned))


# =============================================================================
# Ranking Tests
# =============================================================================


class TestRankSchemesBySparsity(unittest.TestCase):
    """Tests for rank_schemes_by_sparsity."""

    def test_sparsest_first(self):
        """Most sparse scheme should be ranked first."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("d", data)
        ranked = rank_schemes_by_sparsity(t.schemes, t.shape)
        self.assertGreater(len(ranked), 0)
        # First scheme should have lowest sparsity ratio
        from einjax.tensor.stats import sparsity_ratio
        first_ratio = sparsity_ratio(
            ranked[0].num_tuples, t.shape, ranked[0].tile_shape,
        )
        for scheme in ranked[1:]:
            ratio = sparsity_ratio(
                scheme.num_tuples, t.shape, scheme.tile_shape,
            )
            self.assertGreaterEqual(ratio, first_ratio)

    def test_returns_all_schemes(self):
        """All schemes should be included in the ranking."""
        data = np.diag([1.0, 2.0])
        t = SparseTensor("s", data)
        ranked = rank_schemes_by_sparsity(t.schemes, t.shape)
        self.assertEqual(len(t.schemes), len(ranked))

    def test_tiebreak_by_tile_size(self):
        """Schemes with same sparsity ratio are ranked by tile_size descending."""
        data = np.ones((4, 4))  # Dense — all tiles non-empty → ratio=1.0
        t = DenseTensor("d", data)
        ranked = rank_schemes_by_sparsity(t.schemes, t.shape)
        # All have ratio 1.0, so order should be by tile_size descending
        for i in range(len(ranked) - 1):
            self.assertGreaterEqual(ranked[i].tile_size, ranked[i + 1].tile_size)

    def test_diagonal_prefers_element_tiling(self):
        """For a diagonal matrix, element-level tiling has lowest ratio."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("d", data)
        ranked = rank_schemes_by_sparsity(t.schemes, t.shape)
        # (1,1) tiling: 4 tiles out of 16 → ratio = 0.25
        # (4,4) tiling: 1 tile out of 1 → ratio = 1.0
        # So (1,1) should be first
        self.assertEqual((1, 1), ranked[0].tile_shape)


# =============================================================================
# Selection Tests
# =============================================================================


class TestSelectBestSparseTiling(unittest.TestCase):
    """Tests for select_best_sparse_tiling."""

    def test_returns_scheme(self):
        """Should return a valid TilingScheme."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("d", data)
        best = select_best_sparse_tiling(t.schemes, t.shape)
        self.assertIsNotNone(best)
        self.assertIsInstance(best, TilingScheme)

    def test_returns_none_when_all_pruned(self):
        """Should return None when all schemes are pruned."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("d", data)
        best = select_best_sparse_tiling(
            t.schemes, t.shape, max_tile_bytes=0,
        )
        self.assertIsNone(best)

    def test_selects_sparsest(self):
        """Should select the scheme with lowest sparsity ratio."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("d", data)
        best = select_best_sparse_tiling(t.schemes, t.shape)
        # For a 4x4 diagonal, (1,1) gives 4/16=0.25, best ratio
        self.assertEqual((1, 1), best.tile_shape)

    def test_respects_memory_limit(self):
        """Should only select from schemes within memory budget."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("d", data)
        # Allow tiles up to 16 bytes (4 float32 = 2x2 tile)
        best = select_best_sparse_tiling(
            t.schemes, t.shape, max_tile_bytes=16, dtype_size=4,
        )
        self.assertIsNotNone(best)
        self.assertLessEqual(best.tile_size * 4, 16)


# =============================================================================
# Partition Spec Tests
# =============================================================================


class TestGetSparsePartitionSpec(unittest.TestCase):
    """Tests for get_sparse_partition_spec."""

    def test_fully_sharded(self):
        """All dimensions sharded when tile < shape in every dim."""
        scheme = TilingScheme(None, (4, 4), (2, 2))
        spec = get_sparse_partition_spec(scheme)
        self.assertEqual(("x", "y"), spec)

    def test_fully_local(self):
        """All dimensions local when tile == shape in every dim."""
        scheme = TilingScheme(None, (4, 4), (4, 4))
        spec = get_sparse_partition_spec(scheme)
        self.assertEqual((None, None), spec)

    def test_mixed(self):
        """Mixed sharded and local dimensions."""
        scheme = TilingScheme(None, (4, 4), (2, 4))
        spec = get_sparse_partition_spec(scheme)
        self.assertEqual(("x", None), spec)

    def test_custom_axis_names(self):
        """Custom mesh axis names are used."""
        scheme = TilingScheme(None, (4, 4), (2, 2))
        spec = get_sparse_partition_spec(scheme, mesh_axis_names=("a", "b"))
        self.assertEqual(("a", "b"), spec)


# =============================================================================
# Memory Estimation Tests
# =============================================================================


class TestComputeTileMemory(unittest.TestCase):
    """Tests for compute_tile_memory."""

    def test_basic_2d(self):
        """Test memory for a 2x2 float32 tile."""
        mem = compute_tile_memory((2, 2), dtype_size=4)
        # 4 elements * 4 bytes + 2 coords * 4 bytes = 24
        self.assertEqual(24, mem)

    def test_without_coords(self):
        """Test memory without coordinate overhead."""
        mem = compute_tile_memory((2, 2), dtype_size=4, include_coords=False)
        self.assertEqual(16, mem)

    def test_3d_tile(self):
        """Test memory for a 3D tile."""
        mem = compute_tile_memory((2, 3, 4), dtype_size=4)
        # 24 elements * 4 bytes + 3 coords * 4 bytes = 108
        self.assertEqual(108, mem)

    def test_single_element(self):
        """Test memory for a 1x1 tile."""
        mem = compute_tile_memory((1, 1), dtype_size=4)
        # 1 element * 4 bytes + 2 coords * 4 bytes = 12
        self.assertEqual(12, mem)


class TestComputeRelationMemory(unittest.TestCase):
    """Tests for compute_relation_memory."""

    def test_basic(self):
        """Test relation memory estimation."""
        scheme = TilingScheme(None, (4, 4), (2, 2))
        scheme.num_tuples = 3
        mem = compute_relation_memory(scheme, dtype_size=4)
        # coords: 3 tuples * 2 dims * 4 bytes = 24
        # values: 3 tuples * 4 elements * 4 bytes = 48
        self.assertEqual(72, mem)

    def test_empty_relation(self):
        """Test memory for empty relation."""
        scheme = TilingScheme(None, (4, 4), (2, 2))
        scheme.num_tuples = 0
        mem = compute_relation_memory(scheme)
        self.assertEqual(0, mem)

    def test_single_tile(self):
        """Test memory for single-tile relation."""
        scheme = TilingScheme(None, (4, 4), (4, 4))
        scheme.num_tuples = 1
        mem = compute_relation_memory(scheme, dtype_size=4)
        # coords: 1 * 2 * 4 = 8
        # values: 1 * 16 * 4 = 64
        self.assertEqual(72, mem)


# =============================================================================
# Sharding Filter Tests
# =============================================================================


class TestFilterSchemesBySharding(unittest.TestCase):
    """Tests for filter_schemes_by_sharding."""

    def test_min_sharded_dims(self):
        """Filter requiring at least 1 sharded dimension."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("s", data)
        filtered = filter_schemes_by_sharding(
            t.schemes, num_devices=4, min_sharded_dims=1,
        )
        # Fully local scheme (4,4) should be excluded
        self.assertNotIn((4, 4), filtered)
        # Element-level (1,1) — both dims sharded → should be included
        # unless num_shards > num_devices
        for tile_shape, scheme in filtered.items():
            cases = scheme.get_case_assignments()
            num_sharded = sum(1 for c in cases if c == CaseAssignment.UPPER)
            self.assertGreaterEqual(num_sharded, 1)

    def test_max_sharded_dims(self):
        """Filter limiting to at most 1 sharded dimension."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("s", data)
        filtered = filter_schemes_by_sharding(
            t.schemes, num_devices=4, max_sharded_dims=1,
        )
        for tile_shape, scheme in filtered.items():
            cases = scheme.get_case_assignments()
            num_sharded = sum(1 for c in cases if c == CaseAssignment.UPPER)
            self.assertLessEqual(num_sharded, 1)

    def test_device_limit(self):
        """Schemes needing more shards than devices are excluded."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("s", data)
        # Only 2 devices — schemes needing >2 shards along sharded dims excluded
        filtered = filter_schemes_by_sharding(
            t.schemes, num_devices=2, min_sharded_dims=1,
        )
        for tile_shape, scheme in filtered.items():
            cases = scheme.get_case_assignments()
            from math import prod
            num_shards = prod(
                s // t for s, t, c in zip(scheme.shape, scheme.tile_shape, cases)
                if c == CaseAssignment.UPPER
            )
            self.assertLessEqual(num_shards, 2)

    def test_no_constraints(self):
        """No filtering when min=0 and max=None."""
        data = np.diag([1.0, 2.0])
        t = SparseTensor("s", data)
        filtered = filter_schemes_by_sharding(
            t.schemes, num_devices=1,
        )
        self.assertEqual(len(t.schemes), len(filtered))


# =============================================================================
# Device Count Filter Tests (PRD 4.3)
# =============================================================================


class TestFilterSchemesByDeviceCount(unittest.TestCase):
    """Tests for filter_schemes_by_device_count (PRD Section 4.3)."""

    def test_single_device_no_filtering(self):
        """Single-device mode should return all schemes unchanged."""
        data = np.ones((8, 8))
        t = DenseTensor("d", data)
        filtered = filter_schemes_by_device_count(t.schemes, num_devices=1)
        self.assertEqual(len(t.schemes), len(filtered))

    def test_fully_local_always_kept(self):
        """Schemes with no UPPER-case dims (tile == shape) are always kept."""
        data = np.ones((8, 8))
        t = DenseTensor("d", data)
        filtered = filter_schemes_by_device_count(t.schemes, num_devices=8)
        # The (8, 8) tile_shape (fully local) should always be kept
        self.assertIn((8, 8), filtered)

    def test_8_devices_requires_8_upper_tiles(self):
        """With 8 devices, all sharded schemes must have >= 8 UPPER tiles."""
        data = np.ones((16, 16))
        t = DenseTensor("d", data)
        filtered = filter_schemes_by_device_count(t.schemes, num_devices=8)
        for tile_shape, scheme in filtered.items():
            cases = scheme.get_case_assignments()
            num_upper = sum(1 for c in cases if c == CaseAssignment.UPPER)
            if num_upper > 0:
                from math import prod as mprod
                num_upper_tiles = mprod(
                    s // ts for s, ts, c in zip(scheme.shape, scheme.tile_shape, cases)
                    if c == CaseAssignment.UPPER
                )
                self.assertGreaterEqual(num_upper_tiles, 8)

    def test_divisibility_constraint(self):
        """UPPER tiles must be evenly divisible by num_devices."""
        data = np.ones((16, 16))
        t = DenseTensor("d", data)
        filtered = filter_schemes_by_device_count(t.schemes, num_devices=8)
        for tile_shape, scheme in filtered.items():
            cases = scheme.get_case_assignments()
            num_upper = sum(1 for c in cases if c == CaseAssignment.UPPER)
            if num_upper > 0:
                from math import prod as mprod
                num_upper_tiles = mprod(
                    s // ts for s, ts, c in zip(scheme.shape, scheme.tile_shape, cases)
                    if c == CaseAssignment.UPPER
                )
                self.assertEqual(num_upper_tiles % 8, 0)

    def test_prunes_insufficient_tiles(self):
        """Schemes with fewer UPPER tiles than devices are pruned."""
        data = np.ones((8, 8))
        t = DenseTensor("d", data)
        filtered = filter_schemes_by_device_count(t.schemes, num_devices=8)
        # tile_shape (4, 8): only 2 upper tiles (dim 0: 8/4=2), should be pruned
        self.assertNotIn((4, 8), filtered)
        # tile_shape (8, 4): only 2 upper tiles (dim 1: 8/4=2), should be pruned
        self.assertNotIn((8, 4), filtered)

    def test_prunes_indivisible_tiles(self):
        """Schemes where UPPER tiles aren't divisible by devices are pruned."""
        # Shape (12, 12) with 8 devices
        data = np.ones((12, 12))
        t = DenseTensor("d", data)
        filtered = filter_schemes_by_device_count(t.schemes, num_devices=8)
        for tile_shape, scheme in filtered.items():
            cases = scheme.get_case_assignments()
            num_upper = sum(1 for c in cases if c == CaseAssignment.UPPER)
            if num_upper > 0:
                from math import prod as mprod
                num_upper_tiles = mprod(
                    s // ts for s, ts, c in zip(scheme.shape, scheme.tile_shape, cases)
                    if c == CaseAssignment.UPPER
                )
                self.assertEqual(num_upper_tiles % 8, 0)

    def test_min_tiles_per_device(self):
        """min_tiles_per_device raises the minimum UPPER tiles threshold."""
        data = np.ones((16, 16))
        t = DenseTensor("d", data)
        # With min_tiles_per_device=2 and 8 devices, need >= 16 UPPER tiles
        filtered = filter_schemes_by_device_count(
            t.schemes, num_devices=8, min_tiles_per_device=2,
        )
        for tile_shape, scheme in filtered.items():
            cases = scheme.get_case_assignments()
            num_upper = sum(1 for c in cases if c == CaseAssignment.UPPER)
            if num_upper > 0:
                from math import prod as mprod
                num_upper_tiles = mprod(
                    s // ts for s, ts, c in zip(scheme.shape, scheme.tile_shape, cases)
                    if c == CaseAssignment.UPPER
                )
                self.assertGreaterEqual(num_upper_tiles, 16)

    def test_concrete_example_8_devices(self):
        """Concrete example: shape (8,8) with 8 devices.

        tile_shape (1,8) → 8 upper tiles along dim 0 → valid (8 % 8 == 0)
        tile_shape (1,1) → 64 upper tiles → valid (64 % 8 == 0)
        tile_shape (2,8) → 4 upper tiles → pruned (4 < 8)
        """
        data = np.ones((8, 8))
        t = DenseTensor("d", data)
        filtered = filter_schemes_by_device_count(t.schemes, num_devices=8)
        self.assertIn((1, 8), filtered)  # 8 tiles, divisible
        self.assertIn((1, 1), filtered)  # 64 tiles, divisible
        self.assertNotIn((2, 8), filtered)  # 4 tiles < 8
        self.assertIn((8, 8), filtered)  # fully local, always kept

    def test_2d_sharding(self):
        """Schemes sharded on 2 dims: product of both counts must meet threshold."""
        data = np.ones((16, 16))
        t = DenseTensor("d", data)
        filtered = filter_schemes_by_device_count(t.schemes, num_devices=8)
        # tile_shape (4, 4): 4*4 = 16 upper tiles, 16 % 8 == 0 → valid
        self.assertIn((4, 4), filtered)
        # tile_shape (8, 8): 2*2 = 4 upper tiles < 8 → pruned
        self.assertNotIn((8, 8), filtered)


# =============================================================================
# End-to-End Preparation Tests
# =============================================================================


class TestPrepareSparserTiling(unittest.TestCase):
    """Tests for prepare_sparse_tiling."""

    def test_returns_scheme(self):
        """Should return a valid TilingScheme."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("d", data)
        best = prepare_sparse_tiling(t)
        self.assertIsNotNone(best)
        self.assertIsInstance(best, TilingScheme)

    def test_sparse_prefers_fine_tiling(self):
        """For a sparse diagonal, fine tiling is preferred."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("d", data)
        best = prepare_sparse_tiling(t)
        # Element-level tiling should be best for diagonal
        self.assertEqual((1, 1), best.tile_shape)

    def test_dense_prefers_coarse_tiling(self):
        """For a fully dense tensor, coarsest tiling is preferred.

        All tiles are non-empty (ratio=1.0), so tiebreak by largest tile.
        """
        data = np.ones((4, 4))
        t = DenseTensor("d", data)
        best = prepare_sparse_tiling(t)
        # Fully dense: all ratios are 1.0, tiebreak by tile_size desc
        self.assertEqual((4, 4), best.tile_shape)

    def test_with_memory_limit(self):
        """Memory limit is respected during preparation."""
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("d", data)
        best = prepare_sparse_tiling(t, max_tile_bytes=4, dtype_size=4)
        self.assertIsNotNone(best)
        self.assertLessEqual(best.tile_size * 4, 4)

    def test_1d_sparse(self):
        """Works with 1D sparse tensors."""
        data = np.array([0.0, 1.0, 0.0, 3.0, 0.0, 0.0])
        t = SparseTensor("v", data)
        best = prepare_sparse_tiling(t)
        self.assertIsNotNone(best)
        self.assertIsInstance(best, TilingScheme)


# =============================================================================
# Package Import Tests
# =============================================================================


class TestPackageTilingImports(unittest.TestCase):
    """Test that tiling functions are importable from einjax."""

    def test_top_level_imports(self):
        """Test tiling functions importable from einjax."""
        import einjax
        self.assertIsNotNone(einjax.prune_infeasible_schemes)
        self.assertIsNotNone(einjax.rank_schemes_by_sparsity)
        self.assertIsNotNone(einjax.select_best_sparse_tiling)
        self.assertIsNotNone(einjax.get_sparse_partition_spec)
        self.assertIsNotNone(einjax.compute_tile_memory)
        self.assertIsNotNone(einjax.compute_relation_memory)
        self.assertIsNotNone(einjax.filter_schemes_by_sharding)
        self.assertIsNotNone(einjax.filter_schemes_by_device_count)
        self.assertIsNotNone(einjax.prepare_sparse_tiling)

    def test_tensor_package_imports(self):
        """Test imports from einjax.tensor package."""
        from einjax.tensor import (
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
        self.assertIsNotNone(prune_infeasible_schemes)
        self.assertIsNotNone(filter_schemes_by_device_count)
        self.assertIsNotNone(prepare_sparse_tiling)


if __name__ == "__main__":
    unittest.main()
