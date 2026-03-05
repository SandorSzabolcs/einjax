"""
Tests for einjax sparse tensor layer: SparseTensor, SparseTensorRelation,
and sparsity statistics functions.

Ported from einsql/test.py TestSparseTensor (lines 200-236) and
TestSparseSQLGeneration (lines 290-387). Assertions use NumPy array
comparisons instead of SQL string matching.
"""

import unittest

import numpy as np

from einjax.tensor.sparse import SparseTensor, SparseTensorRelation, _extract_coo
from einjax.tensor.stats import (
    compute_sparsity_stats_coo,
    compute_sparsity_stats_dense,
    sparsity_ratio,
    update_scheme_sparsity,
)
from einjax.core.types import CaseAssignment, TilingScheme


# =============================================================================
# SparseTensorRelation Tests
# =============================================================================


class TestSparseTensorRelation(unittest.TestCase):
    """Tests for the tensor-relational sparse representation."""

    def test_construction(self):
        """Test basic construction of a SparseTensorRelation."""
        coords = np.array([[0, 0], [1, 1]], dtype=np.int32)
        values = np.zeros((2, 2, 2), dtype=np.float64)
        values[0, 0, 0] = 1.0
        values[1, 1, 1] = 2.0
        rel = SparseTensorRelation(
            coords=coords,
            values=values,
            shape=(4, 4),
            tile_shape=(2, 2),
        )
        self.assertEqual(2, rel.num_tuples)
        self.assertEqual(2, rel.ndim)
        self.assertEqual(4, rel.tile_size)
        self.assertEqual((4, 4), rel.shape)
        self.assertEqual((2, 2), rel.tile_shape)

    def test_nnz(self):
        """Test non-zero element counting."""
        coords = np.array([[0, 0]], dtype=np.int32)
        values = np.array([[[1.0, 0.0], [0.0, 3.0]]])
        rel = SparseTensorRelation(
            coords=coords, values=values,
            shape=(2, 2), tile_shape=(2, 2),
        )
        self.assertEqual(2, rel.nnz)

    def test_density(self):
        """Test density computation."""
        coords = np.array([[0, 0]], dtype=np.int32)
        # 2 non-zeros in a 4x4 tensor
        values = np.zeros((1, 2, 2))
        values[0, 0, 0] = 1.0
        values[0, 1, 1] = 2.0
        rel = SparseTensorRelation(
            coords=coords, values=values,
            shape=(4, 4), tile_shape=(2, 2),
        )
        # density = 2 / 16 = 0.125
        self.assertAlmostEqual(0.125, rel.density)

    def test_to_dense(self):
        """Test reconstruction from tiles to dense array."""
        coords = np.array([[0, 0], [1, 1]], dtype=np.int32)
        values = np.zeros((2, 2, 2))
        values[0, 0, 0] = 1.0
        values[0, 0, 1] = 2.0
        values[1, 1, 0] = 5.0
        rel = SparseTensorRelation(
            coords=coords, values=values,
            shape=(4, 4), tile_shape=(2, 2),
        )
        dense = rel.to_dense()
        self.assertEqual((4, 4), dense.shape)
        self.assertEqual(1.0, dense[0, 0])
        self.assertEqual(2.0, dense[0, 1])
        self.assertEqual(5.0, dense[3, 2])
        # Tiles not in coords should be zero
        self.assertEqual(0.0, dense[0, 2])
        self.assertEqual(0.0, dense[2, 0])

    def test_empty_relation(self):
        """Test empty SparseTensorRelation (no non-zero tiles)."""
        coords = np.zeros((0, 2), dtype=np.int32)
        values = np.zeros((0, 3, 3))
        rel = SparseTensorRelation(
            coords=coords, values=values,
            shape=(6, 6), tile_shape=(3, 3),
        )
        self.assertEqual(0, rel.num_tuples)
        self.assertEqual(0, rel.nnz)
        self.assertAlmostEqual(0.0, rel.density)
        dense = rel.to_dense()
        np.testing.assert_array_equal(np.zeros((6, 6)), dense)

    def test_to_dense_3d(self):
        """Test to_dense for a 3D tensor."""
        coords = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int32)
        values = np.zeros((2, 2, 2, 2))
        values[0, 0, 0, 0] = 1.0
        values[1, 1, 1, 1] = 9.0
        rel = SparseTensorRelation(
            coords=coords, values=values,
            shape=(2, 2, 4), tile_shape=(2, 2, 2),
        )
        dense = rel.to_dense()
        self.assertEqual((2, 2, 4), dense.shape)
        self.assertEqual(1.0, dense[0, 0, 0])
        self.assertEqual(9.0, dense[1, 1, 3])


# =============================================================================
# SparseTensor Tests
# =============================================================================


class TestSparseTensor(unittest.TestCase):
    """Tests for SparseTensor with COO-format data.

    Ported from einsql/test.py TestSparseTensor (lines 200-236).
    """

    def test_sparse_from_dense_array(self):
        """Test constructing a SparseTensor from a dense array."""
        data = np.array([[1.0, 0.0, 0.0],
                         [0.0, 2.0, 0.0],
                         [0.0, 0.0, 3.0]])
        t = SparseTensor("s", data)
        self.assertEqual("s", t.name)
        self.assertEqual((3, 3), t.shape)
        self.assertEqual(3, t.nnz)

    def test_sparsity_metrics_identity_tiling(self):
        """Test sparsity metrics for identity (full-size) tiling.

        With tile_shape == shape, the entire tensor is one tile.
        If any element is non-zero, num_tuples = 1.
        """
        data = np.array([[1.0, 0.0], [0.0, 2.0]])
        t = SparseTensor("s", data)
        scheme = t.schemes[t.shape]
        self.assertEqual(1, scheme.num_tuples)

    def test_sparsity_metrics_element_tiling(self):
        """Test sparsity metrics at element-level tiling (tile_shape = (1,...,1)).

        Each tile is a single element; num_tuples = number of non-zeros.
        """
        data = np.array([[1.0, 0.0, 0.0],
                         [0.0, 2.0, 0.0],
                         [0.0, 0.0, 3.0]])
        t = SparseTensor("s", data)
        scheme = t.schemes[(1, 1)]
        self.assertEqual(3, scheme.num_tuples)

    def test_sparsity_metrics_diagonal(self):
        """Test sparsity metrics for a diagonal matrix with (2,2) tiling.

        A 4x4 diagonal matrix with (2,2) tiles: only tiles on the
        block diagonal are non-empty → 2 non-empty tiles.
        """
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("d", data)
        scheme = t.schemes[(2, 2)]
        self.assertEqual(2, scheme.num_tuples)
        self.assertEqual((2, 2), scheme.value_count)

    def test_value_count_row_sparse(self):
        """Test value_count for a row-sparse matrix.

        A 4x4 matrix with non-zeros only in rows 0 and 1:
        With (2,2) tiling, value_count[0]=1 (only tile row 0),
        value_count[1]=2 (both tile columns).
        """
        data = np.zeros((4, 4))
        data[0, 0] = 1.0
        data[0, 3] = 2.0
        data[1, 1] = 3.0
        t = SparseTensor("r", data)
        scheme = t.schemes[(2, 2)]
        self.assertEqual((1, 2), scheme.value_count)
        self.assertEqual(2, scheme.num_tuples)

    def test_all_zeros_sparse(self):
        """Test SparseTensor with all-zero data."""
        data = np.zeros((3, 3))
        t = SparseTensor("z", data)
        scheme = t.schemes[(1, 1)]
        self.assertEqual(0, scheme.num_tuples)

    def test_1d_sparse(self):
        """Test SparseTensor with 1D data."""
        data = np.array([0.0, 1.0, 0.0, 3.0, 0.0, 0.0])
        t = SparseTensor("v", data)
        self.assertEqual((6,), t.shape)
        scheme = t.schemes[(1,)]
        self.assertEqual(2, scheme.num_tuples)
        scheme2 = t.schemes[(2,)]
        self.assertEqual(2, scheme2.num_tuples)  # NZ in tiles [0] and [1]
        scheme3 = t.schemes[(3,)]
        self.assertEqual(2, scheme3.num_tuples)  # NZ in tiles [0] and [1]


class TestSparseTensorToRelation(unittest.TestCase):
    """Tests for SparseTensor.to_relation() conversion."""

    def test_to_relation_identity_tiling(self):
        """Test to_relation with tile_shape == shape (single tile)."""
        data = np.array([[1.0, 0.0], [0.0, 2.0]])
        t = SparseTensor("s", data)
        rel = t.to_relation((2, 2))
        self.assertEqual(1, rel.num_tuples)
        np.testing.assert_array_equal([[0, 0]], rel.coords)
        np.testing.assert_array_equal(data, rel.values[0])

    def test_to_relation_diagonal(self):
        """Test to_relation for a diagonal matrix.

        4x4 diagonal with (2,2) tiles → 2 non-empty tiles on the
        block diagonal.
        """
        data = np.diag([1.0, 2.0, 3.0, 4.0])
        t = SparseTensor("d", data)
        rel = t.to_relation((2, 2))
        self.assertEqual(2, rel.num_tuples)
        np.testing.assert_array_equal([[0, 0], [1, 1]], rel.coords)
        np.testing.assert_array_equal([[1, 0], [0, 2]], rel.values[0])
        np.testing.assert_array_equal([[3, 0], [0, 4]], rel.values[1])

    def test_to_relation_skips_empty_tiles(self):
        """Test that empty tiles are not included in the relation."""
        data = np.zeros((4, 4))
        data[0, 0] = 1.0
        t = SparseTensor("s", data)
        rel = t.to_relation((2, 2))
        # Only tile (0,0) has a non-zero
        self.assertEqual(1, rel.num_tuples)
        np.testing.assert_array_equal([[0, 0]], rel.coords)

    def test_to_relation_roundtrip(self):
        """Test that to_relation → to_dense roundtrips correctly."""
        data = np.array([[1.0, 0.0, 0.0, 2.0],
                         [0.0, 3.0, 0.0, 0.0],
                         [0.0, 0.0, 4.0, 0.0],
                         [5.0, 0.0, 0.0, 6.0]])
        t = SparseTensor("rt", data)
        rel = t.to_relation((2, 2))
        dense = rel.to_dense()
        np.testing.assert_array_equal(data, dense)

    def test_to_relation_non_divisible_error(self):
        """Test that non-divisible tile shape raises ValueError."""
        data = np.array([[1.0, 0.0, 0.0],
                         [0.0, 2.0, 0.0],
                         [0.0, 0.0, 3.0]])
        t = SparseTensor("s", data)
        with self.assertRaises(ValueError):
            t.to_relation((2, 2))

    def test_to_relation_empty_tensor(self):
        """Test to_relation for all-zero tensor."""
        data = np.zeros((4, 4))
        t = SparseTensor("z", data)
        rel = t.to_relation((2, 2))
        self.assertEqual(0, rel.num_tuples)
        self.assertEqual((0, 2), rel.coords.shape)
        self.assertEqual((0, 2, 2), rel.values.shape)

    def test_to_relation_3d(self):
        """Test to_relation for a 3D sparse tensor."""
        data = np.zeros((4, 4, 4))
        data[0, 0, 0] = 1.0
        data[3, 3, 3] = 2.0
        t = SparseTensor("t3", data)
        rel = t.to_relation((2, 2, 2))
        self.assertEqual(2, rel.num_tuples)
        np.testing.assert_array_equal([[0, 0, 0], [1, 1, 1]], rel.coords)
        dense = rel.to_dense()
        np.testing.assert_array_equal(data, dense)


class TestSparseTensorDenseArray(unittest.TestCase):
    """Tests for SparseTensor.to_dense_array() conversion."""

    def test_to_dense_array(self):
        """Test to_dense_array reconstructs correctly."""
        data = np.array([[1.0, 0.0], [0.0, 2.0]])
        t = SparseTensor("s", data)
        dense = t.to_dense_array()
        np.testing.assert_array_equal(data, dense)

    def test_to_dense_array_zeros(self):
        """Test to_dense_array with all zeros."""
        data = np.zeros((3, 3))
        t = SparseTensor("z", data)
        dense = t.to_dense_array()
        np.testing.assert_array_equal(data, dense)


# =============================================================================
# COO Extraction Tests
# =============================================================================


class TestExtractCOO(unittest.TestCase):
    """Tests for _extract_coo helper."""

    def test_from_dense_array(self):
        """Test COO extraction from a dense array."""
        data = np.array([[1.0, 0.0], [0.0, 2.0]])
        indices, values, shape = _extract_coo(data)
        self.assertEqual((2, 2), shape)
        self.assertEqual(2, len(values))
        self.assertEqual(np.int32, indices.dtype)
        # Check the non-zero coordinates
        coords_set = {tuple(indices[i]) for i in range(len(values))}
        self.assertIn((0, 0), coords_set)
        self.assertIn((1, 1), coords_set)

    def test_from_dense_zeros(self):
        """Test COO extraction from an all-zero array."""
        data = np.zeros((3, 3))
        indices, values, shape = _extract_coo(data)
        self.assertEqual((3, 3), shape)
        self.assertEqual(0, len(values))
        self.assertEqual((0, 2), indices.shape)

    def test_from_1d_array(self):
        """Test COO extraction from a 1D array."""
        data = np.array([0.0, 1.0, 0.0, 3.0])
        indices, values, shape = _extract_coo(data)
        self.assertEqual((4,), shape)
        self.assertEqual(2, len(values))

    def test_scipy_sparse(self):
        """Test COO extraction from scipy.sparse matrix."""
        try:
            import scipy.sparse as sp
        except ImportError:
            self.skipTest("scipy not installed")
        mat = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
        indices, values, shape = _extract_coo(mat)
        self.assertEqual((2, 2), shape)
        self.assertEqual(2, len(values))


# =============================================================================
# Sparsity Statistics Tests
# =============================================================================


class TestComputeSparsityStatsCOO(unittest.TestCase):
    """Tests for compute_sparsity_stats_coo."""

    def test_diagonal_matrix(self):
        """Test sparsity stats for a diagonal matrix."""
        indices = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32)
        values = np.array([1.0, 2.0, 3.0, 4.0])
        shape = (4, 4)
        stats = compute_sparsity_stats_coo(
            indices, values, shape,
            tile_shapes=[(1, 1), (2, 2), (4, 4)],
        )
        # Element-level: 4 non-empty tiles
        self.assertEqual(4, stats[(1, 1)][0])
        # 2x2 tiling: 2 non-empty tiles (block diagonal)
        self.assertEqual(2, stats[(2, 2)][0])
        # Full tensor: 1 tile
        self.assertEqual(1, stats[(4, 4)][0])

    def test_value_count(self):
        """Test V(l, U) computation."""
        # Non-zeros in row 0 and column 0 only
        indices = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32)
        values = np.array([1.0, 2.0, 3.0])
        shape = (4, 4)
        stats = compute_sparsity_stats_coo(
            indices, values, shape,
            tile_shapes=[(2, 2)],
        )
        num_tuples, value_count = stats[(2, 2)]
        # Tile (0,0) has all 3 entries → 1 tile
        self.assertEqual(1, num_tuples)
        self.assertEqual(1, value_count[0])  # V(row, U) = 1
        self.assertEqual(1, value_count[1])  # V(col, U) = 1

    def test_empty_values(self):
        """Test with no non-zeros."""
        indices = np.zeros((0, 2), dtype=np.int32)
        values = np.zeros((0,))
        shape = (4, 4)
        stats = compute_sparsity_stats_coo(
            indices, values, shape,
            tile_shapes=[(2, 2)],
        )
        self.assertEqual(0, stats[(2, 2)][0])

    def test_skips_zero_values(self):
        """Test that explicitly stored zeros are skipped."""
        indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
        values = np.array([1.0, 0.0])  # second entry is explicit zero
        shape = (4, 4)
        stats = compute_sparsity_stats_coo(
            indices, values, shape,
            tile_shapes=[(2, 2)],
        )
        self.assertEqual(1, stats[(2, 2)][0])  # only tile (0,0)


class TestComputeSparsityStatsDense(unittest.TestCase):
    """Tests for compute_sparsity_stats_dense."""

    def test_dense_matches_coo(self):
        """Test that dense stats match COO stats for same data."""
        data = np.array([[1.0, 0.0, 0.0, 2.0],
                         [0.0, 3.0, 0.0, 0.0],
                         [0.0, 0.0, 4.0, 0.0],
                         [5.0, 0.0, 0.0, 6.0]])
        tile_shapes = [(1, 1), (2, 2), (4, 4)]
        dense_stats = compute_sparsity_stats_dense(data, tile_shapes)

        nz = np.nonzero(data)
        indices = np.stack(nz, axis=1).astype(np.int32)
        values = data[nz]
        coo_stats = compute_sparsity_stats_coo(
            indices, values, tuple(data.shape), tile_shapes,
        )

        for ts in tile_shapes:
            self.assertEqual(dense_stats[ts], coo_stats[ts])

    def test_all_zeros(self):
        """Test dense stats for all-zero array."""
        data = np.zeros((4, 4))
        stats = compute_sparsity_stats_dense(data, [(2, 2)])
        self.assertEqual(0, stats[(2, 2)][0])
        self.assertEqual((0, 0), stats[(2, 2)][1])


class TestSparsityRatio(unittest.TestCase):
    """Tests for sparsity_ratio helper."""

    def test_full_density(self):
        """Test sparsity ratio of 1.0 for fully dense tensor."""
        # 4 tiles max, 4 non-empty
        self.assertAlmostEqual(
            1.0, sparsity_ratio(4, (4, 4), (2, 2))
        )

    def test_half_density(self):
        """Test sparsity ratio of 0.5."""
        self.assertAlmostEqual(
            0.5, sparsity_ratio(2, (4, 4), (2, 2))
        )

    def test_zero_density(self):
        """Test sparsity ratio of 0.0."""
        self.assertAlmostEqual(
            0.0, sparsity_ratio(0, (4, 4), (2, 2))
        )


class TestUpdateSchemeSparisty(unittest.TestCase):
    """Tests for update_scheme_sparsity helper."""

    def test_updates_fields(self):
        """Test that scheme fields are updated correctly."""
        from einjax.tensor.base import BaseTensor
        t = BaseTensor("t", (4, 4))
        scheme = t.schemes[(2, 2)]
        update_scheme_sparsity(scheme, 3, (2, 2))
        self.assertEqual(3, scheme.num_tuples)
        self.assertEqual((2, 2), scheme.value_count)


# =============================================================================
# Package Import Tests
# =============================================================================


class TestPackageSparseImports(unittest.TestCase):
    """Test that sparse classes are importable from einjax."""

    def test_top_level_sparse_tensor(self):
        """Test SparseTensor importable from einjax."""
        import einjax
        self.assertIsNotNone(einjax.SparseTensor)

    def test_top_level_sparse_tensor_relation(self):
        """Test SparseTensorRelation importable from einjax."""
        import einjax
        self.assertIsNotNone(einjax.SparseTensorRelation)

    def test_top_level_stats(self):
        """Test sparsity stats importable from einjax."""
        import einjax
        self.assertIsNotNone(einjax.compute_sparsity_stats_coo)
        self.assertIsNotNone(einjax.compute_sparsity_stats_dense)
        self.assertIsNotNone(einjax.sparsity_ratio)
        self.assertIsNotNone(einjax.update_scheme_sparsity)

    def test_tensor_package_imports(self):
        """Test imports from einjax.tensor package."""
        from einjax.tensor import SparseTensor, SparseTensorRelation
        from einjax.tensor import compute_sparsity_stats_coo, sparsity_ratio
        self.assertIsNotNone(SparseTensor)
        self.assertIsNotNone(SparseTensorRelation)
        self.assertIsNotNone(compute_sparsity_stats_coo)
        self.assertIsNotNone(sparsity_ratio)


if __name__ == "__main__":
    unittest.main()
