"""
Tests for einjax tensor layer: BaseTensor, IndexedTerm, DenseTensor.

Ported from einsql/test.py TestDenseTensor (lines 237-267). Assertions
use NumPy arrays instead of PyTorch tensors.
"""

import unittest

import numpy as np

from einjax.tensor.base import BaseTensor, IndexedTerm
from einjax.tensor.dense import DenseTensor
from einjax.core.types import (
    BinaryOp,
    CaseAssignment,
    TilingScheme,
)


class TestBaseTensor(unittest.TestCase):
    """Tests for BaseTensor ABC."""

    def test_shape_and_name(self):
        """Test that shape and name are stored correctly."""
        t = BaseTensor("x", (4, 8))
        self.assertEqual("x", t.name)
        self.assertEqual((4, 8), t.shape)
        self.assertEqual(2, t.ndim)

    def test_flops_zero(self):
        """Test that a base tensor has zero flops."""
        t = BaseTensor("x", (3, 3))
        self.assertEqual(0.0, t.flops())

    def test_tiling_schemes_generated(self):
        """Test that all tiling schemes are generated from shape factors."""
        t = BaseTensor("x", (6, 4))
        # 6 has factors [1,2,3,6], 4 has factors [1,2,4]
        # Cartesian product: 4 * 3 = 12 tile shapes
        self.assertEqual(12, len(t.schemes))
        self.assertIn((1, 1), t.tile_shapes)
        self.assertIn((6, 4), t.tile_shapes)
        self.assertIn((2, 2), t.tile_shapes)

    def test_operator_overloads(self):
        """Test that +, -, / produce BinaryOp nodes."""
        a = BaseTensor("a", (3, 3))
        b = BaseTensor("b", (3, 3))
        self.assertIsInstance(a + b, BinaryOp)
        self.assertIsInstance(a - b, BinaryOp)
        self.assertIsInstance(a / b, BinaryOp)

    def test_getitem_creates_indexed_term(self):
        """Test that tensor['ij'] creates an IndexedTerm."""
        t = BaseTensor("t", (3, 4))
        it = t["ij"]
        self.assertIsInstance(it, IndexedTerm)
        self.assertIs(t, it.tensor)
        self.assertEqual("ij", it.indices)


class TestDenseTensor(unittest.TestCase):
    """Tests for DenseTensor with NumPy arrays.

    Ported from einsql/test.py TestDenseTensor (lines 237-267).
    """

    def test_shape_from_data(self):
        """Test that shape is inferred from the data array."""
        data = np.zeros((3, 4, 5))
        t = DenseTensor("x", data)
        self.assertEqual((3, 4, 5), t.shape)
        self.assertEqual("x", t.name)
        self.assertEqual(3, t.ndim)

    def test_tile_value_generation(self):
        """Test sparsity metrics for a fully-dense 3D tensor.

        Ported from einsql TestDenseTensor.test_tile_value_generation.
        A 4x4x4 tensor with all non-zero values: tiling (2,2,2) yields
        8 non-zero tiles (2*2*2).
        """
        data = np.arange(1, 65).reshape((4, 4, 4)).astype(float)
        t = DenseTensor("a", data)
        scheme = t.schemes[(2, 2, 2)]
        self.assertEqual(8, scheme.num_tuples)

    def test_sparsity_detection(self):
        """Test sparsity detection in dense tensors with zeros.

        Ported from einsql TestDenseTensor.test_sparsity_detection.
        A 3x3 matrix with specific zeros should produce known sparsity
        metrics for each tiling scheme.
        """
        data = np.array([[1.0, 2, 0], [3.0, 4, 0], [5.0, 0, 0]])
        t = DenseTensor("a", data)

        self.assertEqual(4, len(t.schemes))

        expected = {
            (1, 1): (5, 3, 2),   # 5 non-zero elements
            (1, 3): (3, 3, 1),   # 3 non-zero rows
            (3, 1): (2, 1, 2),   # 2 non-zero columns
            (3, 3): (1, 1, 1),   # 1 tile (the whole matrix)
        }
        for _, scheme in t.schemes.items():
            self.assertIn(scheme.tile_shape, expected)
            ans = expected[scheme.tile_shape]
            self.assertEqual(ans[0], scheme.num_tuples)
            self.assertEqual(ans[1], scheme.value_count[0])
            self.assertEqual(ans[2], scheme.value_count[1])

    def test_case_assignments(self):
        """Test case assignments derived from tiling schemes."""
        data = np.ones((12, 8))
        t = DenseTensor("t", data)

        # Tile (4, 8): dim 0 is UPPER (4 < 12), dim 1 is LOWER (8 == 8)
        scheme = t.schemes[(4, 8)]
        cases = scheme.get_case_assignments()
        self.assertEqual(CaseAssignment.UPPER, cases[0])
        self.assertEqual(CaseAssignment.LOWER, cases[1])

        # Tile (12, 8): identity tiling, all LOWER
        scheme = t.schemes[(12, 8)]
        cases = scheme.get_case_assignments()
        self.assertEqual([CaseAssignment.LOWER, CaseAssignment.LOWER], cases)

    def test_data_preserved(self):
        """Test that the original data reference is preserved."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = DenseTensor("t", data)
        self.assertIs(data, t.data)

    def test_all_zero_tensor(self):
        """Test sparsity metrics for an all-zero tensor."""
        data = np.zeros((4, 4))
        t = DenseTensor("z", data)
        # All tiles should have 0 non-zero tuples at element-level tiling
        scheme = t.schemes[(1, 1)]
        self.assertEqual(0, scheme.num_tuples)

    def test_1d_tensor(self):
        """Test DenseTensor works with 1D arrays."""
        data = np.array([1.0, 0.0, 3.0, 0.0, 5.0, 6.0])
        t = DenseTensor("v", data)
        self.assertEqual((6,), t.shape)
        self.assertEqual(1, t.ndim)
        # Element-level tiling: 4 non-zero elements
        scheme = t.schemes[(1,)]
        self.assertEqual(4, scheme.num_tuples)

    def test_identity_tiling_always_one_tuple(self):
        """Test that identity tiling (tile == shape) gives num_tuples=1
        when tensor has any non-zero element."""
        data = np.array([[0.0, 1.0], [0.0, 0.0]])
        t = DenseTensor("t", data)
        scheme = t.schemes[t.shape]
        self.assertEqual(1, scheme.num_tuples)


class TestPackageTensorImports(unittest.TestCase):
    """Test that tensor classes are importable from einjax."""

    def test_top_level_imports(self):
        """Test that tensor classes are accessible from einjax."""
        import einjax
        self.assertIsNotNone(einjax.BaseTensor)
        self.assertIsNotNone(einjax.IndexedTerm)
        self.assertIsNotNone(einjax.DenseTensor)


if __name__ == "__main__":
    unittest.main()
