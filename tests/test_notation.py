"""
Tests for einjax notation parsing and core types.

Ported from einsql/test.py: TestImplicitOutput, TestValidation,
TestUtilityFunctions. Assertions are identical since these are
pure-Python functions with no backend dependencies.
"""

import unittest

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
    TilingScheme,
    CaseAssignment,
)


class TestUtilityFunctions(unittest.TestCase):
    """Tests for utility functions."""

    def test_index_to_subscript(self):
        """Test index to subscript conversion."""
        self.assertEqual("i", index_to_subscript(0))
        self.assertEqual("j", index_to_subscript(1))
        self.assertEqual("k", index_to_subscript(2))

    def test_find_all_factors(self):
        """Test factor finding."""
        self.assertEqual([1, 2, 3, 6], find_all_factors(6))
        self.assertEqual([1, 2, 4, 8], find_all_factors(8))
        self.assertEqual([1, 5], find_all_factors(5))

    def test_index_to_subscript_bounds(self):
        """Test that index_to_subscript raises for out-of-range indices."""
        with self.assertRaises(ValueError):
            index_to_subscript(20)  # Would exceed 'z'

    def test_find_all_factors_nonpositive(self):
        """Test that find_all_factors raises for non-positive input."""
        with self.assertRaises(ValueError):
            find_all_factors(0)
        with self.assertRaises(ValueError):
            find_all_factors(-3)

    def test_get_label_dimensions(self):
        """Test label dimension extraction from einsum string."""
        dims = get_label_dimensions("ij,jk->ik", [(3, 4), (4, 5)])
        self.assertEqual({"i": 3, "j": 4, "k": 5}, dims)

    def test_get_label_dimensions_mismatch(self):
        """Test error on inconsistent label dimensions."""
        with self.assertRaises(ValueError) as ctx:
            get_label_dimensions("ij,jk->ik", [(3, 4), (5, 6)])
        self.assertIn("j", str(ctx.exception))


class TestImplicitOutput(unittest.TestCase):
    """Tests for implicit output mode (notation without '->').

    When '->' is omitted, NumPy convention applies: output indices are
    all indices appearing exactly once across inputs, sorted alphabetically.
    """

    def test_implicit_matmul(self):
        """Test 'ij,jk' without '->' normalizes to 'ij,jk->ik'."""
        result = normalize_notation("ij,jk")
        self.assertEqual("ij,jk->ik", result)

    def test_implicit_trace(self):
        """Test 'ii' without '->' produces scalar (empty output)."""
        result = normalize_notation("ii")
        self.assertEqual("ii->", result)

    def test_implicit_identity(self):
        """Test 'ij' without '->' keeps both indices (both appear once)."""
        result = normalize_notation("ij")
        self.assertEqual("ij->ij", result)

    def test_implicit_three_tensor(self):
        """Test implicit output for three-tensor contraction."""
        result = normalize_notation("ij,jk,kl")
        self.assertEqual("ij,jk,kl->il", result)

    def test_explicit_passthrough(self):
        """Test that explicit '->' notation is not modified."""
        result = normalize_notation("ij,jk->ik")
        self.assertEqual("ij,jk->ik", result)

    def test_implicit_hadamard(self):
        """Test 'ij,ij' implicit output: all indices appear twice -> scalar."""
        result = normalize_notation("ij,ij")
        self.assertEqual("ij,ij->", result)


class TestValidation(unittest.TestCase):
    """Tests for input validation with descriptive error messages."""

    def test_wrong_tensor_count(self):
        """Test error when tensor count doesn't match notation."""
        with self.assertRaises(ValueError) as ctx:
            validate_inputs("ij,jk->ik", [(4, 8)])
        self.assertIn("2", str(ctx.exception))  # 2 input groups
        self.assertIn("1", str(ctx.exception))  # 1 tensor

    def test_dimension_mismatch(self):
        """Test clear error on dimension mismatch between tensors."""
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(
                "ij,jk->ik",
                [(4, 8), (6, 16)],
                names=["a", "b"],
            )
        msg = str(ctx.exception)
        self.assertIn("j", msg)
        self.assertIn("8", msg)
        self.assertIn("6", msg)

    def test_invalid_output_label(self):
        """Test error when output has label not present in inputs."""
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(
                "ij,jk->iz",
                [(4, 8), (8, 16)],
                names=["a", "b"],
            )
        msg = str(ctx.exception)
        self.assertIn("z", msg)
        self.assertIn("does not appear", msg)

    def test_ndim_mismatch(self):
        """Test error when label group length doesn't match tensor ndim."""
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(
                "ij,jk->ik",
                [(4, 8), (8,)],
                names=["a", "b"],
            )
        msg = str(ctx.exception)
        self.assertIn("jk", msg)
        self.assertIn("2", msg)
        self.assertIn("1", msg)

    def test_invalid_label_character(self):
        """Test error when label contains non-lowercase-alpha characters."""
        with self.assertRaises(ValueError) as ctx:
            validate_inputs(
                "iJ,Jk->ik",
                [(4, 8), (8, 16)],
                names=["a", "b"],
            )
        msg = str(ctx.exception)
        self.assertIn("J", msg)
        self.assertIn("lowercase", msg)

    def test_valid_input_passes(self):
        """Test that valid inputs pass validation without error."""
        validate_inputs(
            "ij,jk->ik",
            [(4, 8), (8, 16)],
            names=["a", "b"],
        )


class TestExprAST(unittest.TestCase):
    """Tests for expression AST classes."""

    def test_constant_flops(self):
        """Test that a constant has zero flops."""
        c = Constant(42.0)
        self.assertEqual(0.0, c.flops())
        self.assertEqual(42.0, c.value)

    def test_unary_op_flops(self):
        """Test unary operation flop counting."""
        c = Constant(1.0)
        u = UnaryOp("neg", c)
        self.assertEqual(1.0, u.flops())

    def test_binary_op_flops(self):
        """Test binary operation flop counting."""
        a = Constant(1.0)
        b = Constant(2.0)
        op = BinaryOp("+", a, b)
        self.assertEqual(1.0, op.flops())

    def test_nested_expr_flops(self):
        """Test nested expression flop counting."""
        a = Constant(1.0)
        b = Constant(2.0)
        add = BinaryOp("+", a, b)
        neg = UnaryOp("-", add)
        # neg: 1 + (add: 1 + 0 + 0) = 2
        self.assertEqual(2.0, neg.flops())


class TestTilingScheme(unittest.TestCase):
    """Tests for TilingScheme data class."""

    def test_computed_fields(self):
        """Test that post_init computes tile_size, value_count, num_tuples."""
        ts = TilingScheme(node=None, shape=(12, 8), tile_shape=(4, 4))
        self.assertEqual(16, ts.tile_size)
        self.assertEqual((3, 2), ts.value_count)
        self.assertEqual(6, ts.num_tuples)

    def test_identity_tiling(self):
        """Test identity tiling (tile_shape == shape) yields 1 tuple."""
        ts = TilingScheme(node=None, shape=(4, 8), tile_shape=(4, 8))
        self.assertEqual(32, ts.tile_size)
        self.assertEqual((1, 1), ts.value_count)
        self.assertEqual(1, ts.num_tuples)

    def test_case_assignment(self):
        """Test upper/lower case assignment from tiling scheme."""
        ts = TilingScheme(node=None, shape=(12, 8), tile_shape=(4, 8))
        cases = ts.get_case_assignments()
        # dim 0: tile_shape=4 < shape=12 -> UPPER
        self.assertEqual(CaseAssignment.UPPER, cases[0])
        # dim 1: tile_shape=8 == shape=8 -> LOWER
        self.assertEqual(CaseAssignment.LOWER, cases[1])

    def test_all_upper(self):
        """Test all dimensions upper-case (fully sharded)."""
        ts = TilingScheme(node=None, shape=(12, 8), tile_shape=(4, 4))
        cases = ts.get_case_assignments()
        self.assertEqual([CaseAssignment.UPPER, CaseAssignment.UPPER], cases)

    def test_all_lower(self):
        """Test all dimensions lower-case (identity tiling)."""
        ts = TilingScheme(node=None, shape=(4, 8), tile_shape=(4, 8))
        cases = ts.get_case_assignments()
        self.assertEqual([CaseAssignment.LOWER, CaseAssignment.LOWER], cases)


class TestPackageImports(unittest.TestCase):
    """Test that einjax package exports are accessible."""

    def test_top_level_imports(self):
        """Test that all public symbols are importable from einjax."""
        import einjax
        self.assertTrue(callable(einjax.index_to_subscript))
        self.assertTrue(callable(einjax.find_all_factors))
        self.assertTrue(callable(einjax.get_label_dimensions))
        self.assertTrue(callable(einjax.normalize_notation))
        self.assertTrue(callable(einjax.validate_inputs))
        self.assertIsNotNone(einjax.TilingScheme)
        self.assertIsNotNone(einjax.CaseAssignment)
        self.assertIsNotNone(einjax.Expr)


if __name__ == "__main__":
    unittest.main()
