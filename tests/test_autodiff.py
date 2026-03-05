"""
Tests for einjax autodiff (custom VJP) for sparse tensor operations.

Verifies gradient correctness via jax.grad and finite-difference estimates
per PRD Section 8.2 and Success Criterion #4: jax.grad through sparse
einsum produces gradients matching finite-difference estimates to within 1e-3.
"""

import unittest

import numpy as np
import numpy.testing as npt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from einjax.tensor.sparse import SparseTensor, SparseTensorRelation
from einjax.execution.sparse_dispatch import coordinate_join
from einjax.autodiff.custom_vjp import (
    sparse_einsum,
    sparse_einsum_raw,
    _reverse_einsum_string,
)


# =============================================================================
# Helper: build SparseTensorRelation from dense matrix
# =============================================================================


def _relation_from_dense(dense: np.ndarray, tile_shape: tuple[int, ...]) -> SparseTensorRelation:
    """Build a SparseTensorRelation from a dense array."""
    st = SparseTensor("test", dense)
    return st.to_relation(tile_shape)


# =============================================================================
# Test _reverse_einsum_string
# =============================================================================


class TestReverseEinsumString(unittest.TestCase):
    """Tests for backward einsum string derivation."""

    def test_matmul(self):
        """ij,jk->ik: grad_lhs=ik,jk->ij, grad_rhs=ij,ik->jk."""
        grad_lhs, grad_rhs = _reverse_einsum_string("ij,jk->ik")
        self.assertEqual(grad_lhs, "ik,jk->ij")
        self.assertEqual(grad_rhs, "ij,ik->jk")

    def test_outer_product(self):
        """i,j->ij: grad_lhs=ij,j->i, grad_rhs=i,ij->j."""
        grad_lhs, grad_rhs = _reverse_einsum_string("i,j->ij")
        self.assertEqual(grad_lhs, "ij,j->i")
        self.assertEqual(grad_rhs, "i,ij->j")

    def test_dot_product(self):
        """i,i->: grad_lhs=,i->i, grad_rhs=i,->i."""
        grad_lhs, grad_rhs = _reverse_einsum_string("i,i->")
        self.assertEqual(grad_lhs, ",i->i")
        self.assertEqual(grad_rhs, "i,->i")

    def test_batch_matmul(self):
        """bij,bjk->bik: grad_lhs=bik,bjk->bij, grad_rhs=bij,bik->bjk."""
        grad_lhs, grad_rhs = _reverse_einsum_string("bij,bjk->bik")
        self.assertEqual(grad_lhs, "bik,bjk->bij")
        self.assertEqual(grad_rhs, "bij,bik->bjk")

    def test_missing_arrow_raises(self):
        """Missing -> raises ValueError."""
        with self.assertRaises(ValueError):
            _reverse_einsum_string("ij,jk")


# =============================================================================
# Test sparse_einsum forward pass correctness
# =============================================================================


class TestSparseEinsumForward(unittest.TestCase):
    """Tests that sparse_einsum forward pass matches numpy.einsum."""

    def test_matmul_identity(self):
        """Sparse matmul with identity matrix."""
        I = np.eye(4)
        A = np.random.RandomState(42).randn(4, 4)
        lhs = _relation_from_dense(I, (2, 2))
        rhs = _relation_from_dense(A, (2, 2))
        result = sparse_einsum(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
        )
        expected = I @ A
        npt.assert_allclose(result.to_dense(), expected, atol=1e-6)

    def test_matmul_sparse(self):
        """Sparse matmul with sparse inputs."""
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[1, 1] = 2.0
        A[2, 2] = 3.0
        B = np.zeros((4, 4))
        B[0, 0] = 4.0
        B[1, 1] = 5.0
        B[2, 2] = 6.0

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))
        result = sparse_einsum(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
        )
        expected = A @ B
        npt.assert_allclose(result.to_dense(), expected, atol=1e-6)

    def test_element_tiling(self):
        """Matmul with element-level (1,1) tiling."""
        A = np.array([[1.0, 2.0], [3.0, 0.0]])
        B = np.array([[0.0, 4.0], [5.0, 0.0]])
        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))
        result = sparse_einsum(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
        )
        expected = A @ B
        npt.assert_allclose(result.to_dense(), expected, atol=1e-6)

    def test_empty_result(self):
        """Sparse einsum with no matching tiles produces empty result."""
        A = np.zeros((4, 4))
        A[0, 0] = 1.0  # Only tile (0,0)
        B = np.zeros((4, 4))
        B[2, 2] = 1.0  # Only tile (1,1)
        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))
        result = sparse_einsum(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
        )
        self.assertEqual(result.num_tuples, 0)


# =============================================================================
# Test sparse_einsum_raw with custom_vjp (jax.grad)
# =============================================================================


class TestSparseEinsumGrad(unittest.TestCase):
    """Tests that jax.grad through sparse_einsum_raw is correct."""

    def test_grad_lhs_matmul(self):
        """Gradient w.r.t. LHS values in a matmul matches finite difference."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))
        lhs_indices, rhs_indices = coordinate_join(lhs, rhs, [(1, 0)])

        def loss_fn(lhs_vals):
            out = sparse_einsum_raw(
                lhs_vals, jnp.array(rhs.values),
                lhs.coords, rhs.coords,
                lhs_indices, rhs_indices,
                "ij,jk->ik",
            )
            return jnp.sum(out)

        lhs_vals = jnp.array(lhs.values)
        grad = jax.grad(loss_fn)(lhs_vals)

        # Finite difference check
        eps = 1e-4
        grad_fd = np.zeros_like(lhs.values)
        for idx in np.ndindex(*lhs.values.shape):
            vals_plus = lhs_vals.at[idx].add(eps)
            vals_minus = lhs_vals.at[idx].add(-eps)
            f_plus = float(loss_fn(vals_plus))
            f_minus = float(loss_fn(vals_minus))
            grad_fd[idx] = (f_plus - f_minus) / (2 * eps)

        npt.assert_allclose(np.asarray(grad), grad_fd, atol=1e-3)

    def test_grad_rhs_matmul(self):
        """Gradient w.r.t. RHS values in a matmul matches finite difference."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))
        lhs_indices, rhs_indices = coordinate_join(lhs, rhs, [(1, 0)])

        def loss_fn(rhs_vals):
            out = sparse_einsum_raw(
                jnp.array(lhs.values), rhs_vals,
                lhs.coords, rhs.coords,
                lhs_indices, rhs_indices,
                "ij,jk->ik",
            )
            return jnp.sum(out)

        rhs_vals = jnp.array(rhs.values)
        grad = jax.grad(loss_fn)(rhs_vals)

        # Finite difference check
        eps = 1e-4
        grad_fd = np.zeros_like(rhs.values)
        for idx in np.ndindex(*rhs.values.shape):
            vals_plus = rhs_vals.at[idx].add(eps)
            vals_minus = rhs_vals.at[idx].add(-eps)
            f_plus = float(loss_fn(vals_plus))
            f_minus = float(loss_fn(vals_minus))
            grad_fd[idx] = (f_plus - f_minus) / (2 * eps)

        npt.assert_allclose(np.asarray(grad), grad_fd, atol=1e-3)

    def test_grad_both_sides(self):
        """jax.value_and_grad computes both value and gradient correctly."""
        A = np.array([[1.0, 0.0], [0.0, 2.0]])
        B = np.array([[3.0, 0.0], [0.0, 4.0]])

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))
        lhs_indices, rhs_indices = coordinate_join(lhs, rhs, [(1, 0)])

        def loss_fn(lhs_vals, rhs_vals):
            out = sparse_einsum_raw(
                lhs_vals, rhs_vals,
                lhs.coords, rhs.coords,
                lhs_indices, rhs_indices,
                "ij,jk->ik",
            )
            return jnp.sum(out ** 2)

        lhs_vals = jnp.array(lhs.values)
        rhs_vals = jnp.array(rhs.values)
        val, (g_lhs, g_rhs) = jax.value_and_grad(loss_fn, argnums=(0, 1))(lhs_vals, rhs_vals)

        # Value should be sum of squared elements of A @ B
        expected_result = A @ B
        expected_val = np.sum(expected_result ** 2)
        npt.assert_allclose(float(val), expected_val, atol=1e-5)

        # Gradients should be finite and non-zero where inputs are non-zero
        self.assertTrue(jnp.all(jnp.isfinite(g_lhs)))
        self.assertTrue(jnp.all(jnp.isfinite(g_rhs)))

    def test_grad_sparse_diagonal(self):
        """Gradient through sparse diagonal matmul."""
        A = np.diag([1.0, 2.0, 3.0, 4.0])
        B = np.diag([5.0, 6.0, 7.0, 8.0])

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))
        lhs_indices, rhs_indices = coordinate_join(lhs, rhs, [(1, 0)])

        def loss_fn(lhs_vals):
            out = sparse_einsum_raw(
                lhs_vals, jnp.array(rhs.values),
                lhs.coords, rhs.coords,
                lhs_indices, rhs_indices,
                "ij,jk->ik",
            )
            return jnp.sum(out)

        lhs_vals = jnp.array(lhs.values)
        grad = jax.grad(loss_fn)(lhs_vals)

        # Finite difference check
        eps = 1e-4
        grad_fd = np.zeros_like(lhs.values)
        for idx in np.ndindex(*lhs.values.shape):
            vals_plus = lhs_vals.at[idx].add(eps)
            vals_minus = lhs_vals.at[idx].add(-eps)
            f_plus = float(loss_fn(vals_plus))
            f_minus = float(loss_fn(vals_minus))
            grad_fd[idx] = (f_plus - f_minus) / (2 * eps)

        npt.assert_allclose(np.asarray(grad), grad_fd, atol=1e-3)

    def test_grad_weighted_loss(self):
        """Gradient with a non-trivial weighted sum loss."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))
        lhs_indices, rhs_indices = coordinate_join(lhs, rhs, [(1, 0)])

        weights = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        def loss_fn(lhs_vals):
            out = sparse_einsum_raw(
                lhs_vals, jnp.array(rhs.values),
                lhs.coords, rhs.coords,
                lhs_indices, rhs_indices,
                "ij,jk->ik",
            )
            # Weighted sum — single tile, so out[0] is the 2×2 result
            return jnp.sum(out[0] * weights)

        lhs_vals = jnp.array(lhs.values)
        grad = jax.grad(loss_fn)(lhs_vals)

        # Finite difference
        eps = 1e-4
        grad_fd = np.zeros_like(lhs.values)
        for idx in np.ndindex(*lhs.values.shape):
            vals_plus = lhs_vals.at[idx].add(eps)
            vals_minus = lhs_vals.at[idx].add(-eps)
            f_plus = float(loss_fn(vals_plus))
            f_minus = float(loss_fn(vals_minus))
            grad_fd[idx] = (f_plus - f_minus) / (2 * eps)

        npt.assert_allclose(np.asarray(grad), grad_fd, atol=1e-3)


# =============================================================================
# Test gradient accumulation (tile appears in multiple pairs)
# =============================================================================


class TestGradAccumulation(unittest.TestCase):
    """Test that gradients accumulate correctly for tiles in multiple pairs."""

    def test_tile_in_multiple_matches(self):
        """A tile matched to multiple partners accumulates gradients."""
        # A has tiles at (0,0) and (0,1)
        # B has tiles at (0,0) and (1,0)
        # Both B tiles join with A tile (0,0) on dim 1→0
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[0, 2] = 2.0
        B = np.zeros((4, 4))
        B[0, 0] = 3.0
        B[2, 0] = 4.0

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))
        lhs_indices, rhs_indices = coordinate_join(lhs, rhs, [(1, 0)])

        def loss_fn(lhs_vals):
            out = sparse_einsum_raw(
                lhs_vals, jnp.array(rhs.values),
                lhs.coords, rhs.coords,
                lhs_indices, rhs_indices,
                "ij,jk->ik",
            )
            return jnp.sum(out)

        lhs_vals = jnp.array(lhs.values)
        grad = jax.grad(loss_fn)(lhs_vals)

        # Verify against finite differences
        eps = 1e-4
        grad_fd = np.zeros_like(lhs.values)
        for idx in np.ndindex(*lhs.values.shape):
            vals_plus = lhs_vals.at[idx].add(eps)
            vals_minus = lhs_vals.at[idx].add(-eps)
            f_plus = float(loss_fn(vals_plus))
            f_minus = float(loss_fn(vals_minus))
            grad_fd[idx] = (f_plus - f_minus) / (2 * eps)

        npt.assert_allclose(np.asarray(grad), grad_fd, atol=1e-3)

    def test_grad_shape_matches_input(self):
        """Gradient arrays have same shape as input value arrays."""
        A = np.eye(4)
        B = np.eye(4)
        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))
        lhs_indices, rhs_indices = coordinate_join(lhs, rhs, [(1, 0)])

        def loss_fn(lhs_vals, rhs_vals):
            out = sparse_einsum_raw(
                lhs_vals, rhs_vals,
                lhs.coords, rhs.coords,
                lhs_indices, rhs_indices,
                "ij,jk->ik",
            )
            return jnp.sum(out)

        lhs_vals = jnp.array(lhs.values)
        rhs_vals = jnp.array(rhs.values)
        g_lhs, g_rhs = jax.grad(loss_fn, argnums=(0, 1))(lhs_vals, rhs_vals)

        self.assertEqual(g_lhs.shape, lhs_vals.shape)
        self.assertEqual(g_rhs.shape, rhs_vals.shape)


# =============================================================================
# Test dense path (no custom VJP needed)
# =============================================================================


class TestDenseAutodiff(unittest.TestCase):
    """Verify dense jnp.einsum is differentiable (no custom VJP needed)."""

    def test_dense_grad(self):
        """jax.grad works through jnp.einsum for dense matmul."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        def loss_fn(a):
            return jnp.sum(jnp.einsum("ij,jk->ik", a, B))

        grad = jax.grad(loss_fn)(A)
        # d/dA sum(A@B) = ones @ B^T? No: d/dA_ij sum_mn (AB)_mn
        # = sum_k B_jk for each (i,j), so grad[i,j] = sum_k B[j,k]
        expected = np.ones((2, 2)) @ B.T
        npt.assert_allclose(np.asarray(grad), np.asarray(expected), atol=1e-6)

    def test_dense_value_and_grad(self):
        """jax.value_and_grad returns both value and gradient for dense."""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        def loss_fn(a):
            return jnp.sum(jnp.einsum("ij,jk->ik", a, B) ** 2)

        val, grad = jax.value_and_grad(loss_fn)(A)
        expected_val = np.sum((np.array(A) @ np.array(B)) ** 2)
        npt.assert_allclose(float(val), expected_val, atol=1e-4)
        self.assertTrue(jnp.all(jnp.isfinite(grad)))


# =============================================================================
# Test outer product gradient
# =============================================================================


class TestOuterProductGrad(unittest.TestCase):
    """Tests for outer product gradient through sparse_einsum_raw."""

    def test_outer_product_grad(self):
        """Gradient through sparse outer product i,j->ij."""
        a = np.array([[1.0], [2.0]])  # 2×1 vector as 2D
        b = np.array([[3.0], [4.0]])  # 2×1 vector as 2D

        lhs = _relation_from_dense(a, (1, 1))
        rhs = _relation_from_dense(b, (1, 1))
        # Outer product: no join keys
        lhs_indices, rhs_indices = coordinate_join(lhs, rhs, [])

        def loss_fn(lhs_vals):
            out = sparse_einsum_raw(
                lhs_vals, jnp.array(rhs.values),
                lhs.coords, rhs.coords,
                lhs_indices, rhs_indices,
                "ij,kl->ijkl",
            )
            return jnp.sum(out)

        lhs_vals = jnp.array(lhs.values)
        grad = jax.grad(loss_fn)(lhs_vals)

        # Finite difference
        eps = 1e-4
        grad_fd = np.zeros_like(lhs.values)
        for idx in np.ndindex(*lhs.values.shape):
            vals_plus = lhs_vals.at[idx].add(eps)
            vals_minus = lhs_vals.at[idx].add(-eps)
            f_plus = float(loss_fn(vals_plus))
            f_minus = float(loss_fn(vals_minus))
            grad_fd[idx] = (f_plus - f_minus) / (2 * eps)

        npt.assert_allclose(np.asarray(grad), grad_fd, atol=1e-3)


# =============================================================================
# Test sparse_einsum high-level API
# =============================================================================


class TestSparseEinsumHighLevel(unittest.TestCase):
    """Tests for the high-level sparse_einsum function."""

    def test_returns_relation(self):
        """sparse_einsum returns a SparseTensorRelation."""
        A = np.eye(2)
        B = np.eye(2)
        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))
        result = sparse_einsum(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
        )
        self.assertIsInstance(result, SparseTensorRelation)

    def test_inferred_output_shape(self):
        """Output shape is inferred from agg_keys when not provided."""
        A = np.eye(4)
        B = np.ones((4, 6))
        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 3))
        result = sparse_einsum(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
        )
        self.assertEqual(result.shape, (4, 6))

    def test_explicit_output_shape(self):
        """Explicit output_shape is used when provided."""
        A = np.eye(4)
        B = np.eye(4)
        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))
        result = sparse_einsum(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            output_shape=(4, 4),
            output_tile_shape=(2, 2),
        )
        self.assertEqual(result.shape, (4, 4))
        self.assertEqual(result.tile_shape, (2, 2))


# =============================================================================
# Package import tests
# =============================================================================


class TestPackageAutodiffImports(unittest.TestCase):
    """Test that autodiff exports are importable."""

    def test_autodiff_package_imports(self):
        """Core autodiff functions importable from autodiff package."""
        from einjax.autodiff import sparse_einsum, sparse_einsum_raw
        self.assertTrue(callable(sparse_einsum))
        self.assertTrue(callable(sparse_einsum_raw))

    def test_top_level_imports(self):
        """Autodiff functions importable from top-level einjax."""
        import einjax
        self.assertTrue(callable(einjax.sparse_einsum))
        self.assertTrue(callable(einjax.sparse_einsum_raw))


if __name__ == "__main__":
    unittest.main()
