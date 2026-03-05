"""
Tests for EinJAX Pallas kernels and kernel registry.

Tests block-sparse matmul, coordinate join strategies, and kernel
registry dispatch logic. All sparse results are verified against
numpy.einsum / numpy operations.
"""

import unittest

import numpy as np
import numpy.testing as npt

from einjax.tensor.sparse import SparseTensor, SparseTensorRelation
from einjax.kernels.registry import (
    KernelRegistry,
    KernelInfo,
    _detect_backend,
    _pattern_matches,
    _backend_matches,
    _sparsity_matches,
)
from einjax.execution.sparse_dispatch import execute_sparse
from einjax.kernels.pallas_matmul import (
    block_sparse_matmul,
    block_sparse_matmul_generic,
)
from einjax.kernels.pallas_gather import (
    coordinate_join_hash,
    coordinate_join_sorted,
)
from einjax.execution.sparse_dispatch import execute_sparse


# =============================================================================
# Helper
# =============================================================================

def _relation_from_dense(dense: np.ndarray, tile_shape: tuple[int, ...]) -> SparseTensorRelation:
    """Build a SparseTensorRelation from a dense array."""
    st = SparseTensor("test", dense)
    return st.to_relation(tile_shape)


# =============================================================================
# Block-Sparse Matmul Tests (pallas_matmul.py)
# =============================================================================


class TestBlockSparseMatmulGeneric(unittest.TestCase):
    """Tests for block_sparse_matmul_generic."""

    def test_identity_matmul(self):
        """A @ I = A."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.eye(2)

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = block_sparse_matmul_generic(lhs, rhs)
        dense_result = result.to_dense()
        npt.assert_allclose(dense_result, A @ B, atol=1e-6)

    def test_diagonal_matmul(self):
        """Diagonal x diagonal = diagonal."""
        A = np.diag([1.0, 2.0, 3.0, 4.0])
        B = np.diag([5.0, 6.0, 7.0, 8.0])

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = block_sparse_matmul_generic(lhs, rhs)
        dense_result = result.to_dense()
        npt.assert_allclose(dense_result, A @ B, atol=1e-6)

    def test_tiled_matmul(self):
        """Block-sparse matmul with 2x2 tiles."""
        A = np.array([
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 6.0],
            [0.0, 0.0, 7.0, 8.0],
        ])
        B = np.eye(4)

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))

        result = block_sparse_matmul_generic(lhs, rhs)
        dense_result = result.to_dense()
        npt.assert_allclose(dense_result, A @ B, atol=1e-6)

    def test_empty_result(self):
        """Non-overlapping sparse matrices produce empty result."""
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        B = np.zeros((4, 4))
        B[2, 2] = 1.0

        lhs = _relation_from_dense(A, (2, 2))
        rhs = _relation_from_dense(B, (2, 2))

        result = block_sparse_matmul_generic(lhs, rhs)
        dense_result = result.to_dense()
        npt.assert_allclose(dense_result, A @ B, atol=1e-6)

    def test_explicit_block_shape(self):
        """Explicit block_shape parameter works."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = block_sparse_matmul_generic(lhs, rhs, block_shape=(1, 1, 1))
        dense_result = result.to_dense()
        npt.assert_allclose(dense_result, A @ B, atol=1e-6)

    def test_non_square(self):
        """Non-square matmul: (3,4) @ (4,2)."""
        A = np.array([
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [5.0, 0.0, 6.0, 0.0],
        ])
        B = np.array([
            [1.0, 0.0],
            [0.0, 2.0],
            [3.0, 0.0],
            [0.0, 4.0],
        ])

        lhs = _relation_from_dense(A, (1, 1))
        rhs = _relation_from_dense(B, (1, 1))

        result = block_sparse_matmul_generic(lhs, rhs)
        dense_result = result.to_dense()
        npt.assert_allclose(dense_result, A @ B, atol=1e-6)

    def test_ndim_validation(self):
        """Raises ValueError for non-2D inputs."""
        A = np.array([1.0, 2.0, 3.0])
        lhs = _relation_from_dense(A.reshape(3, 1), (1, 1))
        # Create a fake 3D relation
        rhs = SparseTensorRelation(
            coords=np.array([[0, 0, 0]], dtype=np.int32),
            values=np.array([[[[1.0]]]]),
            shape=(1, 1, 1),
            tile_shape=(1, 1, 1),
        )
        with self.assertRaises(ValueError):
            block_sparse_matmul_generic(lhs, rhs)


class TestBlockSparseMatmulRawAPI(unittest.TestCase):
    """Tests for block_sparse_matmul (raw array API)."""

    def test_identity(self):
        """Raw API identity matmul."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.eye(2)

        lhs_rel = _relation_from_dense(A, (1, 1))
        rhs_rel = _relation_from_dense(B, (1, 1))

        result = block_sparse_matmul(
            lhs_coords=lhs_rel.coords,
            lhs_values=lhs_rel.values,
            rhs_coords=rhs_rel.coords,
            rhs_values=rhs_rel.values,
            block_shape=(1, 1, 1),
            lhs_shape=(2, 2),
            rhs_shape=(2, 2),
        )

        dense_result = result.to_dense()
        npt.assert_allclose(dense_result, A @ B, atol=1e-6)

    def test_tiled(self):
        """Raw API with 2x2 block tiles."""
        A = np.array([
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
        B = np.eye(4)

        lhs_rel = _relation_from_dense(A, (2, 2))
        rhs_rel = _relation_from_dense(B, (2, 2))

        result = block_sparse_matmul(
            lhs_coords=lhs_rel.coords,
            lhs_values=lhs_rel.values,
            rhs_coords=rhs_rel.coords,
            rhs_values=rhs_rel.values,
            block_shape=(2, 2, 2),
            lhs_shape=(4, 4),
            rhs_shape=(4, 4),
        )

        dense_result = result.to_dense()
        npt.assert_allclose(dense_result, A @ B, atol=1e-6)

    def test_output_shape(self):
        """Raw API output has correct shape."""
        A = np.diag([1.0, 2.0])
        B = np.diag([3.0, 4.0])

        lhs_rel = _relation_from_dense(A, (1, 1))
        rhs_rel = _relation_from_dense(B, (1, 1))

        result = block_sparse_matmul(
            lhs_coords=lhs_rel.coords,
            lhs_values=lhs_rel.values,
            rhs_coords=rhs_rel.coords,
            rhs_values=rhs_rel.values,
            block_shape=(1, 1, 1),
            lhs_shape=(2, 2),
            rhs_shape=(2, 2),
        )
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tile_shape, (1, 1))


# =============================================================================
# Coordinate Join Tests (pallas_gather.py)
# =============================================================================


class TestCoordinateJoinHash(unittest.TestCase):
    """Tests for hash-based coordinate join."""

    def test_matmul_join(self):
        """Join on single dimension for matmul."""
        lhs_coords = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32)
        rhs_coords = np.array([[0, 0], [1, 0]], dtype=np.int32)

        lhs_idx, rhs_idx = coordinate_join_hash(lhs_coords, rhs_coords, [(1, 0)])
        # LHS(0,0) joins RHS(0,0) on K=0
        # LHS(0,1) joins RHS(1,0) on K=1
        # LHS(1,0) joins RHS(0,0) on K=0
        self.assertEqual(len(lhs_idx), 3)
        self.assertEqual(len(rhs_idx), 3)

    def test_cross_product(self):
        """No join dims gives cross product."""
        lhs_coords = np.array([[0], [1]], dtype=np.int32)
        rhs_coords = np.array([[0], [1], [2]], dtype=np.int32)
        lhs_idx, rhs_idx = coordinate_join_hash(lhs_coords, rhs_coords, [])
        self.assertEqual(len(lhs_idx), 6)

    def test_no_matches(self):
        """Non-overlapping keys give empty result."""
        lhs_coords = np.array([[0, 0]], dtype=np.int32)
        rhs_coords = np.array([[1, 0]], dtype=np.int32)
        lhs_idx, rhs_idx = coordinate_join_hash(lhs_coords, rhs_coords, [(0, 0)])
        self.assertEqual(len(lhs_idx), 0)

    def test_empty_input(self):
        """Empty inputs produce empty output."""
        lhs_coords = np.zeros((0, 2), dtype=np.int32)
        rhs_coords = np.zeros((0, 2), dtype=np.int32)
        lhs_idx, rhs_idx = coordinate_join_hash(lhs_coords, rhs_coords, [(1, 0)])
        self.assertEqual(len(lhs_idx), 0)

    def test_same_dim_shorthand(self):
        """Integer join_dims as shorthand for same-dim join."""
        lhs_coords = np.array([[0, 0], [1, 1]], dtype=np.int32)
        rhs_coords = np.array([[0, 0], [1, 1]], dtype=np.int32)
        lhs_idx, rhs_idx = coordinate_join_hash(lhs_coords, rhs_coords, [0, 1])
        # (0,0) matches (0,0), (1,1) matches (1,1)
        self.assertEqual(len(lhs_idx), 2)


class TestCoordinateJoinSorted(unittest.TestCase):
    """Tests for sort-merge coordinate join."""

    def test_matmul_join(self):
        """Sort-merge join matches hash join for matmul."""
        lhs_coords = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32)
        rhs_coords = np.array([[0, 0], [1, 0]], dtype=np.int32)

        hash_l, hash_r = coordinate_join_hash(lhs_coords, rhs_coords, [(1, 0)])
        sort_l, sort_r = coordinate_join_sorted(lhs_coords, rhs_coords, [(1, 0)])

        # Same number of matches
        self.assertEqual(len(hash_l), len(sort_l))

        # Same match pairs (possibly different order)
        hash_pairs = set(zip(hash_l.tolist(), hash_r.tolist()))
        sort_pairs = set(zip(sort_l.tolist(), sort_r.tolist()))
        self.assertEqual(hash_pairs, sort_pairs)

    def test_cross_product(self):
        """Sort-merge cross product."""
        lhs_coords = np.array([[0], [1]], dtype=np.int32)
        rhs_coords = np.array([[0], [1], [2]], dtype=np.int32)
        lhs_idx, rhs_idx = coordinate_join_sorted(lhs_coords, rhs_coords, [])
        self.assertEqual(len(lhs_idx), 6)

    def test_no_matches(self):
        """Sort-merge with non-overlapping keys."""
        lhs_coords = np.array([[0, 0]], dtype=np.int32)
        rhs_coords = np.array([[1, 0]], dtype=np.int32)
        lhs_idx, rhs_idx = coordinate_join_sorted(lhs_coords, rhs_coords, [(0, 0)])
        self.assertEqual(len(lhs_idx), 0)

    def test_empty_input(self):
        """Sort-merge with empty inputs."""
        lhs_coords = np.zeros((0, 2), dtype=np.int32)
        rhs_coords = np.zeros((0, 2), dtype=np.int32)
        lhs_idx, rhs_idx = coordinate_join_sorted(lhs_coords, rhs_coords, [(1, 0)])
        self.assertEqual(len(lhs_idx), 0)

    def test_multiple_matches(self):
        """Multiple matches per key group."""
        # Two tiles at K=0 on each side
        lhs_coords = np.array([[0, 0], [1, 0]], dtype=np.int32)
        rhs_coords = np.array([[0, 0], [0, 1]], dtype=np.int32)
        lhs_idx, rhs_idx = coordinate_join_sorted(lhs_coords, rhs_coords, [(1, 0)])
        # 2 LHS tiles x 2 RHS tiles with K=0 = 4 matches
        self.assertEqual(len(lhs_idx), 4)

    def test_consistency_with_hash(self):
        """Sort-merge produces same results as hash join on larger input."""
        rng = np.random.RandomState(42)
        n = 50
        lhs_coords = rng.randint(0, 5, size=(n, 2)).astype(np.int32)
        rhs_coords = rng.randint(0, 5, size=(n, 2)).astype(np.int32)

        hash_l, hash_r = coordinate_join_hash(lhs_coords, rhs_coords, [(1, 0)])
        sort_l, sort_r = coordinate_join_sorted(lhs_coords, rhs_coords, [(1, 0)])

        hash_pairs = set(zip(hash_l.tolist(), hash_r.tolist()))
        sort_pairs = set(zip(sort_l.tolist(), sort_r.tolist()))
        self.assertEqual(hash_pairs, sort_pairs)


# =============================================================================
# Kernel Registry Tests (registry.py)
# =============================================================================


class TestKernelRegistryLookup(unittest.TestCase):
    """Tests for KernelRegistry.lookup()."""

    def test_sparse_matmul_lookup(self):
        """Finds the sparse matmul kernel for ij,jk->ik."""
        registry = KernelRegistry()
        fn = registry.lookup("ij,jk->ik", backend="cpu", sparsity="sparse")
        self.assertIs(fn, block_sparse_matmul_generic)

    def test_dense_lookup(self):
        """Finds dense einsum fallback."""
        from einjax.execution.dense_kernels import execute_dense_einsum
        registry = KernelRegistry()
        fn = registry.lookup("ij,jk->ik", backend="cpu", sparsity="dense")
        self.assertIs(fn, execute_dense_einsum)

    def test_wildcard_fallback(self):
        """Non-matmul pattern falls back to generic sparse einsum."""
        registry = KernelRegistry()
        fn = registry.lookup("ij,kj->ik", backend="cpu", sparsity="sparse")
        self.assertIs(fn, execute_sparse)

    def test_no_match_raises(self):
        """Raises ValueError when no kernel matches."""
        registry = KernelRegistry()
        with self.assertRaises(ValueError):
            registry.lookup("ij,jk->ik", backend="cpu", sparsity="unknown_mode")

    def test_generic_matches_any_backend(self):
        """Generic backend matches gpu, tpu, cpu."""
        registry = KernelRegistry()
        for backend in ["gpu", "tpu", "cpu"]:
            fn = registry.lookup("ij,jk->ik", backend=backend, sparsity="sparse")
            self.assertIsNotNone(fn)


class TestKernelRegistryPriority(unittest.TestCase):
    """Tests for kernel priority ordering."""

    def test_higher_priority_wins(self):
        """Higher priority kernel is selected over lower."""
        registry = KernelRegistry()

        def custom_matmul(*args, **kwargs):
            pass

        registry.register(KernelInfo(
            name="custom_sparse_matmul",
            pattern="ij,jk->ik",
            backend="generic",
            sparsity="sparse",
            priority=10,
            kernel_fn=custom_matmul,
        ))

        fn = registry.lookup("ij,jk->ik", backend="cpu", sparsity="sparse")
        self.assertIs(fn, custom_matmul)

    def test_pattern_specific_over_wildcard(self):
        """Pattern-specific kernel (priority 0) beats wildcard (priority -1)."""
        registry = KernelRegistry()
        fn = registry.lookup("ij,jk->ik", backend="cpu", sparsity="sparse")
        # block_sparse_matmul_generic (priority=0) should beat
        # execute_sparse (priority=-1)
        self.assertIs(fn, block_sparse_matmul_generic)


class TestKernelRegistryRegister(unittest.TestCase):
    """Tests for custom kernel registration."""

    def test_register_custom(self):
        """Can register and retrieve a custom kernel."""
        registry = KernelRegistry()

        def my_kernel(*args, **kwargs):
            return "custom_result"

        registry.register(KernelInfo(
            name="my_custom",
            pattern="ijk,jkl->il",
            backend="generic",
            sparsity="sparse",
            priority=5,
            kernel_fn=my_kernel,
        ))

        fn = registry.lookup("ijk,jkl->il", backend="cpu", sparsity="sparse")
        self.assertIs(fn, my_kernel)

    def test_list_kernels(self):
        """list_kernels returns registered kernels."""
        registry = KernelRegistry()
        all_kernels = registry.list_kernels()
        self.assertGreaterEqual(len(all_kernels), 3)  # 3 builtins

    def test_list_kernels_filtered(self):
        """list_kernels with filters."""
        registry = KernelRegistry()
        sparse_kernels = registry.list_kernels(sparsity="sparse")
        self.assertGreaterEqual(len(sparse_kernels), 2)

        dense_kernels = registry.list_kernels(sparsity="dense")
        self.assertGreaterEqual(len(dense_kernels), 1)


class TestKernelRegistryInfo(unittest.TestCase):
    """Tests for lookup_info()."""

    def test_lookup_info_returns_kernelinfo(self):
        """lookup_info returns KernelInfo, not just callable."""
        registry = KernelRegistry()
        info = registry.lookup_info("ij,jk->ik", backend="cpu", sparsity="sparse")
        self.assertIsInstance(info, KernelInfo)
        self.assertEqual(info.pattern, "ij,jk->ik")
        self.assertEqual(info.name, "generic_sparse_matmul")

    def test_lookup_info_no_match(self):
        """lookup_info raises ValueError on no match."""
        registry = KernelRegistry()
        with self.assertRaises(ValueError):
            registry.lookup_info("ij,jk->ik", backend="cpu", sparsity="nonexistent")


# =============================================================================
# Pattern/Backend/Sparsity Matching Tests
# =============================================================================


class TestPatternMatching(unittest.TestCase):
    """Tests for _pattern_matches helper."""

    def test_exact_match(self):
        self.assertTrue(_pattern_matches("ij,jk->ik", "ij,jk->ik"))

    def test_wildcard(self):
        self.assertTrue(_pattern_matches("*", "anything"))

    def test_no_match(self):
        self.assertFalse(_pattern_matches("ij,jk->ik", "ij,kj->ik"))


class TestBackendMatching(unittest.TestCase):
    """Tests for _backend_matches helper."""

    def test_generic_matches_all(self):
        self.assertTrue(_backend_matches("generic", "gpu"))
        self.assertTrue(_backend_matches("generic", "tpu"))
        self.assertTrue(_backend_matches("generic", "cpu"))

    def test_exact_match(self):
        self.assertTrue(_backend_matches("gpu", "gpu"))

    def test_prefix_match(self):
        self.assertTrue(_backend_matches("gpu", "gpu:triton"))

    def test_no_match(self):
        self.assertFalse(_backend_matches("gpu", "tpu"))


class TestSparsityMatching(unittest.TestCase):
    """Tests for _sparsity_matches helper."""

    def test_any_matches_all(self):
        self.assertTrue(_sparsity_matches("any", "dense"))
        self.assertTrue(_sparsity_matches("any", "sparse"))

    def test_exact_match(self):
        self.assertTrue(_sparsity_matches("sparse", "sparse"))

    def test_no_match(self):
        self.assertFalse(_sparsity_matches("sparse", "dense"))


class TestDetectBackend(unittest.TestCase):
    """Tests for _detect_backend."""

    def test_returns_string(self):
        """Backend is one of gpu, tpu, cpu."""
        backend = _detect_backend()
        self.assertIn(backend, ("gpu", "tpu", "cpu"))


# =============================================================================
# Package Import Tests
# =============================================================================


class TestPackageKernelImports(unittest.TestCase):
    """Tests for package-level imports of kernel components."""

    def test_kernels_package_imports(self):
        """Core kernel exports importable from einjax.kernels."""
        from einjax.kernels import (
            KernelRegistry,
            KernelInfo,
            block_sparse_matmul,
            block_sparse_matmul_generic,
            coordinate_join_sorted,
            coordinate_join_hash,
        )
        self.assertIsNotNone(KernelRegistry)
        self.assertIsNotNone(KernelInfo)
        self.assertIsNotNone(block_sparse_matmul)
        self.assertIsNotNone(block_sparse_matmul_generic)
        self.assertIsNotNone(coordinate_join_sorted)
        self.assertIsNotNone(coordinate_join_hash)

    def test_top_level_imports(self):
        """Core kernel exports importable from einjax."""
        from einjax import (
            KernelRegistry,
            KernelInfo,
            block_sparse_matmul,
            block_sparse_matmul_generic,
            coordinate_join_sorted,
            coordinate_join_hash,
        )
        self.assertIsNotNone(KernelRegistry)
        self.assertIsNotNone(KernelInfo)
        self.assertIsNotNone(block_sparse_matmul)
        self.assertIsNotNone(block_sparse_matmul_generic)
        self.assertIsNotNone(coordinate_join_sorted)
        self.assertIsNotNone(coordinate_join_hash)


if __name__ == "__main__":
    unittest.main()
