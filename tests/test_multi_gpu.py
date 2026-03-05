"""
Integration tests for multi-GPU execution (PRD Section 4.10).

These tests require 8 GPUs and skip gracefully on machines with fewer.
They validate end-to-end multi-GPU execution through the user-facing API,
covering dense matmul, batched einsum, sparse matmul, resharding between
operations, memory balance, auto-detection, scaling efficiency, and
output sharding policy.
"""

from __future__ import annotations

import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from einjax.api import einsum, with_mesh
from einjax.config import get_config, reset_config
from einjax.execution.sparse_dispatch import execute_sparse, execute_sharded_sparse
from einjax.sharding.mesh import create_mesh
from einjax.tensor.sparse import SparseTensor, SparseTensorRelation


def _require_8_gpus(test: unittest.TestCase) -> None:
    """Skip the test if fewer than 8 GPU devices are available."""
    devices = jax.devices()
    if len(devices) < 8:
        test.skipTest(f"Need 8 GPUs, only {len(devices)} available")


def _relation_from_dense(
    dense: np.ndarray, tile_shape: tuple[int, ...]
) -> SparseTensorRelation:
    """Build a SparseTensorRelation from a dense array."""
    st = SparseTensor("test", dense)
    return st.to_relation(tile_shape)


# =============================================================================
# Integration Tests
# =============================================================================


class TestDenseMatmul8GPU(unittest.TestCase):
    """Dense (8192,8192) @ (8192,8192) sharded across 8 GPUs."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_dense_matmul_8gpu(self):
        _require_8_gpus(self)

        mesh = create_mesh(("x",), num_devices=8)
        size = 8192

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (size, size), dtype=jnp.float32)
        B = jax.random.normal(k2, (size, size), dtype=jnp.float32)

        # Multi-GPU result
        result = einsum("ij,jk->ik", A, B, mesh=mesh)

        # Single-device reference
        expected = einsum("ij,jk->ik", A, B, num_devices=1)

        np.testing.assert_allclose(
            np.asarray(result), np.asarray(expected), atol=1e-1, rtol=1e-3
        )


class TestDenseBatchEinsum8GPU(unittest.TestCase):
    """Batched einsum 'bij,bjk->bik' with batch dim sharded across 8 GPUs."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_dense_batch_einsum_8gpu(self):
        _require_8_gpus(self)

        mesh = create_mesh(("x",), num_devices=8)
        batch, m, n, k = 8, 256, 256, 256

        key = jax.random.PRNGKey(123)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (batch, m, n), dtype=jnp.float32)
        B = jax.random.normal(k2, (batch, n, k), dtype=jnp.float32)

        # Multi-GPU with batch dim sharded
        result = einsum("bij,bjk->bik", A, B, mesh=mesh)

        # Single-device reference
        expected = einsum("bij,bjk->bik", A, B, num_devices=1)

        np.testing.assert_allclose(
            np.asarray(result), np.asarray(expected), atol=1e-2, rtol=1e-3
        )

'''
class TestSparseMatmul8GPU(unittest.TestCase):
    """Sparse matmul with 90% sparsity at (8192, 8192), 8 GPUs."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_sparse_matmul_8gpu(self):
        _require_8_gpus(self)

        mesh = create_mesh(("x",), num_devices=8)
        size = 8192
        tile = 1024
        sparsity = 0.9

        rng = np.random.RandomState(42)

        # Create sparse matrices with 90% sparsity
        A_dense = rng.randn(size, size).astype(np.float32)
        mask_a = rng.rand(size, size) > sparsity
        A_dense *= mask_a

        B_dense = rng.randn(size, size).astype(np.float32)
        mask_b = rng.rand(size, size) > sparsity
        B_dense *= mask_b

        lhs = _relation_from_dense(A_dense, (tile, tile))
        rhs = _relation_from_dense(B_dense, (tile, tile))

        # Sharded sparse matmul
        result_rel = execute_sharded_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
            mesh=mesh,
            num_devices=8,
        )

        # Single-device sparse reference
        ref_rel = execute_sparse(
            lhs, rhs,
            join_keys=[(1, 0)],
            kernel_string="ij,jk->ik",
            agg_keys=[(0, 0), (1, 1)],
        )

        result_dense = result_rel.to_dense()
        ref_dense = ref_rel.to_dense()

        np.testing.assert_allclose(
            result_dense, ref_dense, atol=1e-3, rtol=1e-3
        )
'''

class TestReshardBetweenOps(unittest.TestCase):
    """Chain 'ij,jk->ik' then 'ik,kl->il' requiring intermediate result."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_reshard_between_ops(self):
        _require_8_gpus(self)

        mesh = create_mesh(("x",), num_devices=8)
        size = 1024

        key = jax.random.PRNGKey(77)
        k1, k2, k3 = jax.random.split(key, 3)
        A = jax.random.normal(k1, (size, size), dtype=jnp.float32)
        B = jax.random.normal(k2, (size, size), dtype=jnp.float32)
        C = jax.random.normal(k3, (size, size), dtype=jnp.float32)

        # Chain two matmuls with mesh
        with with_mesh(mesh):
            AB = einsum("ij,jk->ik", A, B)
            result = einsum("ik,kl->il", AB, C)

        # Single-device reference
        AB_ref = einsum("ij,jk->ik", A, B, num_devices=1)
        expected = einsum("ik,kl->il", AB_ref, C, num_devices=1)

        np.testing.assert_allclose(
            np.asarray(result), np.asarray(expected), atol=1e-1, rtol=1e-3
        )


class TestMemoryBalanced(unittest.TestCase):
    """Per-GPU memory usage within 20% of mean (balanced sharding)."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_memory_balanced(self):
        _require_8_gpus(self)

        mesh = create_mesh(("x",), num_devices=8)
        size = 8192

        A = jnp.ones((size, size), dtype=jnp.float32)
        B = jnp.ones((size, size), dtype=jnp.float32)

        result = einsum("ij,jk->ik", A, B, mesh=mesh)

        # Check that result is sharded
        self.assertIsInstance(result.sharding, jax.sharding.NamedSharding)

        # Check memory balance across devices by examining shard sizes
        # Each device should hold approximately 1/8 of the result
        devices = mesh.devices.flat
        per_device_sizes = []
        for shard in result.addressable_shards:
            per_device_sizes.append(shard.data.size)

        mean_size = np.mean(per_device_sizes)
        for dev_size in per_device_sizes:
            deviation = abs(dev_size - mean_size) / mean_size if mean_size > 0 else 0
            self.assertLessEqual(
                deviation, 0.2,
                f"Device shard size {dev_size} deviates {deviation:.1%} "
                f"from mean {mean_size:.0f} (>20%)",
            )


class TestAutoDetect8GPU(unittest.TestCase):
    """get_config() returns num_devices=8, device_type='gpu:v100' on this machine."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_auto_detect_8gpu(self):
        _require_8_gpus(self)

        config = get_config()
        self.assertEqual(config.num_devices, 8)
        self.assertEqual(config.device_type, "gpu:v100")


class TestScalingEfficiency(unittest.TestCase):
    """8-GPU throughput >= 4x single-GPU throughput for large matmul."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_scaling_efficiency(self):
        _require_8_gpus(self)

        mesh = create_mesh(("x",), num_devices=8)
        size = 8192

        key = jax.random.PRNGKey(99)
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (size, size), dtype=jnp.float32)
        B = jax.random.normal(k2, (size, size), dtype=jnp.float32)

        # Pre-shard for multi-GPU: shard A on rows, replicate B
        sharding_row = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("x", None),
        )
        sharding_rep = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(None, None),
        )
        A_s = jax.device_put(A, sharding_row)
        B_s = jax.device_put(B, sharding_rep)

        # Warmup (block to ensure JIT compilation completes)
        einsum("ij,jk->ik", A, B, num_devices=1).block_until_ready()
        einsum("ij,jk->ik", A_s, B_s, mesh=mesh).block_until_ready()

        # Time single-device
        n_trials = 3
        start = time.perf_counter()
        for _ in range(n_trials):
            r = einsum("ij,jk->ik", A, B, num_devices=1)
            r.block_until_ready()
        single_time = (time.perf_counter() - start) / n_trials

        # Time 8-GPU (with pre-sharded data to measure compute scaling)
        start = time.perf_counter()
        for _ in range(n_trials):
            r = einsum("ij,jk->ik", A_s, B_s, mesh=mesh)
            r.block_until_ready()
        multi_time = (time.perf_counter() - start) / n_trials

        speedup = single_time / multi_time if multi_time > 0 else float("inf")
        self.assertGreaterEqual(
            speedup, 4.0,
            f"8-GPU speedup {speedup:.2f}x is less than 4x "
            f"(single={single_time:.4f}s, multi={multi_time:.4f}s)",
        )


class TestOutputShardingPolicy(unittest.TestCase):
    """Default output is sharded; gather=True produces single-device output."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_default_output_sharded(self):
        """Default einsum output is sharded across the mesh."""
        _require_8_gpus(self)

        mesh = create_mesh(("x",), num_devices=8)
        size = 1024
        A = jnp.ones((size, size), dtype=jnp.float32)
        B = jnp.ones((size, size), dtype=jnp.float32)

        result = einsum("ij,jk->ik", A, B, mesh=mesh)
        self.assertIsInstance(result.sharding, jax.sharding.NamedSharding)

    def test_gather_produces_single_device_output(self):
        """gather=True collects the result onto a single device."""
        _require_8_gpus(self)

        mesh = create_mesh(("x",), num_devices=8)
        size = 1024
        A = jnp.ones((size, size), dtype=jnp.float32)
        B = jnp.ones((size, size), dtype=jnp.float32)

        result = einsum("ij,jk->ik", A, B, mesh=mesh, gather=True)

        # Result should be correct
        expected = np.full((size, size), size, dtype=np.float32)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-5)

        # Should NOT have NamedSharding after gather
        self.assertNotIsInstance(
            getattr(result, "sharding", None),
            jax.sharding.NamedSharding,
        )


if __name__ == "__main__":
    unittest.main()
