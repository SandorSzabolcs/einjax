"""
Tests for sharding modules: partition.py, mesh.py, and reshard.py.

Tests PartitionSpec derivation from tile shapes, device mesh creation,
and resharding between tiling schemes.
"""

from __future__ import annotations

import unittest

from einjax.core.types import TilingScheme
from einjax.sharding.partition import tile_shape_to_partition_spec, derive_partition_specs
from einjax.sharding.mesh import infer_mesh_shape
from einjax.sharding.reshard import (
    needs_reshard,
    estimate_reshard_bytes,
    estimate_reshard_cost,
    compute_target_partition_spec,
    reshard_dense,
    plan_reshard_sequence,
)


class TestTileShapeToPartitionSpec(unittest.TestCase):
    """Test tile_shape → PartitionSpec mapping (PRD Section 3.3)."""

    def test_fully_sharded_2d(self):
        """Both dims sharded: tile < shape on both axes."""
        spec = tile_shape_to_partition_spec(
            shape=(8, 8), tile_shape=(4, 4)
        )
        self.assertEqual(spec, ("x", "y"))

    def test_fully_local_2d(self):
        """Both dims local: tile == shape on both axes → no sharding."""
        spec = tile_shape_to_partition_spec(
            shape=(8, 8), tile_shape=(8, 8)
        )
        self.assertEqual(spec, (None, None))

    def test_mixed_sharding_2d(self):
        """First dim sharded, second local."""
        spec = tile_shape_to_partition_spec(
            shape=(8, 4), tile_shape=(4, 4)
        )
        self.assertEqual(spec, ("x", None))

    def test_mixed_sharding_reversed(self):
        """First dim local, second sharded."""
        spec = tile_shape_to_partition_spec(
            shape=(4, 8), tile_shape=(4, 4)
        )
        self.assertEqual(spec, (None, "x"))

    def test_3d_all_sharded(self):
        """3D tensor, all dimensions sharded."""
        spec = tile_shape_to_partition_spec(
            shape=(8, 6, 4), tile_shape=(4, 3, 2)
        )
        self.assertEqual(spec, ("x", "y", "z"))

    def test_3d_partial_sharding(self):
        """3D tensor, only first and third sharded."""
        spec = tile_shape_to_partition_spec(
            shape=(8, 6, 4), tile_shape=(4, 6, 2)
        )
        self.assertEqual(spec, ("x", None, "y"))

    def test_1d_sharded(self):
        """1D sharded tensor."""
        spec = tile_shape_to_partition_spec(
            shape=(16,), tile_shape=(4,)
        )
        self.assertEqual(spec, ("x",))

    def test_1d_local(self):
        """1D fully local tensor."""
        spec = tile_shape_to_partition_spec(
            shape=(16,), tile_shape=(16,)
        )
        self.assertEqual(spec, (None,))

    def test_custom_axis_names(self):
        """Custom mesh axis names."""
        spec = tile_shape_to_partition_spec(
            shape=(8, 8), tile_shape=(4, 4),
            mesh_axis_names=("batch", "model"),
        )
        self.assertEqual(spec, ("batch", "model"))

    def test_4d_batch_matmul(self):
        """4D tensor: batch and head sharded, spatial dims local."""
        spec = tile_shape_to_partition_spec(
            shape=(32, 16, 64, 64),
            tile_shape=(4, 2, 64, 64),
        )
        self.assertEqual(spec, ("x", "y", None, None))


class TestPartitionSpecValidation(unittest.TestCase):
    """Test validation in tile_shape_to_partition_spec."""

    def test_mismatched_ndim(self):
        """shape and tile_shape must have same number of dimensions."""
        with self.assertRaises(ValueError) as ctx:
            tile_shape_to_partition_spec(
                shape=(8, 8), tile_shape=(4, 4, 4)
            )
        self.assertIn("same number of dimensions", str(ctx.exception))

    def test_tile_exceeds_shape(self):
        """tile_shape[d] cannot exceed shape[d]."""
        with self.assertRaises(ValueError) as ctx:
            tile_shape_to_partition_spec(
                shape=(8, 4), tile_shape=(4, 8)
            )
        self.assertIn("exceeds", str(ctx.exception))

    def test_non_divisible(self):
        """shape[d] must be evenly divisible by tile_shape[d]."""
        with self.assertRaises(ValueError) as ctx:
            tile_shape_to_partition_spec(
                shape=(7, 4), tile_shape=(3, 4)
            )
        self.assertIn("not evenly divisible", str(ctx.exception))

    def test_zero_tile(self):
        """tile_shape[d] must be positive."""
        with self.assertRaises(ValueError) as ctx:
            tile_shape_to_partition_spec(
                shape=(8, 8), tile_shape=(0, 8)
            )
        self.assertIn("must be positive", str(ctx.exception))

    def test_too_many_sharded_dims(self):
        """More sharded dims than available axis names."""
        with self.assertRaises(ValueError) as ctx:
            tile_shape_to_partition_spec(
                shape=(8, 8, 8, 8, 8),
                tile_shape=(4, 4, 4, 4, 4),
                mesh_axis_names=("x", "y"),
            )
        self.assertIn("mesh axes", str(ctx.exception))


class TestDerivePartitionSpecs(unittest.TestCase):
    """Test derive_partition_specs convenience function."""

    def test_sets_partition_spec_on_scheme(self):
        """derive_partition_specs should set scheme.partition_spec."""
        scheme = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        self.assertIsNone(scheme.partition_spec)

        spec = derive_partition_specs(scheme)
        self.assertEqual(spec, ("x", "y"))
        self.assertEqual(scheme.partition_spec, ("x", "y"))

    def test_fully_local_scheme(self):
        """Fully local scheme gets all-None partition spec."""
        scheme = TilingScheme(node=None, shape=(8, 8), tile_shape=(8, 8))
        spec = derive_partition_specs(scheme)
        self.assertEqual(spec, (None, None))
        self.assertEqual(scheme.partition_spec, (None, None))

    def test_case_assignment_consistency(self):
        """Partition spec should be consistent with get_case_assignments()."""
        from einjax.core.types import CaseAssignment

        scheme = TilingScheme(node=None, shape=(8, 6, 4), tile_shape=(4, 6, 2))
        spec = derive_partition_specs(scheme)
        cases = scheme.get_case_assignments()

        for d, (s, c) in enumerate(zip(spec, cases)):
            if c == CaseAssignment.UPPER:
                self.assertIsNotNone(s, f"dim {d} is UPPER but spec is None")
            else:
                self.assertIsNone(s, f"dim {d} is LOWER but spec is not None")


class TestInferMeshShape(unittest.TestCase):
    """Test infer_mesh_shape for device mesh dimension inference."""

    def test_single_axis(self):
        """Single axis gets all devices."""
        self.assertEqual(infer_mesh_shape(8, 1), (8,))

    def test_two_axes_square(self):
        """8 devices across 2 axes → balanced split."""
        shape = infer_mesh_shape(8, 2)
        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0] * shape[1], 8)

    def test_two_axes_power_of_two(self):
        """4 devices across 2 axes → (2, 2)."""
        shape = infer_mesh_shape(4, 2)
        self.assertEqual(shape, (2, 2))

    def test_prime_devices(self):
        """Prime number of devices across 2 axes → (1, p)."""
        shape = infer_mesh_shape(7, 2)
        self.assertEqual(len(shape), 2)
        self.assertEqual(shape[0] * shape[1], 7)

    def test_one_device(self):
        """Single device → (1,) or (1, 1)."""
        self.assertEqual(infer_mesh_shape(1, 1), (1,))
        shape = infer_mesh_shape(1, 2)
        self.assertEqual(shape[0] * shape[1], 1)

    def test_three_axes(self):
        """Three axes with 8 devices."""
        shape = infer_mesh_shape(8, 3)
        self.assertEqual(len(shape), 3)
        from math import prod
        self.assertEqual(prod(shape), 8)

    def test_invalid_num_devices(self):
        with self.assertRaises(ValueError):
            infer_mesh_shape(0, 1)

    def test_invalid_num_axes(self):
        with self.assertRaises(ValueError):
            infer_mesh_shape(8, 0)


class TestCreateMesh(unittest.TestCase):
    """Test create_mesh device mesh creation."""

    def test_single_axis_default(self):
        """Default single-axis mesh uses all available devices."""
        from einjax.sharding.mesh import create_mesh
        mesh = create_mesh(axis_names=("x",))
        self.assertEqual(len(mesh.axis_names), 1)
        self.assertEqual(mesh.axis_names, ("x",))

    def test_explicit_mesh_shape(self):
        """Explicit mesh_shape is respected."""
        from einjax.sharding.mesh import create_mesh
        mesh = create_mesh(axis_names=("x",), mesh_shape=(1,))
        self.assertEqual(mesh.shape, {"x": 1})

    def test_axis_names_mismatch_raises(self):
        """mesh_shape dims must match axis_names count."""
        from einjax.sharding.mesh import create_mesh
        with self.assertRaises(ValueError) as ctx:
            create_mesh(axis_names=("x",), mesh_shape=(1, 1))
        self.assertIn("axis_names", str(ctx.exception))

    def test_too_many_devices_raises(self):
        """Requesting more devices than available should raise."""
        from einjax.sharding.mesh import create_mesh
        with self.assertRaises(ValueError):
            create_mesh(axis_names=("x",), num_devices=9999)


class TestNeedsReshard(unittest.TestCase):
    """Test needs_reshard detection."""

    def test_same_tile_shape_no_reshard(self):
        """Same tile_shape means no resharding needed."""
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        self.assertFalse(needs_reshard(source, target))

    def test_different_tile_shape_needs_reshard(self):
        """Different tile_shape requires resharding."""
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 4))
        self.assertTrue(needs_reshard(source, target))

    def test_fully_local_to_sharded(self):
        """Going from fully local to sharded needs reshard."""
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(8, 8))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        self.assertTrue(needs_reshard(source, target))

    def test_sharded_to_fully_local(self):
        """Going from sharded to fully local needs reshard."""
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(8, 8))
        self.assertTrue(needs_reshard(source, target))


class TestEstimateReshardBytes(unittest.TestCase):
    """Test resharding data movement estimation."""

    def test_same_tile_shape_zero_bytes(self):
        """No data movement when tile shapes match."""
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        self.assertEqual(estimate_reshard_bytes(source, target), 0.0)

    def test_one_dim_changed(self):
        """One of two dims changes tiling → fraction = 1/2."""
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 4))
        # total elements = 64, dtype_size=4, fraction=1/2
        expected = 64 * 4 * 0.5
        self.assertAlmostEqual(estimate_reshard_bytes(source, target), expected)

    def test_all_dims_changed(self):
        """All dims change tiling → fraction = 1.0."""
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 2))
        expected = 64 * 4 * 1.0
        self.assertAlmostEqual(estimate_reshard_bytes(source, target), expected)

    def test_mismatched_shapes_raises(self):
        """Different tensor shapes should raise ValueError."""
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(16, 16), tile_shape=(4, 4))
        with self.assertRaises(ValueError) as ctx:
            estimate_reshard_bytes(source, target)
        self.assertIn("different tensor shapes", str(ctx.exception))

    def test_3d_partial_change(self):
        """3D tensor with 2 of 3 dims changing."""
        source = TilingScheme(node=None, shape=(8, 6, 4), tile_shape=(4, 6, 4))
        target = TilingScheme(node=None, shape=(8, 6, 4), tile_shape=(2, 3, 4))
        # 2 of 3 dims change, total elements = 192
        expected = 192 * 4 * (2.0 / 3.0)
        self.assertAlmostEqual(estimate_reshard_bytes(source, target), expected)

    def test_custom_dtype_size(self):
        """bf16 (2 bytes) reduces transfer estimate."""
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 4))
        bytes_f32 = estimate_reshard_bytes(source, target, dtype_size=4)
        bytes_bf16 = estimate_reshard_bytes(source, target, dtype_size=2)
        self.assertAlmostEqual(bytes_bf16, bytes_f32 / 2)


class TestEstimateReshardCost(unittest.TestCase):
    """Test resharding cost estimation."""

    def test_zero_cost_same_scheme(self):
        """Same scheme → zero bytes → zero time."""
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        cost = estimate_reshard_cost(source, target, interconnect_bandwidth=600e9)
        self.assertEqual(cost, 0.0)

    def test_positive_cost_different_scheme(self):
        """Different schemes → positive transfer time."""
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 4))
        cost = estimate_reshard_cost(source, target, interconnect_bandwidth=600e9)
        self.assertGreater(cost, 0.0)

    def test_faster_interconnect_lower_cost(self):
        """Faster interconnect → lower reshard cost."""
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 4))
        cost_slow = estimate_reshard_cost(source, target, interconnect_bandwidth=100e9)
        cost_fast = estimate_reshard_cost(source, target, interconnect_bandwidth=600e9)
        self.assertGreater(cost_slow, cost_fast)


class TestComputeTargetPartitionSpec(unittest.TestCase):
    """Test PartitionSpec derivation for reshard targets."""

    def test_sets_spec_on_target(self):
        """compute_target_partition_spec sets partition_spec on target."""
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        self.assertIsNone(target.partition_spec)
        spec = compute_target_partition_spec(target)
        self.assertEqual(spec, ("x", "y"))
        self.assertEqual(target.partition_spec, ("x", "y"))

    def test_fully_local_target(self):
        """Fully local target gets all-None spec."""
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(8, 8))
        spec = compute_target_partition_spec(target)
        self.assertEqual(spec, (None, None))


class TestReshardDense(unittest.TestCase):
    """Test reshard_dense for single-device mode."""

    def test_same_scheme_returns_array(self):
        """Same tile_shape is a no-op."""
        import numpy as np
        arr = np.ones((8, 8))
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        result = reshard_dense(arr, source, target)
        self.assertIs(result, arr)

    def test_no_mesh_returns_array(self):
        """Without mesh, resharding is a no-op (single device)."""
        import numpy as np
        arr = np.ones((8, 8))
        source = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        target = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 4))
        result = reshard_dense(arr, source, target, mesh=None)
        self.assertIs(result, arr)


class TestPlanReshardSequence(unittest.TestCase):
    """Test plan_reshard_sequence for identifying reshard points."""

    def test_empty_sequence(self):
        """Empty scheme list → no reshard pairs."""
        self.assertEqual(plan_reshard_sequence([]), [])

    def test_single_scheme(self):
        """Single scheme → no reshard pairs."""
        s = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        self.assertEqual(plan_reshard_sequence([s]), [])

    def test_all_same_no_reshards(self):
        """All same tile_shape → no reshard pairs."""
        schemes = [
            TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
            for _ in range(3)
        ]
        self.assertEqual(plan_reshard_sequence(schemes), [])

    def test_one_change(self):
        """One tile_shape change in a sequence of three."""
        s1 = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        s2 = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 4))
        s3 = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 4))
        pairs = plan_reshard_sequence([s1, s2, s3])
        self.assertEqual(pairs, [(0, 1)])

    def test_multiple_changes(self):
        """Multiple consecutive changes."""
        s1 = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        s2 = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 4))
        s3 = TilingScheme(node=None, shape=(8, 8), tile_shape=(8, 8))
        pairs = plan_reshard_sequence([s1, s2, s3])
        self.assertEqual(pairs, [(0, 1), (1, 2)])

    def test_alternating_changes(self):
        """Alternating tile shapes with no-ops in between."""
        s1 = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        s2 = TilingScheme(node=None, shape=(8, 8), tile_shape=(4, 4))
        s3 = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 2))
        s4 = TilingScheme(node=None, shape=(8, 8), tile_shape=(2, 2))
        pairs = plan_reshard_sequence([s1, s2, s3, s4])
        self.assertEqual(pairs, [(1, 2)])


class TestPackageShardingImports(unittest.TestCase):
    """Verify sharding exports from the einjax package."""

    def test_import_from_sharding(self):
        from einjax.sharding import (
            tile_shape_to_partition_spec,
            derive_partition_specs,
            create_mesh,
            infer_mesh_shape,
            needs_reshard,
            estimate_reshard_bytes,
            estimate_reshard_cost,
            reshard_dense,
            plan_reshard_sequence,
        )
        self.assertIsNotNone(tile_shape_to_partition_spec)
        self.assertIsNotNone(derive_partition_specs)
        self.assertIsNotNone(create_mesh)
        self.assertIsNotNone(infer_mesh_shape)
        self.assertIsNotNone(needs_reshard)
        self.assertIsNotNone(estimate_reshard_bytes)
        self.assertIsNotNone(estimate_reshard_cost)
        self.assertIsNotNone(reshard_dense)
        self.assertIsNotNone(plan_reshard_sequence)

    def test_import_from_top_level(self):
        import einjax
        self.assertTrue(hasattr(einjax, "tile_shape_to_partition_spec"))
        self.assertTrue(hasattr(einjax, "derive_partition_specs"))
        self.assertTrue(hasattr(einjax, "create_mesh"))
        self.assertTrue(hasattr(einjax, "infer_mesh_shape"))
        self.assertTrue(hasattr(einjax, "needs_reshard"))
        self.assertTrue(hasattr(einjax, "estimate_reshard_bytes"))
        self.assertTrue(hasattr(einjax, "estimate_reshard_cost"))
        self.assertTrue(hasattr(einjax, "reshard_dense"))
        self.assertTrue(hasattr(einjax, "plan_reshard_sequence"))


if __name__ == "__main__":
    unittest.main()
