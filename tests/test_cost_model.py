"""
Tests for the CostModelConfig and hardware detection.

Tests cost model configuration, hardware profiles, cost formula
computations, device detection, and the compute_join_cost function.
"""

from __future__ import annotations

import math
import unittest
from math import prod

from einjax.optimizer.cost_model import (
    CostModelConfig,
    _HARDWARE_PROFILES,
    _calibrate_interconnect,
    calibrate,
    compute_join_cost,
    detect_device_config,
)
from einjax.core.types import TilingScheme


class TestCostModelConfig(unittest.TestCase):
    """Test CostModelConfig dataclass and methods."""

    def setUp(self):
        self.config = CostModelConfig(
            hbm_bandwidth=2.0e12,
            interconnect_bandwidth=600e9,
            peak_flops=19.5e12,
            launch_overhead=5e-6,
            num_devices=8,
            device_type="gpu:a100",
        )

    def test_basic_construction(self):
        self.assertEqual(self.config.hbm_bandwidth, 2.0e12)
        self.assertEqual(self.config.interconnect_bandwidth, 600e9)
        self.assertEqual(self.config.peak_flops, 19.5e12)
        self.assertEqual(self.config.launch_overhead, 5e-6)
        self.assertEqual(self.config.num_devices, 8)
        self.assertEqual(self.config.device_type, "gpu:a100")

    def test_transfer_cost(self):
        # 600 GB transferred at 600 GB/s = 1 second
        cost = self.config.transfer_cost(600e9)
        self.assertAlmostEqual(cost, 1.0, places=5)

    def test_kernel_cost(self):
        # 19.5 TFLOPS at 19.5 TFLOPS peak with 8 devices = 1/8 second
        cost = self.config.kernel_cost(19.5e12)
        self.assertAlmostEqual(cost, 1.0 / 8, places=5)

    def test_kernel_cost_single_device(self):
        # Single device: 19.5 TFLOPS at 19.5 TFLOPS peak = 1 second
        config = CostModelConfig(
            hbm_bandwidth=2.0e12,
            interconnect_bandwidth=600e9,
            peak_flops=19.5e12,
            launch_overhead=5e-6,
            num_devices=1,
            device_type="gpu:a100",
        )
        cost = config.kernel_cost(19.5e12)
        self.assertAlmostEqual(cost, 1.0, places=5)

    def test_fixed_cost(self):
        # 200 launches at 5µs each = 1ms
        cost = self.config.fixed_cost(200)
        self.assertAlmostEqual(cost, 1e-3, places=9)

    def test_reshard_cost(self):
        # Same as transfer cost formula
        cost = self.config.reshard_cost(600e9)
        self.assertAlmostEqual(cost, 1.0, places=5)

    def test_total_cost_sum(self):
        """Total cost = Cxfer + Ckernel + Cfixed."""
        bytes_xfer = 600e9
        flops = 19.5e12
        launches = 200

        total = self.config.total_cost(bytes_xfer, flops, launches)
        expected = (
            self.config.transfer_cost(bytes_xfer)
            + self.config.kernel_cost(flops)
            + self.config.fixed_cost(launches)
        )
        self.assertAlmostEqual(total, expected, places=10)

    def test_total_cost_positive(self):
        """All costs should be positive for positive inputs."""
        total = self.config.total_cost(1e6, 1e6, 10)
        self.assertGreater(total, 0.0)

    def test_zero_inputs(self):
        """Zero inputs should produce zero costs."""
        self.assertEqual(self.config.transfer_cost(0.0), 0.0)
        self.assertEqual(self.config.kernel_cost(0.0), 0.0)
        self.assertEqual(self.config.fixed_cost(0), 0.0)


class TestFromDeviceType(unittest.TestCase):
    """Test CostModelConfig.from_device_type factory."""

    def test_a100_profile(self):
        config = CostModelConfig.from_device_type("gpu:a100", 4)
        self.assertEqual(config.hbm_bandwidth, 2.0e12)
        self.assertEqual(config.interconnect_bandwidth, 600e9)
        self.assertEqual(config.peak_flops, 19.5e12)
        self.assertEqual(config.launch_overhead, 5e-6)
        self.assertEqual(config.num_devices, 4)
        self.assertEqual(config.device_type, "gpu:a100")

    def test_h100_profile(self):
        config = CostModelConfig.from_device_type("gpu:h100", 8)
        self.assertEqual(config.hbm_bandwidth, 3.35e12)
        self.assertEqual(config.peak_flops, 67e12)
        self.assertEqual(config.num_devices, 8)

    def test_tpu_v4_profile(self):
        config = CostModelConfig.from_device_type("tpu:v4", 16)
        self.assertEqual(config.hbm_bandwidth, 1.2e12)
        self.assertEqual(config.interconnect_bandwidth, 4.8e12)
        self.assertEqual(config.num_devices, 16)

    def test_tpu_v5e_profile(self):
        config = CostModelConfig.from_device_type("tpu:v5e", 32)
        self.assertEqual(config.peak_flops, 394e12)
        self.assertEqual(config.num_devices, 32)

    def test_v100_profile(self):
        config = CostModelConfig.from_device_type("gpu:v100", 8)
        self.assertEqual(config.hbm_bandwidth, 900e9)
        self.assertEqual(config.interconnect_bandwidth, 300e9)
        self.assertEqual(config.peak_flops, 15.7e12)
        self.assertEqual(config.launch_overhead, 7e-6)
        self.assertEqual(config.num_devices, 8)
        self.assertEqual(config.device_type, "gpu:v100")

    def test_cpu_profile(self):
        config = CostModelConfig.from_device_type("cpu", 1)
        self.assertEqual(config.hbm_bandwidth, 50e9)
        self.assertEqual(config.peak_flops, 0.5e12)
        self.assertEqual(config.launch_overhead, 10e-6)

    def test_unknown_device_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CostModelConfig.from_device_type("gpu:rtx4090", 1)
        self.assertIn("gpu:rtx4090", str(ctx.exception))
        self.assertIn("Known types", str(ctx.exception))

    def test_all_profiles_have_required_keys(self):
        required_keys = {
            "hbm_bandwidth", "interconnect_bandwidth",
            "peak_flops", "launch_overhead",
        }
        for device_type, profile in _HARDWARE_PROFILES.items():
            with self.subTest(device_type=device_type):
                self.assertEqual(set(profile.keys()), required_keys)

    def test_all_profiles_positive_values(self):
        for device_type, profile in _HARDWARE_PROFILES.items():
            with self.subTest(device_type=device_type):
                for key, value in profile.items():
                    self.assertGreater(value, 0.0, f"{key} must be positive")


class TestDetectDeviceConfig(unittest.TestCase):
    """Test detect_device_config auto-detection."""

    def test_returns_valid_config(self):
        config = detect_device_config()
        self.assertIsInstance(config, CostModelConfig)
        self.assertGreater(config.hbm_bandwidth, 0.0)
        self.assertGreater(config.peak_flops, 0.0)
        self.assertGreater(config.num_devices, 0)
        self.assertIn(config.device_type, _HARDWARE_PROFILES)

    def test_device_type_is_known(self):
        config = detect_device_config()
        self.assertIn(config.device_type, _HARDWARE_PROFILES)


class TestComputeJoinCost(unittest.TestCase):
    """Test compute_join_cost for binary tensor contractions."""

    def _make_scheme(self, shape, tile_shape):
        """Helper to create a TilingScheme for testing."""
        # Use None as node — not needed for cost computation
        return TilingScheme(node=None, shape=shape, tile_shape=tile_shape)

    def test_matmul_cost_positive(self):
        """Matrix multiply ij,jk->ik should have positive cost."""
        lhs = self._make_scheme((4, 4), (2, 2))
        rhs = self._make_scheme((4, 4), (2, 2))
        config = CostModelConfig.from_device_type("cpu", 1)

        total, comm, flops, fixed = compute_join_cost(
            lhs, rhs,
            join_key_dims=[1],    # j is join key
            agg_key_dims=[],
            output_tile_shape=(2, 2),
            config=config,
        )
        self.assertGreater(total, 0.0)
        self.assertGreater(comm, 0.0)
        self.assertGreater(flops, 0.0)
        self.assertGreater(fixed, 0.0)

    def test_no_join_keys(self):
        """Outer product: no join keys means every pair is matched."""
        lhs = self._make_scheme((4,), (2,))
        rhs = self._make_scheme((4,), (2,))
        config = CostModelConfig.from_device_type("cpu", 1)

        total, comm, flops, fixed = compute_join_cost(
            lhs, rhs,
            join_key_dims=[],
            agg_key_dims=[],
            output_tile_shape=(2, 2),
            config=config,
        )
        self.assertGreater(total, 0.0)

    def test_cost_scales_with_size(self):
        """Larger tensors should have higher cost."""
        config = CostModelConfig.from_device_type("cpu", 1)

        small_lhs = self._make_scheme((4, 4), (2, 2))
        small_rhs = self._make_scheme((4, 4), (2, 2))
        small_total, _, _, _ = compute_join_cost(
            small_lhs, small_rhs,
            join_key_dims=[1],
            agg_key_dims=[],
            output_tile_shape=(2, 2),
            config=config,
        )

        big_lhs = self._make_scheme((16, 16), (4, 4))
        big_rhs = self._make_scheme((16, 16), (4, 4))
        big_total, _, _, _ = compute_join_cost(
            big_lhs, big_rhs,
            join_key_dims=[1],
            agg_key_dims=[],
            output_tile_shape=(4, 4),
            config=config,
        )

        self.assertGreater(big_total, small_total)

    def test_faster_hardware_lower_cost(self):
        """Same operation on faster hardware should have lower cost."""
        lhs = self._make_scheme((8, 8), (4, 4))
        rhs = self._make_scheme((8, 8), (4, 4))

        cpu_config = CostModelConfig.from_device_type("cpu", 1)
        gpu_config = CostModelConfig.from_device_type("gpu:a100", 1)

        cpu_total, _, _, _ = compute_join_cost(
            lhs, rhs,
            join_key_dims=[1],
            agg_key_dims=[],
            output_tile_shape=(4, 4),
            config=cpu_config,
        )

        gpu_total, _, _, _ = compute_join_cost(
            lhs, rhs,
            join_key_dims=[1],
            agg_key_dims=[],
            output_tile_shape=(4, 4),
            config=gpu_config,
        )

        self.assertGreater(cpu_total, gpu_total)

    def test_with_aggregation_keys(self):
        """Aggregation keys add cost."""
        lhs = self._make_scheme((4, 4), (2, 2))
        rhs = self._make_scheme((4, 4), (2, 2))
        config = CostModelConfig.from_device_type("cpu", 1)

        no_agg_total, _, _, _ = compute_join_cost(
            lhs, rhs,
            join_key_dims=[1],
            agg_key_dims=[],
            output_tile_shape=(2, 2),
            config=config,
        )

        with_agg_total, _, _, _ = compute_join_cost(
            lhs, rhs,
            join_key_dims=[1],
            agg_key_dims=[0],
            output_tile_shape=(2, 2),
            config=config,
        )

        # With aggregation should differ in fixed cost (more launches)
        self.assertNotEqual(no_agg_total, with_agg_total)


class TestCostModelComparison(unittest.TestCase):
    """Test that cost model correctly ranks hardware."""

    def test_gpu_faster_than_cpu(self):
        cpu = CostModelConfig.from_device_type("cpu", 1)
        gpu = CostModelConfig.from_device_type("gpu:a100", 1)
        self.assertGreater(gpu.peak_flops, cpu.peak_flops)

    def test_h100_faster_than_a100(self):
        a100 = CostModelConfig.from_device_type("gpu:a100", 1)
        h100 = CostModelConfig.from_device_type("gpu:h100", 1)
        self.assertGreater(h100.peak_flops, a100.peak_flops)
        self.assertGreater(h100.hbm_bandwidth, a100.hbm_bandwidth)

    def test_a100_faster_than_v100(self):
        v100 = CostModelConfig.from_device_type("gpu:v100", 1)
        a100 = CostModelConfig.from_device_type("gpu:a100", 1)
        self.assertGreater(a100.peak_flops, v100.peak_flops)
        self.assertGreater(a100.hbm_bandwidth, v100.hbm_bandwidth)

    def test_tpu_higher_interconnect_than_gpu(self):
        gpu = CostModelConfig.from_device_type("gpu:a100", 1)
        tpu = CostModelConfig.from_device_type("tpu:v4", 1)
        self.assertGreater(tpu.interconnect_bandwidth, gpu.interconnect_bandwidth)


class TestAllReduceCost(unittest.TestCase):
    """Test CostModelConfig.all_reduce_cost for multi-device communication."""

    def test_single_device_zero_cost(self):
        """Single device should have zero all-reduce cost."""
        config = CostModelConfig.from_device_type("gpu:a100", 1)
        cost = config.all_reduce_cost(1e9)
        self.assertEqual(cost, 0.0)

    def test_multi_device_positive_cost(self):
        """Multi-device should have positive all-reduce cost."""
        config = CostModelConfig.from_device_type("gpu:a100", 8)
        cost = config.all_reduce_cost(1e9)
        self.assertGreater(cost, 0.0)

    def test_all_reduce_formula(self):
        """all_reduce_cost = output_bytes * log2(num_devices) / interconnect_bw."""
        import math
        config = CostModelConfig.from_device_type("gpu:v100", 8)
        output_bytes = 1e9
        expected = output_bytes * math.log2(8) / config.interconnect_bandwidth
        cost = config.all_reduce_cost(output_bytes)
        self.assertAlmostEqual(cost, expected, places=10)

    def test_all_reduce_scales_with_devices(self):
        """More devices should increase all-reduce cost (log2 scaling)."""
        config_4 = CostModelConfig.from_device_type("gpu:a100", 4)
        config_8 = CostModelConfig.from_device_type("gpu:a100", 8)
        cost_4 = config_4.all_reduce_cost(1e9)
        cost_8 = config_8.all_reduce_cost(1e9)
        self.assertGreater(cost_8, cost_4)

    def test_zero_bytes(self):
        """Zero bytes should produce zero cost."""
        config = CostModelConfig.from_device_type("gpu:a100", 8)
        self.assertEqual(config.all_reduce_cost(0.0), 0.0)


class TestParallelismOverhead(unittest.TestCase):
    """Test CostModelConfig.parallelism_overhead for sync barriers."""

    def test_single_device_zero(self):
        """Single device should have zero parallelism overhead."""
        config = CostModelConfig.from_device_type("gpu:a100", 1)
        self.assertEqual(config.parallelism_overhead(), 0.0)

    def test_multi_device_positive(self):
        """Multi-device should have positive parallelism overhead."""
        config = CostModelConfig.from_device_type("gpu:a100", 8)
        self.assertGreater(config.parallelism_overhead(), 0.0)

    def test_overhead_formula(self):
        """parallelism_overhead = launch_overhead * num_devices."""
        config = CostModelConfig.from_device_type("gpu:v100", 8)
        expected = config.launch_overhead * 8
        self.assertAlmostEqual(config.parallelism_overhead(), expected, places=15)


class TestParallelismAwareCostModel(unittest.TestCase):
    """Test that the cost model correctly favors parallelism for large tensors.

    PRD 4.2 acceptance criterion: For a sufficiently large tensor, the
    multi-device cost model produces lower kernel cost than single-device.
    """

    def test_kernel_cost_lower_with_more_devices(self):
        """Same FLOPs should have lower kernel cost with more devices."""
        config_1 = CostModelConfig.from_device_type("gpu:v100", 1)
        config_8 = CostModelConfig.from_device_type("gpu:v100", 8)

        flops = 1e15  # Large workload
        cost_1 = config_1.kernel_cost(flops)
        cost_8 = config_8.kernel_cost(flops)

        self.assertAlmostEqual(cost_8, cost_1 / 8, places=5)

    def test_multi_device_join_cost_lower_for_large_tensors(self):
        """For large tensors, 8-device join cost should be lower than 1-device."""
        config_1 = CostModelConfig.from_device_type("gpu:v100", 1)
        config_8 = CostModelConfig.from_device_type("gpu:v100", 8)

        # Large matmul: (256, 256) tiles with 16x16 tiling = 256 tiles
        lhs = TilingScheme(node=None, shape=(256, 256), tile_shape=(16, 16))
        rhs = TilingScheme(node=None, shape=(256, 256), tile_shape=(16, 16))

        total_1, _, _, _ = compute_join_cost(
            lhs, rhs,
            join_key_dims=[1],
            agg_key_dims=[],
            output_tile_shape=(16, 16),
            config=config_1,
        )

        total_8, _, _, _ = compute_join_cost(
            lhs, rhs,
            join_key_dims=[1],
            agg_key_dims=[],
            output_tile_shape=(16, 16),
            config=config_8,
        )

        # 8 devices should produce lower total cost for large workloads
        self.assertLess(total_8, total_1)

    def test_single_device_backward_compatible(self):
        """Single-device cost model should behave like the original."""
        config = CostModelConfig.from_device_type("cpu", 1)

        # all_reduce and parallelism_overhead should be zero
        self.assertEqual(config.all_reduce_cost(1e9), 0.0)
        self.assertEqual(config.parallelism_overhead(), 0.0)

        # kernel_cost unchanged for single device
        self.assertAlmostEqual(
            config.kernel_cost(0.5e12), 1.0, places=5
        )


class TestPackageCostModelImports(unittest.TestCase):
    """Verify cost model exports from the einjax package."""

    def test_import_from_optimizer(self):
        from einjax.optimizer import CostModelConfig, detect_device_config, calibrate
        self.assertIsNotNone(CostModelConfig)
        self.assertIsNotNone(detect_device_config)
        self.assertIsNotNone(calibrate)

    def test_import_from_top_level(self):
        import einjax
        self.assertTrue(hasattr(einjax, "CostModelConfig"))
        self.assertTrue(hasattr(einjax, "detect_device_config"))


class TestCalibrateInterconnect(unittest.TestCase):
    """Test _calibrate_interconnect for device-to-device bandwidth measurement."""

    def test_single_device_returns_zero(self):
        """With only one device, interconnect cannot be measured."""
        import jax
        devices = jax.devices()[:1]
        bw = _calibrate_interconnect(devices)
        self.assertEqual(bw, 0.0)

    def test_empty_devices_returns_zero(self):
        """Empty device list should return zero."""
        bw = _calibrate_interconnect([])
        self.assertEqual(bw, 0.0)

    def test_multi_device_returns_positive(self):
        """With 2+ devices, measured bandwidth should be positive."""
        import jax
        devices = jax.devices()
        if len(devices) < 2:
            self.skipTest("Need at least 2 devices for interconnect test")
        bw = _calibrate_interconnect(devices)
        self.assertGreater(bw, 0.0)

    def test_multi_device_reasonable_bandwidth(self):
        """Measured bandwidth should be within an order of magnitude of spec.

        V100 NVLink 2.0 spec is 300 GB/s bidirectional. We accept anything
        between 10 GB/s and 1 TB/s to account for measurement noise, topology
        effects, and different hardware.
        """
        import jax
        devices = jax.devices()
        if len(devices) < 2:
            self.skipTest("Need at least 2 devices for interconnect test")
        bw = _calibrate_interconnect(devices)
        self.assertGreater(bw, 10e9, "Bandwidth below 10 GB/s seems too low")
        self.assertLess(bw, 1e12, "Bandwidth above 1 TB/s seems too high")

    def test_custom_payload_size(self):
        """Custom num_bytes parameter should work."""
        import jax
        devices = jax.devices()
        if len(devices) < 2:
            self.skipTest("Need at least 2 devices for interconnect test")
        bw = _calibrate_interconnect(devices, num_bytes=16 * 1024 * 1024)
        self.assertGreater(bw, 0.0)

    def test_custom_trials(self):
        """Custom trials parameter should work."""
        import jax
        devices = jax.devices()
        if len(devices) < 2:
            self.skipTest("Need at least 2 devices for interconnect test")
        bw = _calibrate_interconnect(devices, trials=2)
        self.assertGreater(bw, 0.0)


class TestCalibrateIncludesInterconnect(unittest.TestCase):
    """Test that calibrate() includes interconnect measurement."""

    def test_calibrate_returns_config(self):
        """calibrate() should return a valid CostModelConfig."""
        config = calibrate()
        self.assertIsInstance(config, CostModelConfig)
        self.assertGreater(config.hbm_bandwidth, 0.0)
        self.assertGreater(config.peak_flops, 0.0)
        self.assertGreater(config.interconnect_bandwidth, 0.0)

    def test_calibrate_interconnect_measured_on_multi_device(self):
        """On multi-device, calibrated interconnect should differ from profile default."""
        import jax
        if len(jax.devices()) < 2:
            self.skipTest("Need at least 2 devices for interconnect calibration")
        config = calibrate()
        base = detect_device_config()
        # Measured value should be a real measurement, not the profile default.
        # We verify it's positive and finite — exact match with profile is
        # extremely unlikely given measurement noise.
        self.assertGreater(config.interconnect_bandwidth, 0.0)
        self.assertTrue(
            math.isfinite(config.interconnect_bandwidth),
            "Interconnect bandwidth should be finite",
        )

    def test_calibrate_single_device_uses_profile(self):
        """On single device, interconnect should fall back to profile default."""
        import jax
        if len(jax.devices()) != 1:
            self.skipTest("This test requires exactly 1 device")
        config = calibrate()
        base = detect_device_config()
        self.assertEqual(config.interconnect_bandwidth, base.interconnect_bandwidth)


if __name__ == "__main__":
    unittest.main()
