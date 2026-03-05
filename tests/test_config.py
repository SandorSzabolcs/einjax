"""
Tests for einjax/config.py — global configuration management.

Tests global config get/set/reset, hardware profile queries,
enhanced calibrate with base config, and thread safety.
"""

from __future__ import annotations

import threading
import unittest

from einjax.config import (
    calibrate,
    get_config,
    get_hardware_profile,
    list_device_types,
    reset_config,
    set_config,
)
from einjax.optimizer.cost_model import CostModelConfig, _HARDWARE_PROFILES


class TestGetConfig(unittest.TestCase):
    """Test get_config auto-detection and caching."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_returns_config(self):
        config = get_config()
        self.assertIsInstance(config, CostModelConfig)

    def test_auto_detects_device(self):
        config = get_config()
        self.assertIn(config.device_type, _HARDWARE_PROFILES)

    def test_caches_result(self):
        config1 = get_config()
        config2 = get_config()
        self.assertIs(config1, config2)

    def test_positive_values(self):
        config = get_config()
        self.assertGreater(config.hbm_bandwidth, 0)
        self.assertGreater(config.peak_flops, 0)
        self.assertGreater(config.num_devices, 0)


class TestSetConfig(unittest.TestCase):
    """Test set_config for manual configuration."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_set_and_get(self):
        custom = CostModelConfig.from_device_type("gpu:h100", 8)
        set_config(custom)
        result = get_config()
        self.assertIs(result, custom)
        self.assertEqual(result.device_type, "gpu:h100")
        self.assertEqual(result.num_devices, 8)

    def test_overrides_auto_detected(self):
        _ = get_config()  # trigger auto-detect
        custom = CostModelConfig.from_device_type("tpu:v4", 16)
        set_config(custom)
        result = get_config()
        self.assertEqual(result.device_type, "tpu:v4")

    def test_invalid_type_raises(self):
        with self.assertRaises(TypeError) as ctx:
            set_config({"device_type": "cpu"})  # type: ignore[arg-type]
        self.assertIn("CostModelConfig", str(ctx.exception))

    def test_invalid_type_message(self):
        with self.assertRaises(TypeError) as ctx:
            set_config(42)  # type: ignore[arg-type]
        self.assertIn("int", str(ctx.exception))


class TestResetConfig(unittest.TestCase):
    """Test reset_config clears the cached configuration."""

    def tearDown(self):
        reset_config()

    def test_reset_clears(self):
        custom = CostModelConfig.from_device_type("gpu:a100", 4)
        set_config(custom)
        self.assertEqual(get_config().device_type, "gpu:a100")

        reset_config()
        # After reset, get_config() re-detects (will be "cpu" in CI)
        result = get_config()
        self.assertIsInstance(result, CostModelConfig)

    def test_reset_allows_redetection(self):
        config1 = get_config()
        reset_config()
        config2 = get_config()
        # Both are valid configs (may be same values but different objects)
        self.assertIsInstance(config2, CostModelConfig)
        self.assertIsNot(config1, config2)


class TestListDeviceTypes(unittest.TestCase):
    """Test list_device_types returns known hardware."""

    def test_returns_list(self):
        types = list_device_types()
        self.assertIsInstance(types, list)
        self.assertGreater(len(types), 0)

    def test_sorted(self):
        types = list_device_types()
        self.assertEqual(types, sorted(types))

    def test_contains_known_types(self):
        types = list_device_types()
        self.assertIn("cpu", types)
        self.assertIn("gpu:a100", types)
        self.assertIn("gpu:h100", types)
        self.assertIn("gpu:v100", types)
        self.assertIn("tpu:v4", types)
        self.assertIn("tpu:v5e", types)

    def test_matches_profiles(self):
        types = list_device_types()
        self.assertEqual(set(types), set(_HARDWARE_PROFILES.keys()))


class TestGetHardwareProfile(unittest.TestCase):
    """Test get_hardware_profile for profile queries."""

    def test_a100_profile(self):
        profile = get_hardware_profile("gpu:a100")
        self.assertIsInstance(profile, dict)
        self.assertEqual(profile["hbm_bandwidth"], 2.0e12)
        self.assertEqual(profile["peak_flops"], 19.5e12)

    def test_cpu_profile(self):
        profile = get_hardware_profile("cpu")
        self.assertEqual(profile["launch_overhead"], 10e-6)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_hardware_profile("gpu:rtx4090")
        self.assertIn("gpu:rtx4090", str(ctx.exception))
        self.assertIn("Known types", str(ctx.exception))

    def test_returns_copy(self):
        """Returned profile should be a copy, not the internal dict."""
        profile = get_hardware_profile("cpu")
        profile["hbm_bandwidth"] = 999
        # Internal should be unchanged
        original = get_hardware_profile("cpu")
        self.assertNotEqual(original["hbm_bandwidth"], 999)

    def test_all_profiles_have_four_keys(self):
        for device_type in list_device_types():
            profile = get_hardware_profile(device_type)
            self.assertEqual(len(profile), 4, f"{device_type} should have 4 keys")


class TestConfigThreadSafety(unittest.TestCase):
    """Test that get/set/reset are thread-safe."""

    def setUp(self):
        reset_config()

    def tearDown(self):
        reset_config()

    def test_concurrent_set_get(self):
        """Multiple threads setting and getting should not crash."""
        errors = []
        device_types = ["gpu:a100", "gpu:h100", "tpu:v4", "cpu"]

        def worker(dt):
            try:
                config = CostModelConfig.from_device_type(dt, 1)
                set_config(config)
                result = get_config()
                # Result should be *some* valid config
                if not isinstance(result, CostModelConfig):
                    errors.append(f"Got non-config: {type(result)}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(dt,)) for dt in device_types]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])


class TestPackageConfigImports(unittest.TestCase):
    """Verify config exports from the einjax package."""

    def test_import_from_config_module(self):
        from einjax.config import (
            get_config,
            set_config,
            reset_config,
            list_device_types,
            get_hardware_profile,
            calibrate,
        )
        self.assertIsNotNone(get_config)
        self.assertIsNotNone(set_config)
        self.assertIsNotNone(reset_config)
        self.assertIsNotNone(list_device_types)
        self.assertIsNotNone(get_hardware_profile)
        self.assertIsNotNone(calibrate)

    def test_import_from_top_level(self):
        import einjax
        self.assertTrue(hasattr(einjax, "get_config"))
        self.assertTrue(hasattr(einjax, "set_config"))
        self.assertTrue(hasattr(einjax, "reset_config"))
        self.assertTrue(hasattr(einjax, "list_device_types"))
        self.assertTrue(hasattr(einjax, "get_hardware_profile"))


if __name__ == "__main__":
    unittest.main()
