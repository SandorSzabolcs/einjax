"""
Global configuration management for EinJAX.

Provides a thread-safe global configuration context for hardware cost model
parameters. Users can set a default config once and have it used automatically
by einsum(), analyze(), the DP optimizer, and the execution engine.

Per PRD Section 2 (config.py — "Hardware detection, cost model params")
and Section 4.4 (calibrate with optional base config).

Example:
    import einjax
    from einjax.config import get_config, set_config, list_device_types

    # Auto-detect hardware
    config = get_config()

    # Override with specific hardware
    set_config(einjax.CostModelConfig.from_device_type("gpu:h100", 8))

    # List available hardware profiles
    print(list_device_types())

    # Calibrate from microbenchmarks, using current config as base
    calibrated = einjax.calibrate(config=get_config())
"""

from __future__ import annotations

import threading
from typing import Any

from .optimizer.cost_model import (
    CostModelConfig,
    _HARDWARE_PROFILES,
    calibrate as _calibrate_raw,
    detect_device_config,
)


# =============================================================================
# Global Configuration State
# =============================================================================

_config_lock = threading.Lock()
_global_config: CostModelConfig | None = None


def get_config() -> CostModelConfig:
    """Get the current global cost model configuration.

    If no configuration has been set, auto-detects the hardware
    via detect_device_config() and caches the result.

    Returns:
        The active CostModelConfig.

    Example:
        >>> from einjax.config import get_config
        >>> config = get_config()
        >>> config.device_type
        'cpu'
    """
    global _global_config
    with _config_lock:
        if _global_config is None:
            _global_config = detect_device_config()
        return _global_config


def set_config(config: CostModelConfig) -> None:
    """Set the global cost model configuration.

    Args:
        config: A CostModelConfig instance to use as the default.

    Raises:
        TypeError: If config is not a CostModelConfig instance.

    Example:
        >>> from einjax.config import set_config
        >>> from einjax import CostModelConfig
        >>> set_config(CostModelConfig.from_device_type("gpu:a100", 4))
    """
    if not isinstance(config, CostModelConfig):
        raise TypeError(
            f"Expected CostModelConfig, got {type(config).__name__}"
        )
    global _global_config
    with _config_lock:
        _global_config = config


def reset_config() -> None:
    """Reset the global configuration to None.

    The next call to get_config() will re-detect hardware.
    """
    global _global_config
    with _config_lock:
        _global_config = None


# =============================================================================
# Hardware Profile Queries
# =============================================================================

def list_device_types() -> list[str]:
    """List all known hardware device types.

    Returns:
        Sorted list of device type strings (e.g., ["cpu", "gpu:a100", ...]).
    """
    return sorted(_HARDWARE_PROFILES.keys())


def get_hardware_profile(device_type: str) -> dict[str, float]:
    """Get the hardware profile for a specific device type.

    Args:
        device_type: One of the known device type strings.

    Returns:
        Dictionary with hardware parameters (hbm_bandwidth,
        interconnect_bandwidth, peak_flops, launch_overhead).

    Raises:
        ValueError: If device_type is not recognized.
    """
    if device_type not in _HARDWARE_PROFILES:
        known = ", ".join(sorted(_HARDWARE_PROFILES.keys()))
        raise ValueError(
            f"Unknown device type '{device_type}'. Known types: {known}"
        )
    return dict(_HARDWARE_PROFILES[device_type])


# =============================================================================
# Enhanced calibrate (PRD Section 4.4)
# =============================================================================

def calibrate(config: CostModelConfig | None = None) -> CostModelConfig:
    """Run microbenchmarks to measure actual hardware parameters.

    Per PRD Section 4.4. Measures HBM bandwidth (via large memcpy),
    peak FLOPs (via dense matmul), and kernel launch overhead
    (via trivial kernel timing).

    If a base config is provided, its interconnect_bandwidth is preserved
    (since that requires multi-device benchmarking). Otherwise, auto-detects
    the device first.

    Args:
        config: Optional base config. If None, uses detect_device_config().

    Returns:
        CostModelConfig with calibrated values. Also updates the global
        config via set_config().
    """
    calibrated = _calibrate_raw()

    # If user provided a base config, preserve its interconnect bandwidth
    # and device metadata
    if config is not None:
        calibrated = CostModelConfig(
            hbm_bandwidth=calibrated.hbm_bandwidth,
            interconnect_bandwidth=config.interconnect_bandwidth,
            peak_flops=calibrated.peak_flops,
            launch_overhead=calibrated.launch_overhead,
            num_devices=config.num_devices,
            device_type=config.device_type,
        )

    set_config(calibrated)
    return calibrated
