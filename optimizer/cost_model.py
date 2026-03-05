"""
JAX-adapted cost model for the upper-case-lower-case EinSum system.

Implements CostModelConfig (PRD Section 4.1), hardware profiles (Section 4.2),
cost formulas (Section 4.3), and device detection (Section 9.2).

Adapted from einsql/einsql.py Reduction._update_cost (lines 907-988) and
the paper's Section 7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from einjax.core.types import TilingScheme


# =============================================================================
# Hardware Profiles (PRD Section 4.2)
# =============================================================================

_HARDWARE_PROFILES: dict[str, dict[str, float]] = {
    "gpu:a100": {
        "hbm_bandwidth": 2.0e12,           # 2.0 TB/s
        "interconnect_bandwidth": 600e9,    # 600 GB/s (NVLink)
        "peak_flops": 19.5e12,              # 19.5 TFLOPS fp32
        "launch_overhead": 5e-6,            # 5 µs
    },
    "gpu:h100": {
        "hbm_bandwidth": 3.35e12,           # 3.35 TB/s
        "interconnect_bandwidth": 900e9,    # 900 GB/s (NVLink)
        "peak_flops": 67e12,                # 67 TFLOPS fp32
        "launch_overhead": 3e-6,            # 3 µs
    },
    "gpu:v100": {
        "hbm_bandwidth": 900e9,             # 900 GB/s HBM2
        "interconnect_bandwidth": 300e9,    # 300 GB/s NVLink 2.0
        "peak_flops": 15.7e12,              # 15.7 TFLOPS fp32
        "launch_overhead": 7e-6,            # 7 µs
    },
    "tpu:v4": {
        "hbm_bandwidth": 1.2e12,            # 1.2 TB/s
        "interconnect_bandwidth": 4.8e12,   # 4.8 TB/s (ICI)
        "peak_flops": 275e12,               # 275 TFLOPS bf16
        "launch_overhead": 1e-6,            # 1 µs
    },
    "tpu:v5e": {
        "hbm_bandwidth": 1.6e12,            # 1.6 TB/s
        "interconnect_bandwidth": 6.4e12,   # 6.4 TB/s (ICI)
        "peak_flops": 394e12,               # 394 TFLOPS bf16
        "launch_overhead": 1e-6,            # 1 µs
    },
    "cpu": {
        "hbm_bandwidth": 50e9,              # 50 GB/s
        "interconnect_bandwidth": 50e9,     # N/A, use HBM as proxy
        "peak_flops": 0.5e12,               # 0.5 TFLOPS
        "launch_overhead": 10e-6,           # 10 µs
    },
}


@dataclass
class CostModelConfig:
    """Hardware-specific cost model parameters.

    Per PRD Section 4.1. Captures the hardware characteristics needed
    to compute transfer, kernel, and fixed costs for the DP optimizer.

    Attributes:
        hbm_bandwidth: Memory bandwidth in bytes/sec.
        interconnect_bandwidth: Device-to-device bandwidth in bytes/sec.
        peak_flops: Peak floating-point operations per second.
        launch_overhead: Kernel launch overhead in seconds.
        num_devices: Number of available compute devices.
        device_type: Identifier string (e.g., "gpu:a100", "tpu:v4", "cpu").
    """

    hbm_bandwidth: float
    interconnect_bandwidth: float
    peak_flops: float
    launch_overhead: float
    num_devices: int
    device_type: str

    @classmethod
    def from_device_type(cls, device_type: str, num_devices: int = 1) -> CostModelConfig:
        """Create config from a known device type string.

        Args:
            device_type: One of "gpu:a100", "gpu:h100", "gpu:v100", "tpu:v4", "tpu:v5e", "cpu".
            num_devices: Number of devices.

        Raises:
            ValueError: If device_type is not recognized.
        """
        if device_type not in _HARDWARE_PROFILES:
            known = ", ".join(sorted(_HARDWARE_PROFILES.keys()))
            raise ValueError(
                f"Unknown device type '{device_type}'. Known types: {known}"
            )
        profile = _HARDWARE_PROFILES[device_type]
        return cls(
            hbm_bandwidth=profile["hbm_bandwidth"],
            interconnect_bandwidth=profile["interconnect_bandwidth"],
            peak_flops=profile["peak_flops"],
            launch_overhead=profile["launch_overhead"],
            num_devices=num_devices,
            device_type=device_type,
        )

    def transfer_cost(self, bytes_transferred: float) -> float:
        """Compute data transfer cost in seconds (PRD Section 4.3 Cxfer).

        Args:
            bytes_transferred: Number of bytes to transfer between devices.

        Returns:
            Transfer time in seconds.
        """
        return bytes_transferred / self.interconnect_bandwidth

    def all_reduce_cost(self, output_bytes: float) -> float:
        """Compute all-reduce communication cost when aggregation crosses devices.

        Uses the ring all-reduce model: cost = output_bytes * log2(num_devices)
        / interconnect_bandwidth.

        Args:
            output_bytes: Bytes of output data that must be reduced across devices.

        Returns:
            All-reduce time in seconds. Returns 0 for single-device configs.
        """
        if self.num_devices <= 1:
            return 0.0
        return (
            output_bytes * math.log2(self.num_devices)
            / self.interconnect_bandwidth
        )

    def parallelism_overhead(self) -> float:
        """Compute synchronization barrier overhead for multi-device execution.

        Each device synchronization point costs approximately one launch_overhead.

        Returns:
            Synchronization overhead in seconds. Returns 0 for single-device.
        """
        if self.num_devices <= 1:
            return 0.0
        return self.launch_overhead * self.num_devices

    def kernel_cost(self, flops: float) -> float:
        """Compute kernel execution cost in seconds (PRD Section 4.3 Ckernel).

        When num_devices > 1, FLOPs are divided across devices (each device
        handles 1/num_devices of the tiles).

        Args:
            flops: Total floating-point operations across all devices.

        Returns:
            Kernel time in seconds (per-device wall-clock time).
        """
        return flops / (self.peak_flops * self.num_devices)

    def fixed_cost(self, num_launches: int) -> float:
        """Compute fixed launch overhead in seconds (PRD Section 4.3 Cfixed).

        Args:
            num_launches: Number of kernel launches.

        Returns:
            Total launch overhead in seconds.
        """
        return num_launches * self.launch_overhead

    def reshard_cost(self, reshard_bytes: float) -> float:
        """Compute resharding cost in seconds (PRD Section 4.3).

        Args:
            reshard_bytes: Bytes of data movement for repartitioning.

        Returns:
            Resharding time in seconds.
        """
        return reshard_bytes / self.interconnect_bandwidth

    def total_cost(
        self,
        bytes_transferred: float,
        flops: float,
        num_launches: int,
    ) -> float:
        """Compute total operation cost: Cxfer + Ckernel + Cfixed.

        Args:
            bytes_transferred: Bytes for transfer cost.
            flops: FLOPs for kernel cost.
            num_launches: Number of kernel launches for fixed cost.

        Returns:
            Total cost in seconds.
        """
        return (
            self.transfer_cost(bytes_transferred)
            + self.kernel_cost(flops)
            + self.fixed_cost(num_launches)
        )


def compute_join_cost(
    lhs: TilingScheme,
    rhs: TilingScheme,
    join_key_dims: list[int],
    agg_key_dims: list[int],
    output_tile_shape: tuple[int, ...],
    config: CostModelConfig,
    dtype_size: int = 4,
) -> tuple[float, float, float, float]:
    """Compute cost for a binary tensor contraction.

    Adapted from einsql Reduction._update_cost (lines 907-988) with
    hardware-aware cost parameters from PRD Section 4.3.

    When num_devices > 1, the cost model accounts for parallelism:
    - join_num_tuples is scaled by 1/num_devices for local kernel cost
    - An all-reduce term is added for cross-device aggregation
    - Synchronization barrier overhead is included

    Args:
        lhs: Left-hand tiling scheme.
        rhs: Right-hand tiling scheme.
        join_key_dims: Dimension indices that are join keys.
        agg_key_dims: Dimension indices that are aggregation keys.
        output_tile_shape: Tile shape of the output tensor.
        config: Hardware cost model configuration.
        dtype_size: Bytes per element (default 4 for float32).

    Returns:
        Tuple of (total_cost, comm_cost, kernel_flops, fixed_cost).
    """
    # Transfer cost (PRD Section 4.3)
    bytes_transferred = dtype_size * (
        (len(lhs.tile_shape) + lhs.tile_size) * lhs.num_tuples
        + (len(rhs.tile_shape) + rhs.tile_size) * rhs.num_tuples
    )
    comm = config.transfer_cost(bytes_transferred)

    # Join num_tuples estimation
    join_denom = prod(
        max(lhs.value_count[k], rhs.value_count[k]) for k in join_key_dims
    ) if join_key_dims else 1
    join_num_tuples = (lhs.num_tuples * rhs.num_tuples) / join_denom

    # Kernel FLOPs (PRD Section 4.3)
    tile_join_cost = (lhs.tile_size * rhs.tile_size) / prod(
        output_tile_shape[k] for k in join_key_dims
    ) if join_key_dims else lhs.tile_size * rhs.tile_size
    agg_cost = prod(output_tile_shape[k] for k in agg_key_dims) if agg_key_dims else 0
    join_kernel_cost = 2 * tile_join_cost - agg_cost
    flops = join_num_tuples * join_kernel_cost
    # kernel_cost() already divides by num_devices internally
    kernel = config.kernel_cost(flops)

    # Fixed cost — launches are distributed across devices
    num_launches = int(join_num_tuples) + (int(join_num_tuples) if agg_key_dims else 0)
    per_device_launches = num_launches / config.num_devices
    fixed = config.fixed_cost(int(per_device_launches))

    # Multi-device communication: all-reduce only when aggregation keys exist,
    # meaning output tiles are partially computed on different devices and must
    # be reduced.
    if agg_key_dims:
        output_tile_size = prod(output_tile_shape)
        output_bytes = dtype_size * output_tile_size * join_num_tuples
        all_reduce = config.all_reduce_cost(output_bytes)
        comm += all_reduce

    # Synchronization overhead
    sync = config.parallelism_overhead()

    total = comm + kernel + fixed + sync
    return total, comm, flops, fixed


def detect_device_config() -> CostModelConfig:
    """Auto-detect hardware and return a cost model config.

    Per PRD Section 9.2. Checks jax.devices()[0].platform and device_kind
    to identify GPU model or TPU generation. Falls back to CPU profile.

    Returns:
        CostModelConfig populated with detected hardware parameters.
    """
    try:
        import jax
        devices = jax.devices()
        num_devices = len(devices)
        device = devices[0]
        platform = device.platform  # "gpu", "tpu", "cpu"

        if platform == "gpu":
            device_kind = getattr(device, "device_kind", "").lower()
            if "h100" in device_kind:
                device_type = "gpu:h100"
            elif "a100" in device_kind:
                device_type = "gpu:a100"
            elif "v100" in device_kind:
                device_type = "gpu:v100"
            else:
                # Default to A100 profile for unknown GPUs
                device_type = "gpu:a100"
        elif platform == "tpu":
            device_kind = getattr(device, "device_kind", "").lower()
            if "v5" in device_kind:
                device_type = "tpu:v5e"
            elif "v4" in device_kind:
                device_type = "tpu:v4"
            else:
                device_type = "tpu:v4"
        else:
            device_type = "cpu"

        return CostModelConfig.from_device_type(device_type, num_devices)

    except (ImportError, RuntimeError):
        # JAX not available or no devices — use CPU fallback
        return CostModelConfig.from_device_type("cpu", 1)


def _calibrate_interconnect(
    devices: list,
    num_bytes: int = 64 * 1024 * 1024,
    trials: int = 5,
) -> float:
    """Measure device-to-device interconnect bandwidth.

    Allocates an array on one device, transfers it to another via
    ``jax.device_put``, and measures the throughput. Averages over
    several (device_i, device_j) pairs to smooth out NVLink topology
    variance (V100 has non-uniform NVLink connectivity).

    Args:
        devices: List of JAX devices (must have len >= 2).
        num_bytes: Payload size in bytes for each transfer (default 64 MB).
        trials: Number of timing trials per device pair.

    Returns:
        Measured interconnect bandwidth in bytes/sec.
    """
    import time
    import jax
    import jax.numpy as jnp

    if len(devices) < 2:
        # Cannot measure interconnect with a single device.
        return 0.0

    num_elements = num_bytes // 4  # float32
    # Pick device pairs: (0,1), (2,3), (4,5), ... plus (0, last) for
    # cross-pair measurement. Cap at 4 pairs to keep calibration fast.
    pairs: list[tuple[int, int]] = []
    for i in range(0, len(devices) - 1, 2):
        pairs.append((i, i + 1))
    # Add a cross-pair link if we have enough devices
    if len(devices) >= 4 and (0, len(devices) - 1) not in pairs:
        pairs.append((0, len(devices) - 1))
    pairs = pairs[:4]

    bandwidths: list[float] = []
    for src_idx, dst_idx in pairs:
        src_dev = devices[src_idx]
        dst_dev = devices[dst_idx]

        # Allocate on source device
        data = jax.device_put(jnp.ones(num_elements, dtype=jnp.float32), src_dev)
        data.block_until_ready()

        # Warmup transfer
        warmup = jax.device_put(data, dst_dev)
        warmup.block_until_ready()

        pair_time = 0.0
        for _ in range(trials):
            start = time.perf_counter()
            transferred = jax.device_put(data, dst_dev)
            transferred.block_until_ready()
            pair_time += time.perf_counter() - start

        avg_time = pair_time / trials
        if avg_time > 0:
            bandwidths.append(num_bytes / avg_time)

    if not bandwidths:
        return 0.0
    return sum(bandwidths) / len(bandwidths)


def calibrate() -> CostModelConfig:
    """Run microbenchmarks to measure actual hardware parameters.

    Per PRD Section 4.4. Measures HBM bandwidth, interconnect bandwidth,
    kernel launch overhead, and peak FLOPs via simple JAX operations.

    Returns:
        CostModelConfig with calibrated values.
    """
    import time
    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    num_devices = len(devices)

    # Start from detected defaults
    base = detect_device_config()

    # Measure peak FLOPs via dense matmul
    n = 2048
    a = jnp.ones((n, n), dtype=jnp.float32)
    b = jnp.ones((n, n), dtype=jnp.float32)
    # Warmup
    _ = jnp.dot(a, b).block_until_ready()

    trials = 5
    total_time = 0.0
    for _ in range(trials):
        start = time.perf_counter()
        _ = jnp.dot(a, b).block_until_ready()
        total_time += time.perf_counter() - start
    avg_time = total_time / trials
    matmul_flops = 2.0 * n * n * n  # 2N^3 for matrix multiply
    measured_peak_flops = matmul_flops / avg_time

    # Measure HBM bandwidth via large copy
    size = 256 * 1024 * 1024 // 4  # 256 MB of float32
    x = jnp.ones(size, dtype=jnp.float32)
    _ = (x + 0).block_until_ready()  # warmup

    total_time = 0.0
    for _ in range(trials):
        start = time.perf_counter()
        _ = (x + 0).block_until_ready()
        total_time += time.perf_counter() - start
    avg_time = total_time / trials
    bytes_moved = 2 * size * 4  # read + write
    measured_hbm_bw = bytes_moved / avg_time

    # Measure launch overhead via trivial kernel
    small = jnp.ones(1, dtype=jnp.float32)
    _ = (small + 0).block_until_ready()  # warmup

    num_iters = 100
    start = time.perf_counter()
    for _ in range(num_iters):
        _ = (small + 0).block_until_ready()
    total_time = time.perf_counter() - start
    measured_launch_overhead = total_time / num_iters

    # Measure interconnect bandwidth (device-to-device transfer)
    if num_devices >= 2:
        measured_interconnect_bw = _calibrate_interconnect(devices)
    else:
        measured_interconnect_bw = base.interconnect_bandwidth

    return CostModelConfig(
        hbm_bandwidth=measured_hbm_bw,
        interconnect_bandwidth=measured_interconnect_bw,
        peak_flops=measured_peak_flops,
        launch_overhead=measured_launch_overhead,
        num_devices=num_devices,
        device_type=base.device_type,
    )
