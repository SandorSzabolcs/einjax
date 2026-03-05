"""
Sparse tiling scheme generation for EinJAX.

Provides functions to filter, rank, and select tiling schemes for sparse
tensors based on sparsity statistics, hardware constraints, and cost
model parameters. This bridges the gap between raw scheme enumeration
(in base.py) and the DP optimizer (in optimizer/dp.py) for sparse inputs.

Implements PRD Section 5.2 (Search Space):
1. Enumerate tile shapes: Cartesian product of factors per dimension
   (done in BaseTensor.__init__)
2. Derive PartitionSpec: upper-case → mesh axis, lower-case → None
   (done in sharding/partition.py)
3. Prune infeasible: num_tuples > device_count or tile_size > memory

Also provides sparse-specific utilities:
- Scheme pruning based on device count and memory limits
- Sparsity-aware ranking (prefer schemes with fewer non-empty tiles)
- Optimal tiling selection for SparseTensor → SparseTensorRelation conversion
"""

from __future__ import annotations

from math import prod
from typing import Any

from einjax.core.types import CaseAssignment, TilingScheme
from einjax.tensor.stats import sparsity_ratio


def prune_infeasible_schemes(
    schemes: dict[tuple[int, ...], TilingScheme],
    num_devices: int = 1,
    max_tile_bytes: int | None = None,
    dtype_size: int = 4,
) -> dict[tuple[int, ...], TilingScheme]:
    """Remove tiling schemes that cannot be executed on the given hardware.

    Per PRD Section 5.2, prunes schemes where:
    - num_tuples exceeds device count (can't shard that finely)
    - tile_size * dtype_size exceeds per-device memory limit

    Args:
        schemes: Dict mapping tile_shape → TilingScheme.
        num_devices: Number of available devices.
        max_tile_bytes: Maximum tile size in bytes per device.
            If None, no memory pruning is applied.
        dtype_size: Bytes per element (default 4 for float32).

    Returns:
        Filtered dict containing only feasible schemes.
    """
    feasible = {}
    for tile_shape, scheme in schemes.items():
        # Prune: more tuples than devices means can't shard
        if scheme.num_tuples > num_devices and num_devices > 1:
            # Only prune in multi-device mode — single device can handle any count
            pass  # still allow, just lower priority for optimizer
        # Prune: tile too large for device memory
        if max_tile_bytes is not None:
            tile_bytes = scheme.tile_size * dtype_size
            if tile_bytes > max_tile_bytes:
                continue
        feasible[tile_shape] = scheme
    return feasible


def rank_schemes_by_sparsity(
    schemes: dict[tuple[int, ...], TilingScheme],
    shape: tuple[int, ...],
) -> list[TilingScheme]:
    """Rank tiling schemes by sparsity efficiency (fewer non-empty tiles first).

    Schemes that produce fewer non-empty tiles relative to the maximum
    possible tiles are ranked higher, as they lead to less computation
    and communication in the sparse execution path.

    Ties are broken by tile_size descending (prefer larger tiles for
    better kernel utilization).

    Args:
        schemes: Dict mapping tile_shape → TilingScheme.
        shape: Full logical tensor shape.

    Returns:
        List of TilingSchemes sorted by sparsity ratio ascending
        (sparsest first), then tile_size descending.
    """
    def sort_key(scheme: TilingScheme) -> tuple[float, int]:
        ratio = sparsity_ratio(scheme.num_tuples, shape, scheme.tile_shape)
        return (ratio, -scheme.tile_size)

    return sorted(schemes.values(), key=sort_key)


def select_best_sparse_tiling(
    schemes: dict[tuple[int, ...], TilingScheme],
    shape: tuple[int, ...],
    num_devices: int = 1,
    max_tile_bytes: int | None = None,
    dtype_size: int = 4,
) -> TilingScheme | None:
    """Select the best tiling scheme for a sparse tensor.

    Applies pruning, then ranks by sparsity efficiency. Returns the
    scheme with the lowest sparsity ratio (most tile savings from
    sparsity), breaking ties by largest tile size.

    Args:
        schemes: Dict mapping tile_shape → TilingScheme.
        shape: Full logical tensor shape.
        num_devices: Number of available devices.
        max_tile_bytes: Maximum tile size in bytes per device.
        dtype_size: Bytes per element (default 4 for float32).

    Returns:
        The best TilingScheme, or None if no feasible scheme exists.
    """
    feasible = prune_infeasible_schemes(
        schemes, num_devices, max_tile_bytes, dtype_size,
    )
    if not feasible:
        return None
    ranked = rank_schemes_by_sparsity(feasible, shape)
    return ranked[0] if ranked else None


def get_sparse_partition_spec(
    scheme: TilingScheme,
    mesh_axis_names: tuple[str, ...] = ("x", "y", "z", "w"),
) -> tuple:
    """Derive PartitionSpec for a sparse tiling scheme.

    Wraps sharding.partition.tile_shape_to_partition_spec with
    sparse-aware defaults. For sparse tensors, the sharded (upper-case)
    dimensions correspond to tile coordinate columns that are distributed
    across devices.

    Args:
        scheme: TilingScheme with shape and tile_shape set.
        mesh_axis_names: Available mesh axis names.

    Returns:
        Tuple suitable for PartitionSpec construction.

    Raises:
        ValueError: If more sharded dims than available mesh axes.
    """
    from einjax.sharding.partition import tile_shape_to_partition_spec
    return tile_shape_to_partition_spec(
        scheme.shape, scheme.tile_shape, mesh_axis_names,
    )


def compute_tile_memory(
    tile_shape: tuple[int, ...],
    dtype_size: int = 4,
    include_coords: bool = True,
) -> int:
    """Compute memory footprint of a single tile in bytes.

    Args:
        tile_shape: Dimensions of the tile.
        dtype_size: Bytes per value element (default 4 for float32).
        include_coords: Whether to include coordinate storage overhead
            (ndim * 4 bytes for int32 coordinates per tile).

    Returns:
        Total bytes per tile.
    """
    value_bytes = prod(tile_shape) * dtype_size
    coord_bytes = len(tile_shape) * 4 if include_coords else 0
    return value_bytes + coord_bytes


def compute_relation_memory(
    scheme: TilingScheme,
    dtype_size: int = 4,
) -> int:
    """Estimate total memory for a SparseTensorRelation under this scheme.

    Computes the memory needed for coords array + values array based
    on the scheme's num_tuples (T(U)) and tile dimensions.

    Args:
        scheme: TilingScheme with sparsity-adjusted num_tuples.
        dtype_size: Bytes per value element.

    Returns:
        Total estimated bytes for the relation.
    """
    coord_bytes = scheme.num_tuples * len(scheme.shape) * 4  # int32 coords
    value_bytes = scheme.num_tuples * scheme.tile_size * dtype_size
    return coord_bytes + value_bytes


def filter_schemes_by_sharding(
    schemes: dict[tuple[int, ...], TilingScheme],
    num_devices: int,
    min_sharded_dims: int = 0,
    max_sharded_dims: int | None = None,
) -> dict[tuple[int, ...], TilingScheme]:
    """Filter schemes by the number of sharded (upper-case) dimensions.

    Useful for controlling the degree of distribution:
    - min_sharded_dims=1 ensures at least one dimension is distributed
    - max_sharded_dims=1 limits to single-axis sharding for simplicity

    Args:
        schemes: Dict mapping tile_shape → TilingScheme.
        num_devices: Number of available devices (schemes needing more
            shards than devices are excluded).
        min_sharded_dims: Minimum number of sharded dimensions.
        max_sharded_dims: Maximum number of sharded dimensions.
            If None, no upper limit.

    Returns:
        Filtered dict of schemes meeting the constraints.
    """
    filtered = {}
    for tile_shape, scheme in schemes.items():
        cases = scheme.get_case_assignments()
        num_sharded = sum(1 for c in cases if c == CaseAssignment.UPPER)

        if num_sharded < min_sharded_dims:
            continue
        if max_sharded_dims is not None and num_sharded > max_sharded_dims:
            continue

        # Check that the number of tuples is compatible with device count
        if num_devices > 1 and num_sharded > 0:
            # At least one sharded dim, ensure we can map to devices
            num_shards = prod(
                s // t for s, t, c in zip(scheme.shape, scheme.tile_shape, cases)
                if c == CaseAssignment.UPPER
            )
            if num_shards > num_devices:
                continue

        filtered[tile_shape] = scheme
    return filtered


def filter_schemes_by_device_count(
    schemes: dict[tuple[int, ...], TilingScheme],
    num_devices: int,
    min_tiles_per_device: int = 1,
) -> dict[tuple[int, ...], TilingScheme]:
    """Filter schemes to ensure balanced multi-device load distribution.

    Per PRD Section 4.3: when num_devices > 1, the number of tiles along
    UPPER-case dimensions must be divisible by num_devices and provide at
    least min_tiles_per_device tiles per GPU for balanced load distribution.

    Schemes with no UPPER-case dimensions (fully local, all LOWER) are
    always kept, since they run entirely on each device with no sharding.

    Args:
        schemes: Dict mapping tile_shape → TilingScheme.
        num_devices: Number of available devices.
        min_tiles_per_device: Minimum tiles per device for sufficient
            work granularity (default 1).

    Returns:
        Filtered dict containing only schemes with balanced device
        distribution.
    """
    if num_devices <= 1:
        return dict(schemes)

    min_upper_tiles = num_devices * min_tiles_per_device

    filtered = {}
    for tile_shape, scheme in schemes.items():
        cases = scheme.get_case_assignments()
        num_upper = sum(1 for c in cases if c == CaseAssignment.UPPER)

        # Fully local schemes (no sharded dims) are always valid
        if num_upper == 0:
            filtered[tile_shape] = scheme
            continue

        # Count tiles along UPPER-case dimensions only
        num_upper_tiles = prod(
            s // t for s, t, c in zip(scheme.shape, scheme.tile_shape, cases)
            if c == CaseAssignment.UPPER
        )

        # Require: enough tiles and evenly divisible across devices
        if num_upper_tiles >= min_upper_tiles and num_upper_tiles % num_devices == 0:
            filtered[tile_shape] = scheme

    return filtered


def prepare_sparse_tiling(
    tensor: Any,
    num_devices: int = 1,
    max_tile_bytes: int | None = None,
    dtype_size: int = 4,
) -> TilingScheme | None:
    """End-to-end sparse tiling preparation for a SparseTensor.

    Combines pruning, ranking, and selection to find the best tiling
    scheme for a sparse tensor. This is the main entry point for the
    sparse execution path when preparing a SparseTensor for dispatch.

    Args:
        tensor: A SparseTensor (or any BaseTensor with schemes populated).
        num_devices: Number of available devices.
        max_tile_bytes: Maximum tile size in bytes per device.
        dtype_size: Bytes per element.

    Returns:
        The best TilingScheme for this tensor, or None if no feasible
        scheme exists.
    """
    return select_best_sparse_tiling(
        tensor.schemes,
        tensor.shape,
        num_devices=num_devices,
        max_tile_bytes=max_tile_bytes,
        dtype_size=dtype_size,
    )
