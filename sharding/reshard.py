"""Repartitioning logic for converting tensors between tiling schemes.

Implements PRD Section 6.4: when consecutive operations in the DAG need
different tiling, the output of one must be resharded to match the input
tiling of the next.

For dense tensors: uses jax.device_put with the target NamedSharding.
For sparse (SparseTensorRelation): recomputes tile coords under the new
tile_shape, re-sorts, and merges tiles sharing the same output coord.

Also provides cost estimation for resharding decisions in the DP optimizer.
"""

from __future__ import annotations

from math import prod

from einjax.core.types import TilingScheme
from einjax.sharding.partition import tile_shape_to_partition_spec


def needs_reshard(
    source: TilingScheme,
    target: TilingScheme,
) -> bool:
    """Determine whether resharding is needed between two tiling schemes.

    Resharding is needed when the tile shapes differ, since that changes
    which dimensions are sharded (upper-case) vs. local (kernel).

    Args:
        source: Tiling scheme of the producer operation.
        target: Tiling scheme of the consumer operation.

    Returns:
        True if resharding is required.
    """
    return source.tile_shape != target.tile_shape


def estimate_reshard_bytes(
    source: TilingScheme,
    target: TilingScheme,
    dtype_size: int = 4,
) -> float:
    """Estimate bytes transferred during resharding.

    When tile shapes change, tiles must be split or merged. In the worst
    case, every element in the tensor must be communicated. In practice,
    only dimensions that change sharding need data movement.

    The estimate is the total tensor size in bytes multiplied by the
    fraction of dimensions whose tiling changes. If no dimensions change,
    zero bytes are needed.

    Args:
        source: Source tiling scheme.
        target: Target tiling scheme.
        dtype_size: Bytes per element (default 4 for float32).

    Returns:
        Estimated bytes transferred.
    """
    if source.shape != target.shape:
        raise ValueError(
            f"Cannot reshard between different tensor shapes: "
            f"{source.shape} vs {target.shape}"
        )

    if source.tile_shape == target.tile_shape:
        return 0.0

    ndim = len(source.shape)
    # Count dimensions where tiling changes
    changed_dims = sum(
        1 for s, t in zip(source.tile_shape, target.tile_shape) if s != t
    )

    # Total tensor elements
    total_elements = prod(source.shape)

    # Fraction of data that needs movement: proportional to changed dims
    fraction = changed_dims / ndim if ndim > 0 else 0.0

    return total_elements * dtype_size * fraction


def estimate_reshard_cost(
    source: TilingScheme,
    target: TilingScheme,
    interconnect_bandwidth: float,
    dtype_size: int = 4,
) -> float:
    """Estimate resharding time in seconds.

    Per PRD Section 4.3:
        reshard_cost = reshard_bytes / interconnect_bandwidth

    Args:
        source: Source tiling scheme.
        target: Target tiling scheme.
        interconnect_bandwidth: Device-to-device bandwidth in bytes/sec.
        dtype_size: Bytes per element (default 4 for float32).

    Returns:
        Estimated resharding time in seconds.
    """
    reshard_bytes = estimate_reshard_bytes(source, target, dtype_size)
    return reshard_bytes / interconnect_bandwidth


def compute_target_partition_spec(
    target: TilingScheme,
    mesh_axis_names: tuple[str, ...] = ("x", "y", "z", "w"),
) -> tuple:
    """Compute the PartitionSpec for a target tiling scheme.

    Convenience function that derives the PartitionSpec from the target
    scheme's shape and tile_shape, and stores it on the scheme.

    Args:
        target: Target tiling scheme.
        mesh_axis_names: Available mesh axis names.

    Returns:
        The derived PartitionSpec tuple.
    """
    spec = tile_shape_to_partition_spec(
        target.shape, target.tile_shape, mesh_axis_names
    )
    target.partition_spec = spec
    return spec


def reshard_dense(
    array,
    source: TilingScheme,
    target: TilingScheme,
    mesh=None,
    mesh_axis_names: tuple[str, ...] = ("x", "y", "z", "w"),
):
    """Reshard a dense JAX array from one tiling scheme to another.

    Uses jax.device_put with the target NamedSharding to trigger
    JAX's built-in resharding (which uses XLA's GSPMD / Shardy).

    If no mesh is provided, returns the array unchanged (single-device
    mode where resharding is a no-op logically).

    Args:
        array: JAX array to reshard.
        source: Current tiling scheme.
        target: Desired tiling scheme.
        mesh: jax.sharding.Mesh for device placement. If None,
            resharding is a no-op (single device).
        mesh_axis_names: Mesh axis names for PartitionSpec derivation.

    Returns:
        The resharded JAX array.
    """
    if source.tile_shape == target.tile_shape:
        return array

    if mesh is None:
        return array

    import jax

    target_spec = compute_target_partition_spec(target, mesh_axis_names)
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*target_spec)
    )
    return jax.device_put(array, sharding)


def plan_reshard_sequence(
    schemes: list[TilingScheme],
) -> list[tuple[int, int]]:
    """Identify which consecutive operations need resharding.

    Given an ordered list of tiling schemes (from the DP optimizer's
    execution plan), returns pairs of indices (i, i+1) where resharding
    is needed between operation i's output and operation i+1's input.

    Args:
        schemes: Ordered list of tiling schemes in execution order.

    Returns:
        List of (source_index, target_index) pairs that need resharding.
    """
    reshard_pairs = []
    for i in range(len(schemes) - 1):
        if needs_reshard(schemes[i], schemes[i + 1]):
            reshard_pairs.append((i, i + 1))
    return reshard_pairs
