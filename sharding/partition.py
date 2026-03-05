"""Tile shape to PartitionSpec mapping.

Implements PRD Section 3.3 and Section 5.2: derives JAX PartitionSpec from
tile_shape by classifying each dimension as upper-case (sharded) or
lower-case (kernel).

Upper-case dimensions (tile_shape[d] < shape[d]) are mapped to mesh axis
names in the PartitionSpec. Lower-case dimensions (tile_shape[d] == shape[d])
are mapped to None (replicated / local).
"""

from __future__ import annotations

from einjax.core.types import CaseAssignment, TilingScheme


def tile_shape_to_partition_spec(
    shape: tuple[int, ...],
    tile_shape: tuple[int, ...],
    mesh_axis_names: tuple[str, ...] = ("x", "y", "z", "w"),
) -> tuple:
    """Derive a PartitionSpec from tensor shape and tile shape.

    Per PRD Section 3.3:
    - Dimensions where tile_shape[d] < shape[d] are UPPER (sharded) and get
      a mesh axis name.
    - Dimensions where tile_shape[d] == shape[d] are LOWER (kernel) and get
      None.

    Mesh axis names are assigned in order to the sharded dimensions. If there
    are more sharded dimensions than available mesh axes, a ValueError is
    raised.

    Args:
        shape: Full logical tensor shape.
        tile_shape: Tile dimensions for each axis.
        mesh_axis_names: Available mesh axis names to assign to sharded dims.

    Returns:
        A tuple suitable for jax.sharding.PartitionSpec construction.
        Each element is either a mesh axis name (str) or None.

    Raises:
        ValueError: If shape and tile_shape have different lengths, if any
            tile dimension doesn't evenly divide the tensor dimension, or
            if more sharded dimensions than mesh axes.
    """
    if len(shape) != len(tile_shape):
        raise ValueError(
            f"shape ({len(shape)}D) and tile_shape ({len(tile_shape)}D) "
            f"must have the same number of dimensions"
        )

    for d, (s, t) in enumerate(zip(shape, tile_shape)):
        if t > s:
            raise ValueError(
                f"tile_shape[{d}]={t} exceeds shape[{d}]={s}"
            )
        if t <= 0:
            raise ValueError(
                f"tile_shape[{d}]={t} must be positive"
            )
        if s % t != 0:
            raise ValueError(
                f"shape[{d}]={s} is not evenly divisible by tile_shape[{d}]={t}"
            )

    # Classify each dimension
    cases = [
        CaseAssignment.LOWER if t == s else CaseAssignment.UPPER
        for s, t in zip(shape, tile_shape)
    ]

    # Count sharded (upper-case) dimensions
    num_sharded = sum(1 for c in cases if c == CaseAssignment.UPPER)
    if num_sharded > len(mesh_axis_names):
        raise ValueError(
            f"Need {num_sharded} mesh axes for sharded dimensions but only "
            f"{len(mesh_axis_names)} axis names available: {mesh_axis_names}"
        )

    # Assign mesh axis names to sharded dimensions in order
    spec = []
    axis_idx = 0
    for case in cases:
        if case == CaseAssignment.UPPER:
            spec.append(mesh_axis_names[axis_idx])
            axis_idx += 1
        else:
            spec.append(None)

    return tuple(spec)


def derive_partition_specs(
    scheme: TilingScheme,
    mesh_axis_names: tuple[str, ...] = ("x", "y", "z", "w"),
) -> tuple:
    """Derive PartitionSpec for a TilingScheme and store it on the scheme.

    Convenience wrapper around tile_shape_to_partition_spec that also
    sets the partition_spec field on the TilingScheme.

    Args:
        scheme: A TilingScheme with shape and tile_shape set.
        mesh_axis_names: Available mesh axis names.

    Returns:
        The derived PartitionSpec tuple.
    """
    spec = tile_shape_to_partition_spec(
        scheme.shape, scheme.tile_shape, mesh_axis_names
    )
    scheme.partition_spec = spec
    return spec
