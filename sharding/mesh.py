"""Device mesh helpers for EinJAX.

Provides utilities for creating JAX device meshes from available hardware.
Per PRD Section 5.1 and Section 9.1, all sharding uses JAX's backend-
independent Mesh + NamedSharding + PartitionSpec APIs.
"""

from __future__ import annotations

from math import prod


def infer_mesh_shape(
    num_devices: int,
    num_axes: int,
) -> tuple[int, ...]:
    """Infer a balanced mesh shape for the given number of devices and axes.

    Distributes devices across axes as evenly as possible. For a single axis,
    the mesh shape is (num_devices,). For multiple axes, greedily assigns
    the largest prime factor to the first axis, then recurses.

    Args:
        num_devices: Total number of devices.
        num_axes: Number of mesh axes to distribute across.

    Returns:
        Tuple of axis sizes whose product equals num_devices.

    Raises:
        ValueError: If num_devices < 1 or num_axes < 1.
    """
    if num_devices < 1:
        raise ValueError(f"num_devices must be >= 1, got {num_devices}")
    if num_axes < 1:
        raise ValueError(f"num_axes must be >= 1, got {num_axes}")

    if num_axes == 1:
        return (num_devices,)

    # Find the largest factor of num_devices that is <= num_devices^(1/num_axes)
    # to get a balanced split
    target = int(round(num_devices ** (1.0 / num_axes)))

    # Search for the closest factor to target
    best = 1
    for f in range(1, num_devices + 1):
        if num_devices % f == 0 and f <= target:
            best = f

    remaining = num_devices // best
    rest = infer_mesh_shape(remaining, num_axes - 1)
    return (best,) + rest


def create_mesh(
    axis_names: tuple[str, ...] = ("x",),
    num_devices: int | None = None,
    mesh_shape: tuple[int, ...] | None = None,
):
    """Create a JAX device mesh for SPMD computation.

    Per PRD Section 9.1, uses jax.sharding.Mesh for backend-independent
    device placement.

    Args:
        axis_names: Names for each mesh axis (e.g., ("x",) or ("x", "y")).
        num_devices: Number of devices to use. If None, uses all available.
            Ignored if mesh_shape is provided.
        mesh_shape: Explicit shape for the mesh. If None, inferred from
            num_devices and len(axis_names).

    Returns:
        A jax.sharding.Mesh instance.

    Raises:
        ValueError: If mesh_shape product doesn't match available devices,
            or axis_names length doesn't match mesh dimensions.
    """
    import jax
    import numpy as np

    devices = jax.devices()
    available = len(devices)

    if mesh_shape is not None:
        if len(mesh_shape) != len(axis_names):
            raise ValueError(
                f"mesh_shape has {len(mesh_shape)} dims but "
                f"{len(axis_names)} axis_names provided"
            )
        total = prod(mesh_shape)
        if total > available:
            raise ValueError(
                f"mesh_shape {mesh_shape} requires {total} devices "
                f"but only {available} available"
            )
        device_array = np.array(devices[:total]).reshape(mesh_shape)
    else:
        if num_devices is None:
            num_devices = available
        if num_devices > available:
            raise ValueError(
                f"Requested {num_devices} devices but only {available} available"
            )
        shape = infer_mesh_shape(num_devices, len(axis_names))
        device_array = np.array(devices[:num_devices]).reshape(shape)

    return jax.sharding.Mesh(device_array, axis_names)
