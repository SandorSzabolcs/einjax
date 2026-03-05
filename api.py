"""
User-facing API for EinJAX.

Provides einsum(), analyze(), and with_mesh() as the primary entry points.
Per PRD Section 4.8, the API accepts Einstein summation notation and dense
arrays, with optional multi-GPU execution via mesh parameters.

Example (single device):
    import einjax
    import jax.numpy as jnp

    A = jnp.ones((3, 4))
    B = jnp.ones((4, 5))
    result = einjax.einsum("ij,jk->ik", A, B)

Example (multi-GPU):
    result = einjax.einsum("ij,jk->ik", A, B, num_devices=8)

Example (context manager):
    with einjax.with_mesh(mesh):
        result = einjax.einsum("ij,jk->ik", A, B)
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import Any, Generator

import jax
import jax.numpy as jnp
import numpy as np

from .core.notation import (
    get_label_dimensions,
    normalize_notation,
    validate_inputs,
)
from .execution.dense_kernels import execute_dense_einsum, execute_sharded_einsum
from .sharding.mesh import create_mesh


# =============================================================================
# Thread-local mesh context for with_mesh()
# =============================================================================

_mesh_context = threading.local()


def _get_context_mesh() -> jax.sharding.Mesh | None:
    """Return the mesh set by the innermost with_mesh() context, or None."""
    stack = getattr(_mesh_context, "stack", None)
    if stack:
        return stack[-1]
    return None


@contextlib.contextmanager
def with_mesh(mesh: jax.sharding.Mesh) -> Generator[jax.sharding.Mesh, None, None]:
    """Context manager that sets a mesh for all einjax operations.

    Any call to ``einjax.einsum()`` inside this context will use the
    provided mesh for sharded execution, unless the call passes its
    own explicit ``mesh`` argument.

    Args:
        mesh: A ``jax.sharding.Mesh`` to use for sharded execution.

    Yields:
        The same mesh (for convenience).

    Example:
        >>> import einjax
        >>> from einjax import create_mesh
        >>> mesh = create_mesh(("x",), num_devices=8)
        >>> with einjax.with_mesh(mesh):
        ...     result = einjax.einsum("ij,jk->ik", A, B)
    """
    if not hasattr(_mesh_context, "stack"):
        _mesh_context.stack = []
    _mesh_context.stack.append(mesh)
    try:
        yield mesh
    finally:
        _mesh_context.stack.pop()


def einsum(
    einsum_string: str,
    *operands: Any,
    num_devices: int | None = None,
    mesh: jax.sharding.Mesh | None = None,
    mesh_shape: tuple[int, ...] | None = None,
    gather: bool = False,
) -> jnp.ndarray:
    """Compute Einstein summation on dense arrays using JAX.

    Supports single-device and multi-GPU execution. When a mesh is
    provided (explicitly, via context manager, or auto-created from
    ``num_devices``), the computation is distributed across devices
    using XLA's GSPMD.

    Args:
        einsum_string: Einstein summation notation (e.g., "ij,jk->ik").
            Supports both explicit ("ij,jk->ik") and implicit ("ij,jk")
            output notation following NumPy convention.
        *operands: Input arrays (numpy.ndarray, jax.Array, or any
            arraylike accepted by jnp.asarray).
        num_devices: Number of devices to use.
            - None (default): auto-detect via ``get_config().num_devices``.
              Uses single-device path if only 1 device is detected.
            - 1: force single-device execution.
            - N > 1: create a mesh with N devices and use sharded execution.
        mesh: Explicit ``jax.sharding.Mesh`` to use. Overrides
            ``num_devices`` and ``mesh_shape`` when provided.
        mesh_shape: Explicit mesh topology (e.g., ``(4, 2)``).
            Requires ``num_devices`` to be compatible. Ignored when
            ``mesh`` is provided.
        gather: If True, gather the result to a single device instead
            of leaving it sharded across the mesh. Has no effect on
            single-device execution. Default is False (output remains
            sharded).

    Returns:
        Result of the einsum contraction as a JAX array. When executed
        on multiple devices, the result is a sharded JAX array unless
        ``gather=True``.

    Raises:
        ValueError: If notation is invalid or operand shapes are
            inconsistent with the notation.

    Example:
        >>> import einjax
        >>> import jax.numpy as jnp
        >>> A = jnp.ones((3, 4))
        >>> B = jnp.ones((4, 5))
        >>> result = einjax.einsum("ij,jk->ik", A, B)
        >>> result.shape
        (3, 5)
        >>> # Multi-GPU:
        >>> result = einjax.einsum("ij,jk->ik", A, B, num_devices=8)
        >>> # Multi-GPU with gathered output:
        >>> result = einjax.einsum("ij,jk->ik", A, B, num_devices=8, gather=True)
    """
    # Normalize implicit -> explicit output notation
    einsum_string = normalize_notation(einsum_string)

    # Convert operands to JAX arrays
    jax_operands = tuple(jnp.asarray(op) for op in operands)

    # Validate
    shapes = [tuple(op.shape) for op in jax_operands]
    validate_inputs(einsum_string, shapes)

    # Resolve the mesh: explicit arg > context manager > auto-create
    active_mesh = mesh
    if active_mesh is None:
        active_mesh = _get_context_mesh()

    if active_mesh is None and num_devices != 1:
        # Auto-detect device count if not specified
        if num_devices is None:
            from .config import get_config
            detected = get_config().num_devices
        else:
            detected = num_devices

        # Only create a mesh for multi-device
        if detected > 1:
            if mesh_shape is not None:
                axis_names = tuple(
                    chr(ord("x") + i) for i in range(len(mesh_shape))
                )
                active_mesh = create_mesh(
                    axis_names=axis_names, mesh_shape=mesh_shape,
                )
            else:
                active_mesh = create_mesh(
                    axis_names=("x",), num_devices=detected,
                )

    # Single-device fast path
    if active_mesh is None:
        return execute_dense_einsum(einsum_string, *jax_operands)

    # Multi-device: derive partition specs and dispatch to sharded einsum
    in_specs, out_spec = _derive_sharding_specs(einsum_string, shapes, active_mesh)

    result = execute_sharded_einsum(
        einsum_string, *jax_operands,
        mesh=active_mesh, in_specs=in_specs, out_spec=out_spec,
    )

    if gather:
        result = jax.device_get(result)
        result = jnp.asarray(result)

    return result


def _derive_sharding_specs(
    formula: str,
    shapes: list[tuple[int, ...]],
    mesh: jax.sharding.Mesh,
) -> tuple[list[tuple[str | None, ...]], tuple[str | None, ...]]:
    """Derive input and output sharding specs for an einsum on a 1-D mesh.

    Picks a single free (output) index label to shard across the first
    mesh axis.  Every operand dimension carrying that label is sharded;
    all other dimensions are replicated.  This avoids conflicting
    sharding constraints that would force expensive all-gather / reduce
    collectives inside GSPMD.

    Args:
        formula: Einsum formula (e.g., ``"ij,jk->ik"``).
        shapes: List of operand shapes.
        mesh: JAX device mesh.

    Returns:
        ``(in_specs, out_spec)`` — one PartitionSpec tuple per input
        and one for the output.
    """
    first_axis = mesh.axis_names[0]
    axis_size = mesh.shape[first_axis]

    input_part, output_labels = formula.split("->")
    input_labels_list = input_part.split(",")

    # Build label -> dimension size mapping
    label_sizes: dict[str, int] = {}
    for subscripts, shape in zip(input_labels_list, shapes):
        for label, size in zip(subscripts, shape):
            label_sizes[label] = size

    # Pick the best output label to shard: first output label whose
    # size is divisible by the mesh axis size.
    shard_label: str | None = None
    for label in output_labels:
        size = label_sizes.get(label, 0)
        if size >= axis_size and size % axis_size == 0:
            shard_label = label
            break

    # Build per-operand specs
    in_specs: list[tuple[str | None, ...]] = []
    for subscripts, shape in zip(input_labels_list, shapes):
        spec = tuple(
            first_axis if label == shard_label else None
            for label in subscripts
        )
        in_specs.append(spec)

    # Build output spec
    out_spec = tuple(
        first_axis if label == shard_label else None
        for label in output_labels
    )

    return in_specs, out_spec


def _auto_shard_spec(
    shape: tuple[int, ...],
    mesh: jax.sharding.Mesh,
) -> tuple[str | None, ...]:
    """Derive a sharding spec for a tensor, sharding the first divisible dim.

    Shards the first dimension whose size is divisible by the number
    of devices along the first mesh axis. All other dimensions are
    replicated (None).

    Args:
        shape: Tensor shape.
        mesh: JAX device mesh.

    Returns:
        Tuple suitable for PartitionSpec construction.
    """
    first_axis = mesh.axis_names[0]
    axis_size = mesh.shape[first_axis]
    spec: list[str | None] = []
    sharded = False
    for size in shape:
        if not sharded and size >= axis_size and size % axis_size == 0:
            spec.append(first_axis)
            sharded = True
        else:
            spec.append(None)
    return tuple(spec)


def _derive_output_spec(
    formula: str,
    in_specs: list[tuple[str | None, ...]],
) -> tuple[str | None, ...]:
    """Derive output PartitionSpec from einsum formula and input specs.

    Maps each input index label to its sharding axis, then builds the
    output spec from the output labels.

    Args:
        formula: Einsum formula (e.g., "ij,jk->ik").
        in_specs: PartitionSpec tuples for each input.

    Returns:
        PartitionSpec tuple for the output.
    """
    input_part, output_labels = formula.split("->")
    input_labels_list = input_part.split(",")

    # Build label -> axis mapping (first occurrence wins)
    label_to_axis: dict[str, str | None] = {}
    for subscripts, spec in zip(input_labels_list, in_specs):
        for label, axis in zip(subscripts, spec):
            if label not in label_to_axis:
                label_to_axis[label] = axis

    return tuple(label_to_axis.get(label) for label in output_labels)


@dataclass
class AnalysisResult:
    """Result of analyzing an einsum expression without executing it.

    Provides cost breakdown and tiling information for the operation.
    """

    einsum_string: str
    input_shapes: list[tuple[int, ...]]
    output_shape: tuple[int, ...]
    label_dimensions: dict[str, int]
    contracted_indices: list[str]
    free_indices: list[str]

    def __repr__(self) -> str:
        contracted = ", ".join(self.contracted_indices) or "(none)"
        free = ", ".join(self.free_indices) or "(none)"
        return (
            f"AnalysisResult(\n"
            f"  notation='{self.einsum_string}',\n"
            f"  input_shapes={self.input_shapes},\n"
            f"  output_shape={self.output_shape},\n"
            f"  contracted_indices=[{contracted}],\n"
            f"  free_indices=[{free}],\n"
            f"  label_dimensions={self.label_dimensions},\n"
            f")"
        )


def analyze(einsum_string: str, *operands: Any) -> AnalysisResult:
    """Analyze an einsum expression without executing it.

    Returns shape information, contracted/free index classification,
    and label dimension mapping.

    Args:
        einsum_string: Einstein summation notation.
        *operands: Input arrays (used only for shape information).

    Returns:
        AnalysisResult with operation metadata.

    Example:
        >>> import einjax
        >>> import numpy as np
        >>> plan = einjax.analyze("ij,jk->ik", np.zeros((3,4)), np.zeros((4,5)))
        >>> plan.output_shape
        (3, 5)
        >>> plan.contracted_indices
        ['j']
    """
    einsum_string = normalize_notation(einsum_string)

    shapes = [tuple(np.asarray(op).shape) for op in operands]
    validate_inputs(einsum_string, shapes)

    label_dims = get_label_dimensions(einsum_string, shapes)

    input_part, output_part = einsum_string.split("->")
    input_labels = set(input_part.replace(",", ""))
    output_labels = set(output_part)

    contracted = sorted(input_labels - output_labels)
    free = sorted(output_labels)

    output_shape = tuple(label_dims[label] for label in output_part)

    return AnalysisResult(
        einsum_string=einsum_string,
        input_shapes=shapes,
        output_shape=output_shape,
        label_dimensions=label_dims,
        contracted_indices=contracted,
        free_indices=free,
    )
