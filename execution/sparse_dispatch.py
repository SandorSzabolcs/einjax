"""
Sparse tensor-relational execution path for EinJAX.

Implements the three-phase sparse execution from the paper (Section 5):
1. Coordinate join: match tile coords on join_keys
2. Kernel: vmap(jnp.einsum) over matched (lhs_tile, rhs_tile) pairs
3. Aggregate: segment_sum over agg_keys to combine tiles with same output coord

Per PRD Section 6.3, this replaces the SQL CTE path from einsql with
direct NumPy/JAX computation on SparseTensorRelation data.

Per PRD Section 4.6, execute_sharded_sparse() adds multi-device support:
- Phase 1 (coordinate join) runs on CPU (small relational metadata)
- Phase 2 (kernel einsum) is partitioned across devices via jax.jit + sharding
- Phase 3 (segment sum) runs per-device then merges cross-device results
"""

from __future__ import annotations

from math import prod

import jax
import numpy as np
import jax.numpy as jnp

from einjax.tensor.sparse import SparseTensorRelation


def coordinate_join(
    lhs: SparseTensorRelation,
    rhs: SparseTensorRelation,
    join_keys: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Match tile coordinates on join key dimensions.

    For each pair of join_keys (lhs_dim, rhs_dim), matches tiles whose
    coordinate in lhs_dim equals the coordinate in rhs_dim. Returns
    index arrays into lhs and rhs for the matched pairs.

    Uses a hash join: builds a dict from LHS join-key values to LHS
    row indices, then probes with RHS join-key values.

    Args:
        lhs: Left-hand SparseTensorRelation.
        rhs: Right-hand SparseTensorRelation.
        join_keys: List of (lhs_dim, rhs_dim) pairs that must match.

    Returns:
        (lhs_indices, rhs_indices) — int arrays of the same length,
        indexing into lhs.coords/values and rhs.coords/values for
        each matched tile pair.
    """
    if not join_keys:
        # No join keys = cross product
        n_lhs = lhs.num_tuples
        n_rhs = rhs.num_tuples
        lhs_idx = np.repeat(np.arange(n_lhs), n_rhs)
        rhs_idx = np.tile(np.arange(n_rhs), n_lhs)
        return lhs_idx, rhs_idx

    # Extract join-key columns from each side
    lhs_dims = [k[0] for k in join_keys]
    rhs_dims = [k[1] for k in join_keys]

    # Build hash table on LHS join keys
    lhs_key_to_indices: dict[tuple[int, ...], list[int]] = {}
    for i in range(lhs.num_tuples):
        key = tuple(int(lhs.coords[i, d]) for d in lhs_dims)
        lhs_key_to_indices.setdefault(key, []).append(i)

    # Probe with RHS join keys
    lhs_matches: list[int] = []
    rhs_matches: list[int] = []

    for j in range(rhs.num_tuples):
        key = tuple(int(rhs.coords[j, d]) for d in rhs_dims)
        if key in lhs_key_to_indices:
            for i in lhs_key_to_indices[key]:
                lhs_matches.append(i)
                rhs_matches.append(j)

    return np.array(lhs_matches, dtype=np.int64), np.array(rhs_matches, dtype=np.int64)


def kernel_einsum(
    lhs_values: np.ndarray,
    rhs_values: np.ndarray,
    lhs_indices: np.ndarray,
    rhs_indices: np.ndarray,
    kernel_string: str,
) -> np.ndarray:
    """Execute vmap(jnp.einsum) over matched tile pairs.

    For each matched (lhs_tile, rhs_tile) pair, computes
    jnp.einsum(kernel_string, lhs_tile, rhs_tile).

    Args:
        lhs_values: LHS values array of shape (num_lhs_tuples, *lhs_tile_shape).
        rhs_values: RHS values array of shape (num_rhs_tuples, *rhs_tile_shape).
        lhs_indices: Indices into lhs_values for matched pairs.
        rhs_indices: Indices into rhs_values for matched pairs.
        kernel_string: Einsum notation for the per-tile computation
            (e.g., "ij,jk->ik" for matmul tiles).

    Returns:
        Array of shape (num_matches, *output_tile_shape) with per-pair results.
    """
    if len(lhs_indices) == 0:
        # Compute output tile shape from kernel_string with zero-size inputs
        # Parse the output shape from the kernel string
        output_spec = kernel_string.split("->")[1] if "->" in kernel_string else ""
        output_tile_shape = _infer_output_tile_shape(
            kernel_string, lhs_values.shape[1:], rhs_values.shape[1:]
        )
        return np.zeros((0, *output_tile_shape), dtype=lhs_values.dtype)

    # Gather matched tiles
    matched_lhs = lhs_values[lhs_indices]  # (num_matches, *lhs_tile_shape)
    matched_rhs = rhs_values[rhs_indices]  # (num_matches, *rhs_tile_shape)

    # Add batch dimension label to kernel_string for vectorized computation
    # e.g., "ij,jk->ik" becomes "zij,zjk->zik"
    batch_kernel = _add_batch_dim(kernel_string, "z")

    result = jnp.einsum(batch_kernel, jnp.array(matched_lhs), jnp.array(matched_rhs))
    return np.asarray(result)


def segment_sum(
    values: np.ndarray,
    output_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate values by output tile coordinate using segment sum.

    Groups tiles with the same output coordinate and sums their values.

    Args:
        values: Array of shape (num_matches, *output_tile_shape).
        output_coords: Array of shape (num_matches, output_ndim) with
            the output tile coordinate for each matched pair.

    Returns:
        (unique_coords, aggregated_values) where:
        - unique_coords: (num_output_tuples, output_ndim)
        - aggregated_values: (num_output_tuples, *output_tile_shape)
    """
    if len(values) == 0:
        ndim = output_coords.shape[1] if output_coords.ndim > 1 else 0
        tile_shape = values.shape[1:] if values.ndim > 1 else ()
        return (
            np.zeros((0, ndim), dtype=np.int32),
            np.zeros((0, *tile_shape), dtype=values.dtype),
        )

    # Find unique output coordinates and group indices
    coord_keys: dict[tuple[int, ...], list[int]] = {}
    for i in range(len(output_coords)):
        key = tuple(int(output_coords[i, d]) for d in range(output_coords.shape[1]))
        coord_keys.setdefault(key, []).append(i)

    sorted_keys = sorted(coord_keys.keys())
    unique_coords = np.array(sorted_keys, dtype=np.int32)

    tile_shape = values.shape[1:]
    aggregated = np.zeros((len(sorted_keys), *tile_shape), dtype=values.dtype)

    for out_idx, key in enumerate(sorted_keys):
        for src_idx in coord_keys[key]:
            aggregated[out_idx] += values[src_idx]

    return unique_coords, aggregated


def _compute_output_coords(
    lhs: SparseTensorRelation,
    rhs: SparseTensorRelation,
    lhs_indices: np.ndarray,
    rhs_indices: np.ndarray,
    join_keys: list[tuple[int, int]],
    agg_keys: list[int],
    output_shape: tuple[int, ...],
    output_tile_shape: tuple[int, ...],
) -> np.ndarray:
    """Derive output tile coordinates from matched input coordinates.

    For each matched pair, computes the output tile coordinate by collecting
    the non-contracted (output) dimension coordinates from the LHS and RHS.

    The agg_keys parameter specifies which output dimensions come from which
    input. Positive values index into LHS coords, negative values (encoded
    as dim + lhs.ndim) index into RHS coords.

    Args:
        lhs: Left-hand SparseTensorRelation.
        rhs: Right-hand SparseTensorRelation.
        lhs_indices: Match indices into LHS.
        rhs_indices: Match indices into RHS.
        join_keys: List of (lhs_dim, rhs_dim) join key pairs.
        agg_keys: List of (source, dim) pairs where source 0=lhs, 1=rhs.
        output_shape: Full logical output shape.
        output_tile_shape: Output tile dimensions.

    Returns:
        Array of shape (num_matches, output_ndim) with output tile coords.
    """
    num_matches = len(lhs_indices)
    output_ndim = len(output_tile_shape)

    if num_matches == 0:
        return np.zeros((0, output_ndim), dtype=np.int32)

    output_coords = np.zeros((num_matches, output_ndim), dtype=np.int32)

    # agg_keys encodes output dimension sources:
    # Each entry is (source_tensor_index, dim_in_that_tensor)
    # source_tensor_index: 0 = LHS, 1 = RHS
    for out_d, (src_tensor, src_dim) in enumerate(agg_keys):
        if src_tensor == 0:
            output_coords[:, out_d] = lhs.coords[lhs_indices, src_dim]
        else:
            output_coords[:, out_d] = rhs.coords[rhs_indices, src_dim]

    return output_coords


def execute_sparse(
    lhs: SparseTensorRelation,
    rhs: SparseTensorRelation,
    join_keys: list[tuple[int, int]],
    kernel_string: str,
    agg_keys: list[tuple[int, int]],
    output_shape: tuple[int, ...] | None = None,
    output_tile_shape: tuple[int, ...] | None = None,
) -> SparseTensorRelation:
    """Execute sparse tensor-relational operation.

    Three phases (matching paper Section 5):
    1. Coordinate join: match tile coords on join_keys
       - Hash join on sorted coords
    2. Kernel: vmap(jnp.einsum) over matched (lhs_tile, rhs_tile) pairs
    3. Aggregate: segment_sum over agg_keys to combine tiles with same
       output coord

    Args:
        lhs: Left-hand SparseTensorRelation.
        rhs: Right-hand SparseTensorRelation.
        join_keys: List of (lhs_dim, rhs_dim) pairs identifying
            dimensions that are contracted (joined) between operands.
        kernel_string: Einsum notation for the per-tile computation
            (e.g., "ij,jk->ik").
        agg_keys: List of (source_tensor, dim) pairs identifying
            which input dimensions form the output dimensions.
            source_tensor: 0 for LHS, 1 for RHS.
        output_shape: Full logical output shape. If None, inferred
            from inputs and agg_keys.
        output_tile_shape: Output tile shape. If None, inferred from
            kernel_string and input tile shapes.

    Returns:
        SparseTensorRelation with the contracted/aggregated result.
    """
    # Infer output tile shape from kernel string if not provided
    if output_tile_shape is None:
        output_tile_shape = _infer_output_tile_shape(
            kernel_string, lhs.tile_shape, rhs.tile_shape
        )

    # Infer output shape from inputs and agg_keys if not provided
    if output_shape is None:
        output_shape = tuple(
            lhs.shape[dim] if src == 0 else rhs.shape[dim]
            for src, dim in agg_keys
        )

    # Phase 1: Coordinate join
    lhs_indices, rhs_indices = coordinate_join(lhs, rhs, join_keys)

    # Phase 2: Per-tile kernel computation
    tile_results = kernel_einsum(
        lhs.values, rhs.values, lhs_indices, rhs_indices, kernel_string
    )

    # Compute output tile coordinates for each matched pair
    output_coords = _compute_output_coords(
        lhs, rhs, lhs_indices, rhs_indices,
        join_keys, agg_keys, output_shape, output_tile_shape,
    )

    # Phase 3: Aggregate tiles with same output coordinate
    final_coords, final_values = segment_sum(tile_results, output_coords)

    return SparseTensorRelation(
        coords=final_coords,
        values=final_values,
        shape=output_shape,
        tile_shape=output_tile_shape,
    )


def _partition_matched_pairs(
    num_matches: int,
    num_devices: int,
) -> list[tuple[int, int]]:
    """Partition matched tile pairs across devices for parallel execution.

    Distributes matched pairs as evenly as possible using contiguous slices.
    Each device gets a (start, end) range into the matched-pair arrays.

    Args:
        num_matches: Total number of matched tile pairs.
        num_devices: Number of devices to partition across.

    Returns:
        List of (start, end) index tuples, one per device.
    """
    if num_matches == 0:
        return [(0, 0)] * num_devices

    base_size = num_matches // num_devices
    remainder = num_matches % num_devices
    partitions = []
    offset = 0
    for d in range(num_devices):
        size = base_size + (1 if d < remainder else 0)
        partitions.append((offset, offset + size))
        offset += size
    return partitions


def execute_sharded_sparse(
    lhs: SparseTensorRelation,
    rhs: SparseTensorRelation,
    join_keys: list[tuple[int, int]],
    kernel_string: str,
    agg_keys: list[tuple[int, int]],
    output_shape: tuple[int, ...] | None = None,
    output_tile_shape: tuple[int, ...] | None = None,
    mesh: jax.sharding.Mesh | None = None,
    num_devices: int | None = None,
) -> SparseTensorRelation:
    """Execute sharded sparse tensor-relational operation across multiple devices.

    Multi-device version of execute_sparse(). The three-phase pipeline is
    adapted for multi-GPU execution:

    Phase 1 (Coordinate Join): Runs on CPU — coordinate tables are small
    relational metadata. The full hash join runs single-threaded; the result
    is a set of matched (lhs_idx, rhs_idx) pairs.

    Phase 2 (Kernel Einsum): Embarrassingly parallel. Matched tile pairs
    are partitioned across devices. Each device runs batched jnp.einsum
    on its local subset via jax.jit with sharding constraints. No
    cross-device communication is needed.

    Phase 3 (Segment Sum): Each device produces partial output tiles.
    Results are gathered and merged on CPU via segment_sum over the
    combined output coordinates.

    When no mesh is provided, falls back to single-device execute_sparse().

    Args:
        lhs: Left-hand SparseTensorRelation.
        rhs: Right-hand SparseTensorRelation.
        join_keys: List of (lhs_dim, rhs_dim) pairs for coordinate matching.
        kernel_string: Einsum notation for per-tile computation.
        agg_keys: List of (source_tensor, dim) pairs for output dimensions.
        output_shape: Full logical output shape. If None, inferred.
        output_tile_shape: Output tile shape. If None, inferred.
        mesh: JAX device mesh for sharded execution. If None, falls back
            to single-device.
        num_devices: Number of devices to partition across. If None,
            derived from mesh or defaults to 1.

    Returns:
        SparseTensorRelation with the contracted/aggregated result.
    """
    # Fall back to single-device when no mesh
    if mesh is None:
        return execute_sparse(
            lhs, rhs, join_keys, kernel_string, agg_keys,
            output_shape, output_tile_shape,
        )

    # Determine number of devices
    if num_devices is None:
        num_devices = len(mesh.devices.flat)

    # Fall back to single-device when num_devices=1
    if num_devices <= 1:
        return execute_sparse(
            lhs, rhs, join_keys, kernel_string, agg_keys,
            output_shape, output_tile_shape,
        )

    # Infer shapes
    if output_tile_shape is None:
        output_tile_shape = _infer_output_tile_shape(
            kernel_string, lhs.tile_shape, rhs.tile_shape
        )
    if output_shape is None:
        output_shape = tuple(
            lhs.shape[dim] if src == 0 else rhs.shape[dim]
            for src, dim in agg_keys
        )

    # Phase 1: Coordinate join (CPU — small relational metadata)
    lhs_indices, rhs_indices = coordinate_join(lhs, rhs, join_keys)
    num_matches = len(lhs_indices)

    if num_matches == 0:
        return SparseTensorRelation(
            coords=np.zeros((0, len(output_tile_shape)), dtype=np.int32),
            values=np.zeros((0, *output_tile_shape), dtype=lhs.values.dtype),
            shape=output_shape,
            tile_shape=output_tile_shape,
        )

    # Compute output coordinates (CPU — small relational metadata)
    output_coords = _compute_output_coords(
        lhs, rhs, lhs_indices, rhs_indices,
        join_keys, agg_keys, output_shape, output_tile_shape,
    )

    # Phase 2: Sharded kernel einsum
    # Partition matched pairs across devices
    partitions = _partition_matched_pairs(num_matches, num_devices)

    batch_kernel = _add_batch_dim(kernel_string, "z")

    # Gather all matched tiles once (CPU/numpy)
    all_matched_lhs = lhs.values[lhs_indices]
    all_matched_rhs = rhs.values[rhs_indices]

    # Convert to JAX arrays and shard across devices
    jax_lhs = jnp.array(all_matched_lhs)
    jax_rhs = jnp.array(all_matched_rhs)

    # Pad to make evenly divisible by num_devices for clean sharding
    pad_size = (num_devices - num_matches % num_devices) % num_devices
    if pad_size > 0:
        lhs_pad_shape = (pad_size, *jax_lhs.shape[1:])
        rhs_pad_shape = (pad_size, *jax_rhs.shape[1:])
        jax_lhs = jnp.concatenate([jax_lhs, jnp.zeros(lhs_pad_shape, dtype=jax_lhs.dtype)])
        jax_rhs = jnp.concatenate([jax_rhs, jnp.zeros(rhs_pad_shape, dtype=jax_rhs.dtype)])

    # Shard along the batch (match) dimension across devices
    batch_spec = jax.sharding.PartitionSpec("x", *([None] * (jax_lhs.ndim - 1)))
    sharding = jax.sharding.NamedSharding(mesh, batch_spec)

    jax_lhs = jax.device_put(jax_lhs, sharding)
    jax_rhs = jax.device_put(jax_rhs, sharding)

    # JIT-compile the batched einsum with sharding constraints
    out_ndim = len(output_tile_shape)
    out_spec = jax.sharding.PartitionSpec("x", *([None] * out_ndim))
    out_sharding = jax.sharding.NamedSharding(mesh, out_spec)

    sharded_fn = jax.jit(
        lambda lhs_tiles, rhs_tiles: jnp.einsum(batch_kernel, lhs_tiles, rhs_tiles),
        in_shardings=(sharding, sharding),
        out_shardings=out_sharding,
    )
    tile_results_jax = sharded_fn(jax_lhs, jax_rhs)

    # Gather results back to CPU and strip padding
    tile_results = np.asarray(jax.device_get(tile_results_jax))
    if pad_size > 0:
        tile_results = tile_results[:num_matches]

    # Phase 3: Aggregate tiles with same output coordinate (CPU)
    final_coords, final_values = segment_sum(tile_results, output_coords)

    return SparseTensorRelation(
        coords=final_coords,
        values=final_values,
        shape=output_shape,
        tile_shape=output_tile_shape,
    )


def _add_batch_dim(kernel_string: str, batch_label: str = "z") -> str:
    """Add a batch dimension label to an einsum string.

    Transforms "ij,jk->ik" into "zij,zjk->zik" for vectorized computation
    over matched tile pairs.

    Args:
        kernel_string: Original einsum notation.
        batch_label: Label for the batch dimension.

    Returns:
        Einsum string with batch dimension prepended to all terms.
    """
    if "->" not in kernel_string:
        raise ValueError(f"kernel_string must contain '->': {kernel_string}")

    inputs_str, output_str = kernel_string.split("->")
    input_terms = inputs_str.split(",")

    batched_inputs = ",".join(batch_label + term for term in input_terms)
    batched_output = batch_label + output_str

    return f"{batched_inputs}->{batched_output}"


def _infer_output_tile_shape(
    kernel_string: str,
    lhs_tile_shape: tuple[int, ...],
    rhs_tile_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Infer the output tile shape from the kernel string and input tile shapes.

    Maps each label in the kernel string to a dimension size from the inputs,
    then reads off the output labels to determine the output tile shape.

    Args:
        kernel_string: Einsum notation (e.g., "ij,jk->ik").
        lhs_tile_shape: Shape of LHS tiles (without batch dimension).
        rhs_tile_shape: Shape of RHS tiles (without batch dimension).

    Returns:
        Output tile shape as a tuple of ints.
    """
    if "->" not in kernel_string:
        raise ValueError(f"kernel_string must contain '->': {kernel_string}")

    inputs_str, output_str = kernel_string.split("->")
    input_terms = inputs_str.split(",")

    # Build label → size mapping from input terms
    label_sizes: dict[str, int] = {}

    if len(input_terms) >= 1:
        for d, label in enumerate(input_terms[0]):
            if d < len(lhs_tile_shape):
                label_sizes[label] = lhs_tile_shape[d]

    if len(input_terms) >= 2:
        for d, label in enumerate(input_terms[1]):
            if d < len(rhs_tile_shape):
                label_sizes[label] = rhs_tile_shape[d]

    # Read output shape from output labels
    output_tile_shape = tuple(label_sizes[label] for label in output_str)
    return output_tile_shape
