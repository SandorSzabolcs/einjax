"""
Custom VJP rules for sparse tensor-relational einsum operations.

Per PRD Section 8.2, the sparse execution path requires custom_vjp
because the sparse coordinate structure is non-differentiable.
Coordinates (coords) are treated as static structure; gradients
flow only through the values arrays.

The backward pass reuses the coordinate-join from the forward pass
(cached in residuals) to avoid redundant matching.

Dense operations (Section 8.1) need no custom gradient rules since
jnp.einsum and shard_map are already differentiable through JAX's
standard autodiff.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from einjax.execution.sparse_dispatch import (
    coordinate_join,
    _add_batch_dim,
    _infer_output_tile_shape,
)
from einjax.tensor.sparse import SparseTensorRelation


def _reverse_einsum_string(kernel_string: str) -> tuple[str, str]:
    """Derive backward einsum strings for LHS and RHS gradients.

    For a forward einsum "ij,jk->ik":
    - grad_lhs: contract grad (ik) with rhs (jk) over k → ij
      → "ik,jk->ij"  (swap output with rhs input, contract over rhs-only labels)
    - grad_rhs: contract lhs (ij) with grad (ik) over i → jk
      → "ij,ik->jk"  (swap output with lhs input, contract over lhs-only labels)

    Args:
        kernel_string: Forward einsum notation (e.g., "ij,jk->ik").

    Returns:
        (grad_lhs_string, grad_rhs_string) einsum notation strings.
    """
    if "->" not in kernel_string:
        raise ValueError(f"kernel_string must contain '->': {kernel_string}")

    inputs_str, output_str = kernel_string.split("->")
    input_terms = inputs_str.split(",")
    lhs_labels = input_terms[0]
    rhs_labels = input_terms[1]

    # grad_lhs: einsum(grad, rhs) -> lhs_labels
    # The grad has output_str labels, rhs has rhs_labels
    grad_lhs_string = f"{output_str},{rhs_labels}->{lhs_labels}"

    # grad_rhs: einsum(lhs, grad) -> rhs_labels
    # The lhs has lhs_labels, grad has output_str labels
    grad_rhs_string = f"{lhs_labels},{output_str}->{rhs_labels}"

    return grad_lhs_string, grad_rhs_string


def _compute_output_coords_fwd(
    lhs: SparseTensorRelation,
    rhs: SparseTensorRelation,
    lhs_indices: np.ndarray,
    rhs_indices: np.ndarray,
    agg_keys: list[tuple[int, int]],
) -> np.ndarray:
    """Compute output coordinates from matched input coordinates.

    For each matched pair, collects the output dimension coordinates
    from the appropriate input tensor based on agg_keys.
    """
    num_matches = len(lhs_indices)
    output_ndim = len(agg_keys)

    if num_matches == 0:
        return np.zeros((0, output_ndim), dtype=np.int32)

    output_coords = np.zeros((num_matches, output_ndim), dtype=np.int32)
    for out_d, (src_tensor, src_dim) in enumerate(agg_keys):
        if src_tensor == 0:
            output_coords[:, out_d] = lhs.coords[lhs_indices, src_dim]
        else:
            output_coords[:, out_d] = rhs.coords[rhs_indices, src_dim]

    return output_coords


def _segment_sum_fwd(
    values: np.ndarray,
    output_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate values by output coordinate, returning unique coords and sums."""
    if len(values) == 0:
        ndim = output_coords.shape[1] if output_coords.ndim > 1 else 0
        tile_shape = values.shape[1:] if values.ndim > 1 else ()
        return (
            np.zeros((0, ndim), dtype=np.int32),
            np.zeros((0, *tile_shape), dtype=values.dtype),
        )

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


def sparse_einsum(
    lhs: SparseTensorRelation,
    rhs: SparseTensorRelation,
    join_keys: list[tuple[int, int]],
    kernel_string: str,
    agg_keys: list[tuple[int, int]],
    output_shape: tuple[int, ...] | None = None,
    output_tile_shape: tuple[int, ...] | None = None,
) -> SparseTensorRelation:
    """Differentiable sparse tensor-relational contraction.

    Forward pass: executes the three-phase sparse pipeline
    (coordinate join → kernel einsum → segment sum).

    Backward pass: gradients flow through values only. The sparsity
    pattern (coords) is treated as static structure. Gradient w.r.t.
    lhs.values is computed by contracting the output gradient with
    rhs.values, and vice versa.

    Per PRD Section 8.2, works with jax.grad, jax.value_and_grad,
    and higher-order derivatives.

    Args:
        lhs: Left-hand SparseTensorRelation.
        rhs: Right-hand SparseTensorRelation.
        join_keys: List of (lhs_dim, rhs_dim) join key pairs.
        kernel_string: Einsum notation for per-tile computation.
        agg_keys: List of (source_tensor, dim) pairs for output dims.
        output_shape: Full output shape (inferred if None).
        output_tile_shape: Output tile shape (inferred if None).

    Returns:
        SparseTensorRelation with contracted result.
    """
    if output_tile_shape is None:
        output_tile_shape = _infer_output_tile_shape(
            kernel_string, lhs.tile_shape, rhs.tile_shape
        )

    if output_shape is None:
        output_shape = tuple(
            lhs.shape[dim] if src == 0 else rhs.shape[dim]
            for src, dim in agg_keys
        )

    # Phase 1: Coordinate join (non-differentiable — structure only)
    lhs_indices, rhs_indices = coordinate_join(lhs, rhs, join_keys)

    if len(lhs_indices) == 0:
        coords = np.zeros((0, len(output_tile_shape)), dtype=np.int32)
        values = np.zeros((0, *output_tile_shape), dtype=lhs.values.dtype)
        return SparseTensorRelation(
            coords=coords, values=values,
            shape=output_shape, tile_shape=output_tile_shape,
        )

    # Phase 2: Kernel — differentiable through jnp.einsum
    matched_lhs = jnp.array(lhs.values[lhs_indices])
    matched_rhs = jnp.array(rhs.values[rhs_indices])
    batch_kernel = _add_batch_dim(kernel_string, "z")
    tile_results = jnp.einsum(batch_kernel, matched_lhs, matched_rhs)

    # Compute output tile coordinates
    output_coords = _compute_output_coords_fwd(
        lhs, rhs, lhs_indices, rhs_indices, agg_keys,
    )

    # Phase 3: Aggregate (segment sum)
    final_coords, final_values = _segment_sum_fwd(
        np.asarray(tile_results), output_coords
    )

    return SparseTensorRelation(
        coords=final_coords,
        values=final_values,
        shape=output_shape,
        tile_shape=output_tile_shape,
    )


def sparse_einsum_raw(
    lhs_values: jax.Array,
    rhs_values: jax.Array,
    lhs_coords: np.ndarray,
    rhs_coords: np.ndarray,
    lhs_indices: np.ndarray,
    rhs_indices: np.ndarray,
    kernel_string: str,
) -> jax.Array:
    """Differentiable sparse tile-pair contraction via custom_vjp.

    This is the core differentiable primitive. It takes pre-matched
    tile values and indices and computes the batched einsum. The
    coordinate join (matching) is assumed to have been done already
    and is treated as static.

    Uses jax.custom_vjp so that jax.grad flows through tile values
    while treating coords and match indices as non-differentiable.
    kernel_string and index arrays are closed over (not JAX-traced).

    Args:
        lhs_values: JAX array of LHS tile values (num_lhs, *lhs_tile_shape).
        rhs_values: JAX array of RHS tile values (num_rhs, *rhs_tile_shape).
        lhs_coords: NumPy int32 array of LHS tile coordinates.
        rhs_coords: NumPy int32 array of RHS tile coordinates.
        lhs_indices: NumPy int64 array of matched LHS indices.
        rhs_indices: NumPy int64 array of matched RHS indices.
        kernel_string: Einsum notation (e.g., "ij,jk->ik").

    Returns:
        JAX array of shape (num_matches, *output_tile_shape).
    """
    # kernel_string is a Python str (not a valid JAX type), so it must be
    # closed over rather than passed as an argument to custom_vjp.
    @jax.custom_vjp
    def _impl(lhs_values: jax.Array, rhs_values: jax.Array) -> jax.Array:
        if len(lhs_indices) == 0:
            output_tile_shape = _infer_output_tile_shape(
                kernel_string, lhs_values.shape[1:], rhs_values.shape[1:]
            )
            return jnp.zeros((0, *output_tile_shape), dtype=lhs_values.dtype)
        matched_lhs = lhs_values[lhs_indices]
        matched_rhs = rhs_values[rhs_indices]
        batch_kernel = _add_batch_dim(kernel_string, "z")
        return jnp.einsum(batch_kernel, matched_lhs, matched_rhs)

    def _fwd(lhs_values: jax.Array, rhs_values: jax.Array):
        result = _impl(lhs_values, rhs_values)
        return result, (lhs_values, rhs_values)

    def _bwd(residuals, g: jax.Array):
        lhs_values, rhs_values = residuals
        grad_lhs_string, grad_rhs_string = _reverse_einsum_string(kernel_string)
        num_matches = len(lhs_indices)

        grad_lhs_full = jnp.zeros_like(lhs_values)
        grad_rhs_full = jnp.zeros_like(rhs_values)

        if num_matches > 0:
            matched_rhs = rhs_values[rhs_indices]
            matched_lhs = lhs_values[lhs_indices]

            batch_grad_lhs_str = _add_batch_dim(grad_lhs_string, "z")
            batch_grad_rhs_str = _add_batch_dim(grad_rhs_string, "z")

            per_pair_grad_lhs = jnp.einsum(batch_grad_lhs_str, g, matched_rhs)
            per_pair_grad_rhs = jnp.einsum(batch_grad_rhs_str, matched_lhs, g)

            grad_lhs_full = grad_lhs_full.at[lhs_indices].add(per_pair_grad_lhs)
            grad_rhs_full = grad_rhs_full.at[rhs_indices].add(per_pair_grad_rhs)

        return grad_lhs_full, grad_rhs_full

    _impl.defvjp(_fwd, _bwd)
    return _impl(lhs_values, rhs_values)
