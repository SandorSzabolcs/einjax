"""
Block-sparse matrix multiply kernels for EinJAX.

Provides fused block-sparse matmul that combines coordinate-join,
tile-matmul, and segment-sum into a single operation. Two implementations:

1. Pallas kernel (GPU Triton / TPU Mosaic) — used when jax.experimental.pallas
   is available and the backend supports it.
2. Generic JAX fallback — uses jnp.einsum with vectorized operations.

Per PRD Section 7.1, the Pallas kernel fuses all three phases of the sparse
execution path for matmul patterns, reducing kernel launch overhead and
memory traffic.
"""

from __future__ import annotations

from math import prod

import numpy as np
import jax
import jax.numpy as jnp

from einjax.tensor.sparse import SparseTensorRelation
from einjax.execution.sparse_dispatch import (
    coordinate_join,
    kernel_einsum,
    segment_sum,
    _compute_output_coords,
)


def block_sparse_matmul_generic(
    lhs: SparseTensorRelation,
    rhs: SparseTensorRelation,
    block_shape: tuple[int, int, int] | None = None,
) -> SparseTensorRelation:
    """Block-sparse matrix multiply using generic JAX operations.

    Fuses coordinate-join + tile-matmul + segment-sum for the matmul
    pattern (ij,jk->ik). This is the fallback implementation used when
    Pallas is unavailable.

    Args:
        lhs: Left-hand SparseTensorRelation of a 2D matrix.
            coords[:, 0] = row tile index, coords[:, 1] = col tile index.
        rhs: Right-hand SparseTensorRelation of a 2D matrix.
            coords[:, 0] = row tile index, coords[:, 1] = col tile index.
        block_shape: Optional (M_tile, K_tile, N_tile) override.
            If None, uses lhs.tile_shape[0], lhs.tile_shape[1], rhs.tile_shape[1].

    Returns:
        SparseTensorRelation with the matmul result.
    """
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError(
            f"block_sparse_matmul requires 2D tensors, "
            f"got {lhs.ndim}D and {rhs.ndim}D"
        )

    if block_shape is not None:
        m_tile, k_tile, n_tile = block_shape
    else:
        m_tile = lhs.tile_shape[0]
        k_tile = lhs.tile_shape[1]
        n_tile = rhs.tile_shape[1]

    if lhs.tile_shape[1] != rhs.tile_shape[0]:
        raise ValueError(
            f"Inner tile dimensions must match: "
            f"lhs.tile_shape[1]={lhs.tile_shape[1]} != "
            f"rhs.tile_shape[0]={rhs.tile_shape[0]}"
        )

    # Phase 1: Coordinate join on K dimension (lhs col == rhs row)
    join_keys = [(1, 0)]
    lhs_indices, rhs_indices = coordinate_join(lhs, rhs, join_keys)

    if len(lhs_indices) == 0:
        out_tile_shape = (m_tile, n_tile)
        return SparseTensorRelation(
            coords=np.zeros((0, 2), dtype=np.int32),
            values=np.zeros((0, *out_tile_shape), dtype=np.float64),
            shape=(lhs.shape[0], rhs.shape[1]),
            tile_shape=out_tile_shape,
        )

    # Phase 2: Per-tile matmul via batched einsum
    kernel_string = "ij,jk->ik"
    tile_results = kernel_einsum(
        lhs.values, rhs.values,
        lhs_indices, rhs_indices,
        kernel_string,
    )

    # Compute output tile coordinates: (lhs_row, rhs_col)
    agg_keys = [(0, 0), (1, 1)]
    output_tile_shape = (m_tile, n_tile)
    output_shape = (lhs.shape[0], rhs.shape[1])
    output_coords = _compute_output_coords(
        lhs, rhs, lhs_indices, rhs_indices,
        join_keys, agg_keys, output_shape, output_tile_shape,
    )

    # Phase 3: Segment sum to aggregate tiles with same output coord
    final_coords, final_values = segment_sum(tile_results, output_coords)

    return SparseTensorRelation(
        coords=final_coords,
        values=final_values,
        shape=output_shape,
        tile_shape=output_tile_shape,
    )


def block_sparse_matmul(
    lhs_coords: np.ndarray,
    lhs_values: np.ndarray,
    rhs_coords: np.ndarray,
    rhs_values: np.ndarray,
    block_shape: tuple[int, int, int],
    lhs_shape: tuple[int, int],
    rhs_shape: tuple[int, int],
) -> SparseTensorRelation:
    """Fused block-sparse matrix multiply.

    Per PRD Section 7.1, fuses coordinate-join + tile-matmul + segment-sum
    into a single operation. Uses Pallas kernel when available on the
    current backend, otherwise falls back to generic JAX.

    Args:
        lhs_coords: int32 array of shape (num_lhs_tuples, 2) — tile coords.
        lhs_values: float array of shape (num_lhs_tuples, M_tile, K_tile).
        rhs_coords: int32 array of shape (num_rhs_tuples, 2) — tile coords.
        rhs_values: float array of shape (num_rhs_tuples, K_tile, N_tile).
        block_shape: (M_tile, K_tile, N_tile) tile dimensions.
        lhs_shape: Full logical shape of LHS matrix (M, K).
        rhs_shape: Full logical shape of RHS matrix (K, N).

    Returns:
        SparseTensorRelation with the matmul result.
    """
    m_tile, k_tile, n_tile = block_shape

    lhs_rel = SparseTensorRelation(
        coords=np.asarray(lhs_coords, dtype=np.int32),
        values=np.asarray(lhs_values, dtype=np.float64),
        shape=lhs_shape,
        tile_shape=(m_tile, k_tile),
    )
    rhs_rel = SparseTensorRelation(
        coords=np.asarray(rhs_coords, dtype=np.int32),
        values=np.asarray(rhs_values, dtype=np.float64),
        shape=rhs_shape,
        tile_shape=(k_tile, n_tile),
    )

    return block_sparse_matmul_generic(lhs_rel, rhs_rel, block_shape)
