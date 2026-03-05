"""
Sparsity statistics for EinJAX tensors.

Provides T(U) and V(l, U) statistics from the paper:
- T(U): Number of non-empty tuples (tiles) under tiling scheme U
- V(l, U): Number of distinct values for label l under tiling U

These statistics drive the cost model: sparse tensors with fewer
non-empty tiles have lower communication and computation costs.

Ported from einsql/einsql.py sparsity metrics (lines 695-722),
replacing torch with numpy.
"""

from __future__ import annotations

from math import prod
from typing import Any

import numpy as np

from einjax.core.types import TilingScheme


def compute_sparsity_stats_coo(
    indices: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, ...],
    tile_shapes: list[tuple[int, ...]],
) -> dict[tuple[int, ...], tuple[int, tuple[int, ...]]]:
    """Compute T(U) and V(l,U) for COO-format data across tiling schemes.

    For each tiling scheme U:
    - T(U) = number of non-empty tiles (tile coordinates with at least one
      non-zero entry)
    - V(l, U) = number of distinct tile coordinate values along dimension l

    Args:
        indices: int32 array of shape (nnz, ndim) — non-zero coordinates.
        values: float64 array of shape (nnz,) — non-zero values.
        shape: Full logical tensor shape.
        tile_shapes: List of tile shape tuples to evaluate.

    Returns:
        Dict mapping tile_shape → (num_tuples, value_count) where:
        - num_tuples = T(U) — number of non-empty tiles
        - value_count = tuple of V(l,U) for each dimension l
    """
    ndim = len(shape)
    tuple_counter: dict[tuple[int, ...], list[set[Any]]] = {
        ts: [set() for _ in range(ndim + 1)]
        for ts in tile_shapes
    }

    for nz_idx in range(len(values)):
        if values[nz_idx] == 0:
            continue

        coords = tuple(int(indices[nz_idx, d]) for d in range(ndim))

        for tile_shape, sets in tuple_counter.items():
            tile_coord = tuple(
                coords[d] // tile_shape[d] for d in range(ndim)
            )
            for d in range(ndim):
                sets[d].add(tile_coord[d])
            sets[-1].add(tile_coord)

    result = {}
    for tile_shape in tile_shapes:
        sets = tuple_counter[tile_shape]
        num_tuples = len(sets[-1])
        value_count = tuple(len(sets[d]) for d in range(ndim))
        result[tile_shape] = (num_tuples, value_count)

    return result


def compute_sparsity_stats_dense(
    data: np.ndarray,
    tile_shapes: list[tuple[int, ...]],
) -> dict[tuple[int, ...], tuple[int, tuple[int, ...]]]:
    """Compute T(U) and V(l,U) for dense array data across tiling schemes.

    Traverses all elements, counting which tile coordinates contain
    non-zero values. Same algorithm as DenseTensor._compute_sparsity_metrics.

    Args:
        data: Dense numpy array.
        tile_shapes: List of tile shape tuples to evaluate.

    Returns:
        Dict mapping tile_shape → (num_tuples, value_count).
    """
    shape = tuple(data.shape)
    ndim = len(shape)
    tuple_counter: dict[tuple[int, ...], list[set[Any]]] = {
        ts: [set() for _ in range(ndim + 1)]
        for ts in tile_shapes
    }

    # Extract non-zeros and compute stats via COO path
    nz_indices_raw = np.nonzero(data)
    if len(nz_indices_raw[0]) == 0:
        return {ts: (0, tuple(0 for _ in range(ndim))) for ts in tile_shapes}

    indices = np.stack(nz_indices_raw, axis=1).astype(np.int32)
    values = data[nz_indices_raw]

    return compute_sparsity_stats_coo(indices, values, shape, tile_shapes)


def sparsity_ratio(
    num_tuples: int,
    shape: tuple[int, ...],
    tile_shape: tuple[int, ...],
) -> float:
    """Compute the sparsity ratio T(U) / max_tuples.

    A ratio of 1.0 means every tile is non-empty (effectively dense).
    A ratio near 0.0 means the tensor is highly sparse under this tiling.

    Args:
        num_tuples: T(U) — actual non-empty tiles.
        shape: Full tensor shape.
        tile_shape: Tile dimensions.

    Returns:
        Fraction of tiles that are non-empty, in [0.0, 1.0].
    """
    max_tuples = prod(s // ts for s, ts in zip(shape, tile_shape))
    if max_tuples == 0:
        return 0.0
    return num_tuples / max_tuples


def update_scheme_sparsity(
    scheme: TilingScheme,
    num_tuples: int,
    value_count: tuple[int, ...],
) -> None:
    """Update a TilingScheme with computed sparsity statistics.

    Args:
        scheme: TilingScheme to update.
        num_tuples: T(U) for this scheme.
        value_count: V(l,U) tuple for this scheme.
    """
    scheme.num_tuples = num_tuples
    scheme.value_count = value_count
