"""
Coordinate join kernel implementations for EinJAX.

Provides two coordinate join strategies:
1. Hash join — O(n+m) average case, used for small-to-medium inputs.
2. Sort-merge join — O(n log n + m log m), stable for large inputs.

Per PRD Section 7.2, the Pallas version would compile these to Triton
(GPU) or Mosaic (TPU). This module provides the generic JAX/NumPy
implementations that serve as both the fallback and the reference
for correctness testing of future Pallas kernels.
"""

from __future__ import annotations

import numpy as np


def coordinate_join_hash(
    lhs_coords: np.ndarray,
    rhs_coords: np.ndarray,
    join_dims: list[int] | list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Hash-based coordinate join on specified dimensions.

    Builds a hash table on LHS join-key values, then probes with RHS.
    This is the default strategy used by execute_sparse's coordinate_join.

    Args:
        lhs_coords: int32 array of shape (num_lhs, ndim_lhs).
        rhs_coords: int32 array of shape (num_rhs, ndim_rhs).
        join_dims: Either list of int (same dim index on both sides)
            or list of (lhs_dim, rhs_dim) tuples.

    Returns:
        (lhs_indices, rhs_indices) — matched pair index arrays.
    """
    # Normalize join_dims to (lhs_dim, rhs_dim) tuples
    if join_dims and isinstance(join_dims[0], int):
        join_keys: list[tuple[int, int]] = [(d, d) for d in join_dims]
    else:
        join_keys = list(join_dims)  # type: ignore[arg-type]

    n_lhs = lhs_coords.shape[0]
    n_rhs = rhs_coords.shape[0]

    if n_lhs == 0 or n_rhs == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if not join_keys:
        # Cross product
        lhs_idx = np.repeat(np.arange(n_lhs), n_rhs)
        rhs_idx = np.tile(np.arange(n_rhs), n_lhs)
        return lhs_idx, rhs_idx

    lhs_dims = [k[0] for k in join_keys]
    rhs_dims = [k[1] for k in join_keys]

    # Build hash table on LHS
    lhs_table: dict[tuple[int, ...], list[int]] = {}
    for i in range(n_lhs):
        key = tuple(int(lhs_coords[i, d]) for d in lhs_dims)
        lhs_table.setdefault(key, []).append(i)

    # Probe with RHS
    lhs_matches: list[int] = []
    rhs_matches: list[int] = []

    for j in range(n_rhs):
        key = tuple(int(rhs_coords[j, d]) for d in rhs_dims)
        if key in lhs_table:
            for i in lhs_table[key]:
                lhs_matches.append(i)
                rhs_matches.append(j)

    return np.array(lhs_matches, dtype=np.int64), np.array(rhs_matches, dtype=np.int64)


def coordinate_join_sorted(
    lhs_coords: np.ndarray,
    rhs_coords: np.ndarray,
    join_dims: list[int] | list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Sort-merge coordinate join on specified dimensions.

    Sorts both sides by join-key values, then performs a merge-join.
    More predictable performance than hash join for large inputs.

    Per PRD Section 7.2, this is the algorithm that would be implemented
    as a Pallas kernel for GPU/TPU acceleration.

    Args:
        lhs_coords: int32 array of shape (num_lhs, ndim_lhs).
        rhs_coords: int32 array of shape (num_rhs, ndim_rhs).
        join_dims: Either list of int (same dim index on both sides)
            or list of (lhs_dim, rhs_dim) tuples.

    Returns:
        (lhs_indices, rhs_indices) — matched pair index arrays.
    """
    # Normalize join_dims
    if join_dims and isinstance(join_dims[0], int):
        join_keys: list[tuple[int, int]] = [(d, d) for d in join_dims]
    else:
        join_keys = list(join_dims)  # type: ignore[arg-type]

    n_lhs = lhs_coords.shape[0]
    n_rhs = rhs_coords.shape[0]

    if n_lhs == 0 or n_rhs == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if not join_keys:
        # Cross product
        lhs_idx = np.repeat(np.arange(n_lhs), n_rhs)
        rhs_idx = np.tile(np.arange(n_rhs), n_lhs)
        return lhs_idx, rhs_idx

    lhs_dims = [k[0] for k in join_keys]
    rhs_dims = [k[1] for k in join_keys]

    # Extract join-key columns
    lhs_keys = np.array(
        [tuple(int(lhs_coords[i, d]) for d in lhs_dims) for i in range(n_lhs)]
    )
    rhs_keys = np.array(
        [tuple(int(rhs_coords[j, d]) for d in rhs_dims) for j in range(n_rhs)]
    )

    # Sort both sides by join keys (lexicographic)
    lhs_order = np.lexsort(lhs_keys.T[::-1]) if lhs_keys.ndim > 1 else np.argsort(lhs_keys.ravel())
    rhs_order = np.lexsort(rhs_keys.T[::-1]) if rhs_keys.ndim > 1 else np.argsort(rhs_keys.ravel())

    sorted_lhs_keys = lhs_keys[lhs_order]
    sorted_rhs_keys = rhs_keys[rhs_order]

    # Merge join
    lhs_matches: list[int] = []
    rhs_matches: list[int] = []

    i = 0
    j = 0

    while i < n_lhs and j < n_rhs:
        lk = tuple(sorted_lhs_keys[i]) if sorted_lhs_keys.ndim > 1 else (int(sorted_lhs_keys[i]),)
        rk = tuple(sorted_rhs_keys[j]) if sorted_rhs_keys.ndim > 1 else (int(sorted_rhs_keys[j]),)

        if lk < rk:
            i += 1
        elif lk > rk:
            j += 1
        else:
            # Find the range of equal keys on both sides
            i_start = i
            while i < n_lhs:
                ik = tuple(sorted_lhs_keys[i]) if sorted_lhs_keys.ndim > 1 else (int(sorted_lhs_keys[i]),)
                if ik != lk:
                    break
                i += 1

            j_start = j
            while j < n_rhs:
                jk = tuple(sorted_rhs_keys[j]) if sorted_rhs_keys.ndim > 1 else (int(sorted_rhs_keys[j]),)
                if jk != rk:
                    break
                j += 1

            # Cross product for the matching group
            for ii in range(i_start, i):
                for jj in range(j_start, j):
                    lhs_matches.append(int(lhs_order[ii]))
                    rhs_matches.append(int(rhs_order[jj]))

    return np.array(lhs_matches, dtype=np.int64), np.array(rhs_matches, dtype=np.int64)
