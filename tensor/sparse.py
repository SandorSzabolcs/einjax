"""
Sparse tensor implementations for EinJAX.

Provides two representations:
- SparseTensor: User-facing sparse tensor wrapping scipy COO data.
  Extends BaseTensor with COO-based sparsity metric computation.
  Ported from einsql/einsql.py SparseTensor (lines 680-801),
  replacing PyTorch with scipy.sparse/NumPy.

- SparseTensorRelation: Internal tensor-relational representation
  where each "tuple" is a tile coordinate plus a dense sub-tensor
  of values. This is the core data structure for the sparse
  execution path (PRD Section 3.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

from .base import BaseTensor


@dataclass
class SparseTensorRelation:
    """Sparse tensor in tensor-relational form.

    Each entry is a tile coordinate plus a dense sub-tensor of values.
    The coords array plays the role of the upper-case (relational) indices;
    the values array holds the lower-case (kernel) data.

    Per PRD Section 3.1:
        coords: int32 array of shape (num_tuples, ndim) — tile coordinates
        values: float32 array of shape (num_tuples, *tile_shape) — dense tile data
        shape: full logical tensor shape
        tile_shape: dimensions of each dense tile

    This replaces both jax.experimental.sparse.BCOO (experimental/unstable)
    and the PostgreSQL row representation from einsql.
    """
    coords: np.ndarray
    values: np.ndarray
    shape: tuple[int, ...]
    tile_shape: tuple[int, ...]

    @property
    def num_tuples(self) -> int:
        """Number of non-empty tiles."""
        return self.coords.shape[0]

    @property
    def ndim(self) -> int:
        """Number of tensor dimensions."""
        return len(self.shape)

    @property
    def tile_size(self) -> int:
        """Number of elements per tile."""
        return prod(self.tile_shape)

    @property
    def nnz(self) -> int:
        """Total number of non-zero elements across all tiles."""
        return int(np.count_nonzero(self.values))

    @property
    def density(self) -> float:
        """Fraction of non-zero elements in the logical tensor."""
        total = prod(self.shape)
        return self.nnz / total if total > 0 else 0.0

    def to_dense(self) -> np.ndarray:
        """Reconstruct the full dense tensor from tiles.

        Returns:
            Dense numpy array of shape self.shape.
        """
        result = np.zeros(self.shape, dtype=self.values.dtype)
        for t in range(self.num_tuples):
            coord = tuple(self.coords[t])
            slices = tuple(
                slice(coord[d] * self.tile_shape[d],
                      (coord[d] + 1) * self.tile_shape[d])
                for d in range(self.ndim)
            )
            result[slices] = self.values[t]
        return result


class SparseTensor(BaseTensor):
    """Sparse tensor wrapping COO-format data.

    Accepts scipy.sparse matrices, NumPy arrays, or any array-like with
    a .shape attribute. Internally stores COO indices and values as NumPy
    arrays. Computes sparsity metrics for all tiling schemes.

    Ported from einsql/einsql.py SparseTensor (lines 680-801),
    replacing PyTorch sparse with scipy.sparse/NumPy COO format.
    """

    def __init__(self, name: str, data: Any):
        """Initialize a sparse tensor.

        Args:
            name: Identifier for this tensor.
            data: Sparse data — scipy.sparse matrix, or dense array-like.
                  If dense, non-zero entries are extracted automatically.
        """
        # Extract COO indices and values
        self._indices, self._values, shape = _extract_coo(data)
        self._shape_full = shape
        super().__init__(name, shape)
        self.data = data
        self._compute_sparsity_metrics()

    @property
    def nnz(self) -> int:
        """Number of stored non-zero elements."""
        return len(self._values)

    def _compute_sparsity_metrics(self) -> None:
        """Compute sparsity information from COO format (N-dimensional).

        Ported from einsql/einsql.py SparseTensor._compute_sparsity_metrics
        (lines 695-722). For each tiling scheme, counts the number of
        non-empty tiles (num_tuples) and the number of distinct tile
        coordinates per dimension (value_count).
        """
        ndim = len(self.shape)
        indices = self._indices  # shape: (nnz, ndim)
        values = self._values    # shape: (nnz,)

        # Filter out zero values once using vectorized ops
        nz_mask = values != 0
        nz_indices = indices[nz_mask]

        if len(nz_indices) == 0:
            for tile_shape, scheme in self.schemes.items():
                scheme.num_tuples = 0
                scheme.value_count = tuple(0 for _ in range(ndim))
            return

        for tile_shape, scheme in self.schemes.items():
            tile_shape_arr = np.array(tile_shape, dtype=np.int32)
            tile_coords = nz_indices // tile_shape_arr  # (nnz, ndim)

            # Count unique tile coordinate values per dimension
            value_count = tuple(
                len(np.unique(tile_coords[:, d])) for d in range(ndim)
            )

            # Count unique tile coordinate tuples
            unique_tiles = np.unique(tile_coords, axis=0)
            scheme.num_tuples = len(unique_tiles)
            scheme.value_count = value_count

    def to_relation(self, tile_shape: tuple[int, ...]) -> SparseTensorRelation:
        """Convert to tensor-relational form with the given tile shape.

        Groups COO entries by tile coordinate, builds dense tile arrays,
        and returns a SparseTensorRelation. Only non-empty tiles are included.

        Args:
            tile_shape: Dimensions of each dense tile. Must evenly divide
                        the tensor shape.

        Returns:
            SparseTensorRelation with coords and dense tile values.
        """
        ndim = len(self.shape)
        for d in range(ndim):
            if self.shape[d] % tile_shape[d] != 0:
                raise ValueError(
                    f"tile_shape[{d}]={tile_shape[d]} does not evenly "
                    f"divide shape[{d}]={self.shape[d]}"
                )

        indices = self._indices
        values = self._values

        # Filter out zero values using vectorized ops
        nz_mask = values != 0
        nz_indices = indices[nz_mask]
        nz_values = values[nz_mask]

        if len(nz_values) == 0:
            coords_arr = np.zeros((0, ndim), dtype=np.int32)
            values_arr = np.zeros((0, *tile_shape), dtype=np.float64)
            return SparseTensorRelation(
                coords=coords_arr, values=values_arr,
                shape=self.shape, tile_shape=tile_shape,
            )

        tile_shape_arr = np.array(tile_shape, dtype=np.int32)

        # Compute tile coordinates and within-tile positions vectorized
        tile_coords = nz_indices // tile_shape_arr  # (nnz, ndim)
        within_tile = nz_indices % tile_shape_arr   # (nnz, ndim)

        # Find unique tile coordinates and map each element to its tile index
        unique_tiles, inverse = np.unique(
            tile_coords, axis=0, return_inverse=True,
        )
        num_tiles = len(unique_tiles)

        # Build tile values using vectorized indexing
        values_arr = np.zeros((num_tiles, *tile_shape), dtype=np.float64)
        idx = (inverse,) + tuple(within_tile[:, d] for d in range(ndim))
        values_arr[idx] = nz_values

        return SparseTensorRelation(
            coords=unique_tiles.astype(np.int32),
            values=values_arr,
            shape=self.shape,
            tile_shape=tile_shape,
        )

    def to_dense_array(self) -> np.ndarray:
        """Reconstruct the full dense array from COO data.

        Returns:
            Dense numpy array of shape self.shape.
        """
        result = np.zeros(self.shape, dtype=np.float64)
        for nz_idx in range(len(self._values)):
            coords = tuple(int(self._indices[nz_idx, d]) for d in range(self.ndim))
            result[coords] = self._values[nz_idx]
        return result


def _extract_coo(data: Any) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Extract COO indices and values from various sparse formats.

    Supports:
    - scipy.sparse matrices (coo_matrix, csr_matrix, csc_matrix, etc.)
    - Dense numpy arrays (non-zeros extracted automatically)
    - Any array-like with .shape and np.asarray() support

    Returns:
        (indices, values, shape) where:
        - indices: int32 array of shape (nnz, ndim)
        - values: float64 array of shape (nnz,)
        - shape: tuple of dimension sizes
    """
    # Try scipy.sparse first
    try:
        import scipy.sparse as sp
        if sp.issparse(data):
            coo = sp.coo_matrix(data) if data.ndim == 2 else data.tocoo()
            indices = np.stack([coo.row, coo.col], axis=1).astype(np.int32)
            values = np.asarray(coo.data, dtype=np.float64)
            return indices, values, tuple(coo.shape)
    except ImportError:
        pass

    # Dense array — extract non-zeros
    arr = np.asarray(data, dtype=np.float64)
    shape = tuple(arr.shape)
    nz_indices = np.nonzero(arr)
    if len(nz_indices[0]) == 0:
        indices = np.zeros((0, arr.ndim), dtype=np.int32)
        values = np.zeros((0,), dtype=np.float64)
    else:
        indices = np.stack(nz_indices, axis=1).astype(np.int32)
        values = arr[nz_indices]

    return indices, values, shape
