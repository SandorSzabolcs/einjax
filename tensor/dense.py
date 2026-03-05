"""
Dense tensor implementation for EinJAX.

Ported from einsql/einsql.py DenseTensor class (lines 456-567).
Replaces PyTorch with NumPy for data storage and sparsity metric
computation. JAX arrays are accepted and converted to NumPy for
metadata computation; the original array is preserved for execution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from einjax.core.types import TilingScheme
from .base import BaseTensor


class DenseTensor(BaseTensor):
    """Dense tensor wrapping a NumPy or JAX array.

    Accepts numpy.ndarray or jax.Array. Internally stores a NumPy copy
    for sparsity analysis. The original data reference is preserved in
    self.data for downstream execution.

    Ported from einsql/einsql.py DenseTensor (lines 456-567), with
    torch.Tensor replaced by numpy.ndarray.
    """

    def __init__(self, name: str, data: Any):
        """Initialize a dense tensor.

        Args:
            name: Identifier for this tensor.
            data: Array data — numpy.ndarray, jax.Array, or any
                  object with .shape and np.asarray() support.
        """
        self._data_np = np.asarray(data)
        super().__init__(name, tuple(self._data_np.shape))
        self.data = data
        self._compute_sparsity_metrics()

    def _compute_sparsity_metrics(self) -> None:
        """Compute sparsity information for each tiling scheme.

        Ported from einsql DenseTensor._compute_sparsity_metrics
        (lines 464-472). Traverses all elements and tracks which tile
        coordinates contain non-zero values, updating each scheme's
        num_tuples and value_count to reflect actual sparsity.
        """
        tuple_counter: dict[tuple[int, ...], list[set[Any]]] = {
            s: [set() for _ in range(len(self.shape) + 1)]
            for s in self.tile_shapes
        }
        self._traverse_elements(tuple_counter, self._data_np, [])

        for tile_shape, scheme in self.schemes.items():
            sets = tuple_counter[tile_shape]
            scheme.num_tuples = len(sets[-1])
            scheme.value_count = tuple(len(s) for s in sets[:-1])

    def _traverse_elements(
        self,
        tuple_counter: dict[tuple[int, ...], list[set[Any]]],
        subarray: np.ndarray,
        index: list[int],
    ) -> None:
        """Recursively traverse tensor elements to count non-zeros.

        Ported from einsql DenseTensor._traverse_elements (lines 474-500).
        """
        dim = len(index)

        if dim + 1 == len(self.shape):
            for i in range(self.shape[dim]):
                if subarray[i] == 0:
                    continue

                index.append(i)
                for tile_shape, sets in tuple_counter.items():
                    outer_index = tuple(
                        idx // ts for idx, ts in zip(index, tile_shape)
                    )
                    for d, idx_val in enumerate(outer_index):
                        sets[d].add(idx_val)
                    sets[-1].add(outer_index)
                index.pop()
            return

        for i in range(self.shape[dim]):
            index.append(i)
            self._traverse_elements(tuple_counter, subarray[i], index)
            index.pop()
