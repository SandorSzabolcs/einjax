"""
Base tensor class for EinJAX.

Ported from einsql/einsql.py Tensor class (lines 415-453) and
IndexedTerm (lines 403-409). Removes SQL-specific methods;
adds JAX-oriented interface.
"""

from __future__ import annotations

import itertools
from typing import Any

from einjax.core.notation import find_all_factors
from einjax.core.types import BinaryOp, Expr, TilingScheme


class IndexedTerm:
    """A tensor with index labels for Einstein notation.

    Ported from einsql/einsql.py lines 403-409.
    """

    def __init__(self, tensor: BaseTensor, indices: str):
        self.tensor = tensor
        self.indices = indices


class BaseTensor(Expr):
    """Base class for tensors with tiling scheme support.

    Ported from einsql/einsql.py Tensor class (lines 415-453).
    A tensor tracks all possible tiling configurations and their
    associated cost metrics for optimization purposes.

    Subclasses must hold actual data and implement _compute_sparsity_metrics().
    """

    def __init__(self, name: str, shape: tuple[int, ...]):
        self.name = name
        self.shape = shape

        # Generate all possible tiling schemes (einsql lines 428-434)
        factors = [find_all_factors(d) for d in shape]
        self.tile_shapes = list(itertools.product(*factors))
        self.schemes: dict[tuple[int, ...], TilingScheme] = {
            tile_shape: TilingScheme(self, self.shape, tile_shape)
            for tile_shape in self.tile_shapes
        }

    def __add__(self, other: BaseTensor) -> BinaryOp:
        return BinaryOp("+", self, other)

    def __sub__(self, other: BaseTensor) -> BinaryOp:
        return BinaryOp("-", self, other)

    def __truediv__(self, other: BaseTensor) -> BinaryOp:
        return BinaryOp("/", self, other)

    def __getitem__(self, indices: str) -> IndexedTerm:
        return IndexedTerm(self, indices)

    def flops(self) -> float:
        return 0.0

    @property
    def ndim(self) -> int:
        return len(self.shape)
