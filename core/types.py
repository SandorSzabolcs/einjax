"""
Core data types for EinJAX.

Ported from einsql/einsql.py (lines 85-169). TilingScheme is extended
with JAX-specific fields (partition_spec, mesh_axes) per PRD Section 3.2.
CaseAssignment is new, per PRD Section 3.3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from math import prod
from typing import Any


# =============================================================================
# Expression AST Classes
# =============================================================================

class Expr(ABC):
    """Abstract base class for all expressions."""

    @abstractmethod
    def flops(self) -> float:
        """Return the number of floating point operations."""
        pass


class UnaryOp(Expr):
    """Unary operation expression."""

    def __init__(self, opcode: str, expr: Expr):
        self.opcode = opcode
        self.expr = expr

    def flops(self) -> float:
        return 1.0 + self.expr.flops()


class BinaryOp(Expr):
    """Binary operation expression."""

    def __init__(self, opcode: str, lhs: Expr, rhs: Expr):
        self.opcode = opcode
        self.lhs = lhs
        self.rhs = rhs

    def flops(self) -> float:
        return 1.0 + self.lhs.flops() + self.rhs.flops()


class Constant(Expr):
    """Constant value expression."""

    def __init__(self, value: float):
        self.value = value

    def flops(self) -> float:
        return 0.0


# =============================================================================
# Case Assignment
# =============================================================================

class CaseAssignment(Enum):
    """Classification of a tensor dimension as upper-case or lower-case.

    Per PRD Section 3.3:
    - UPPER (sharded): tile_shape[d] < shape[d] — partitioned across devices
    - LOWER (kernel):  tile_shape[d] == shape[d] — fully contained in each shard
    """
    UPPER = "upper"
    LOWER = "lower"


# =============================================================================
# Tiling Scheme
# =============================================================================

@dataclass
class TilingScheme:
    """Represents a tiling configuration for a tensor.

    Tiling divides a tensor into smaller blocks for efficient processing.
    Each scheme tracks the cost metrics for its configuration.

    Extended from einsql's TilingScheme with JAX-specific fields:
    partition_spec and mesh_axes (per PRD Section 3.2).
    """
    node: Any
    shape: tuple[int, ...]
    tile_shape: tuple[int, ...]

    # Computed fields
    tile_size: int = field(init=False)
    value_count: tuple[int, ...] = field(init=False)
    num_tuples: int = field(init=False)

    # Cost metrics
    cost: float = 0.0
    comm: float = 0.0
    flops: float = 0.0
    accumulated_cost: float = 0.0
    accumulated_comm: float = 0.0
    accumulated_flops: float = 0.0

    # Dependency tracking
    source: tuple = field(default_factory=tuple)
    dependencies: set = field(default_factory=set)

    # JAX-specific extensions (PRD Section 3.2)
    partition_spec: Any | None = None   # PartitionSpec, kept as Any to avoid hard JAX dep
    mesh_axes: dict[str, int] | None = None  # axis name -> device count

    def __post_init__(self):
        self.tile_size = prod(self.tile_shape)
        self.value_count = tuple(
            length // tile_length
            for length, tile_length in zip(self.shape, self.tile_shape)
        )
        self.num_tuples = prod(self.value_count)

    def __hash__(self):
        return hash((id(self.node), self.tile_shape))

    def get_case_assignments(self) -> list[CaseAssignment]:
        """Derive upper/lower case assignment for each dimension.

        Per PRD Section 3.3:
        - Upper-case: tile_shape[d] < shape[d]
        - Lower-case: tile_shape[d] == shape[d]
        """
        return [
            CaseAssignment.LOWER if ts == s else CaseAssignment.UPPER
            for s, ts in zip(self.shape, self.tile_shape)
        ]
