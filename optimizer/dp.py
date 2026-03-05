"""
Dynamic programming optimizer for tile_shape + PartitionSpec selection.

Implements Algorithm 1 from the paper, adapted from einsql/einsql.py
Reduction.__init__ / _compute_all_costs / _update_cost (lines 808-988).
The key change is extracting the optimizer into a standalone class and
extending the search space to include PartitionSpec derivation.

See PRD Section 5 for full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any

from einjax.core.notation import index_to_subscript
from einjax.core.types import TilingScheme
from einjax.optimizer.cost_model import CostModelConfig, compute_join_cost


@dataclass
class ReductionInfo:
    """Metadata for a binary tensor contraction.

    Captures the join and aggregation structure inferred from Einstein
    notation, mirroring einsql's Reduction.infer_shape_and_keys().

    Attributes:
        output_shape: Shape of the output tensor.
        join_keys: List of ((tensor, dim), (tensor, dim)) pairs identifying
            dimensions that are joined (contracted) between operands.
        aggregation_keys: List of (tensor, dim) pairs identifying
            dimensions that appear in the output.
        lhs: The left-hand IndexedTerm.
        rhs: The right-hand IndexedTerm.
        lhs_key_indices: Dimension indices in LHS used as join keys.
        rhs_key_indices: Dimension indices in RHS used as join keys.
    """
    output_shape: tuple[int, ...]
    join_keys: list[tuple[tuple[Any, int], tuple[Any, int]]]
    aggregation_keys: list[tuple[Any, int]]
    lhs: Any  # IndexedTerm
    rhs: Any  # IndexedTerm
    lhs_key_indices: tuple[int, ...]
    rhs_key_indices: tuple[int, ...]


def infer_reduction_info(
    output_labels: str,
    lhs_term: Any,
    rhs_term: Any,
) -> ReductionInfo:
    """Infer output shape, join keys, and aggregation keys from notation.

    Ported from einsql Reduction.infer_shape_and_keys (lines 816-860).

    Args:
        output_labels: Output index labels (e.g., "ik" for matmul).
        lhs_term: IndexedTerm for left operand.
        rhs_term: IndexedTerm for right operand.

    Returns:
        ReductionInfo with all contraction metadata.
    """
    subscripts: dict[str, tuple[int, tuple[Any, int]]] = {}
    join_keys: list[tuple[tuple[Any, int], tuple[Any, int]]] = []

    for tensor_term in (lhs_term, rhs_term):
        for i, subscript in enumerate(tensor_term.indices):
            if subscript not in subscripts:
                subscripts[subscript] = (
                    tensor_term.tensor.shape[i],
                    (tensor_term.tensor, i),
                )
            else:
                value = subscripts[subscript]
                join_keys.append((value[1], (tensor_term.tensor, i)))

    output_shape = tuple(subscripts[s][0] for s in output_labels)
    aggregation_keys = [subscripts[s][1] for s in output_labels]

    lhs_key_indices = tuple(key[0][1] for key in join_keys)
    rhs_key_indices = tuple(key[1][1] for key in join_keys)

    return ReductionInfo(
        output_shape=output_shape,
        join_keys=join_keys,
        aggregation_keys=aggregation_keys,
        lhs=lhs_term,
        rhs=rhs_term,
        lhs_key_indices=lhs_key_indices,
        rhs_key_indices=rhs_key_indices,
    )


class DPOptimizer:
    """Dynamic programming optimizer for tile_shape selection.

    Implements Algorithm 1 from the paper, extended to jointly optimize
    tiling across a computation DAG. For each binary contraction, enumerates
    all compatible (lhs_scheme, rhs_scheme) pairs grouped by join key
    compatibility, scores each via the hardware-aware cost model, and selects
    the scheme minimizing accumulated_cost.

    Adapted from einsql/einsql.py Reduction._compute_all_costs and
    _update_cost (lines 889-988).

    Args:
        cost_config: Hardware cost model configuration.
    """

    def __init__(self, cost_config: CostModelConfig):
        self.cost_config = cost_config

    def optimize_reduction(
        self,
        output_tensor: Any,
        info: ReductionInfo,
    ) -> dict[tuple[int, ...], TilingScheme]:
        """Find optimal tiling scheme for a binary contraction.

        For each output tile_shape candidate, enumerates compatible
        (lhs_scheme, rhs_scheme) pairs and selects the one with minimum
        accumulated cost.

        Args:
            output_tensor: The output tensor (BaseTensor) whose schemes
                will be populated with costs.
            info: ReductionInfo from infer_reduction_info().

        Returns:
            The output tensor's schemes dict, with cost/source/dependencies
            populated for all reachable schemes.
        """
        lhs_tensor = info.lhs.tensor
        rhs_tensor = info.rhs.tensor

        # Initialize output schemes to infinity
        for scheme in output_tensor.schemes.values():
            scheme.cost = float("inf")
            scheme.accumulated_cost = float("inf")

        # Group LHS schemes by join key dimensions (einsql lines 894-898)
        lhs_groups: dict[tuple[int, ...], list[TilingScheme]] = {}
        for tile_shape, lhs_scheme in lhs_tensor.schemes.items():
            key = tuple(tile_shape[idx] for idx in info.lhs_key_indices)
            lhs_groups.setdefault(key, []).append(lhs_scheme)

        # Match RHS schemes with compatible LHS schemes (einsql lines 900-905)
        for tile_shape, rhs_scheme in rhs_tensor.schemes.items():
            key = tuple(tile_shape[idx] for idx in info.rhs_key_indices)
            if key in lhs_groups:
                for lhs_scheme in lhs_groups[key]:
                    self._update_cost(output_tensor, info, lhs_scheme, rhs_scheme)

        return output_tensor.schemes

    def _update_cost(
        self,
        output_tensor: Any,
        info: ReductionInfo,
        lhs: TilingScheme,
        rhs: TilingScheme,
    ) -> None:
        """Score a (lhs_scheme, rhs_scheme) pair and update the output scheme.

        Adapted from einsql Reduction._update_cost (lines 907-988) with
        hardware-aware cost formulas from CostModelConfig.

        Args:
            output_tensor: Output tensor whose schemes are being optimized.
            info: Contraction metadata.
            lhs: Left-hand tiling scheme candidate.
            rhs: Right-hand tiling scheme candidate.
        """
        # Derive output tile_shape from aggregation keys (einsql lines 910-923)
        tile_shape = []
        agg_num_tuples = 1

        for tensor, dim in info.aggregation_keys:
            if tensor is lhs.node:
                tile_shape.append(lhs.tile_shape[dim])
                agg_num_tuples *= lhs.value_count[dim]
            elif tensor is rhs.node:
                tile_shape.append(rhs.tile_shape[dim])
                agg_num_tuples *= rhs.value_count[dim]
            else:
                return  # aggregation key not in either operand

        tile_shape_t = tuple(tile_shape)
        scheme = output_tensor.schemes.get(tile_shape_t)
        if scheme is None:
            return

        # Communication cost (transfer bytes) — einsql lines 929-931
        dtype_size = 4  # float32
        lhs_size = (len(lhs.tile_shape) + lhs.tile_size) * lhs.num_tuples
        rhs_size = (len(rhs.tile_shape) + rhs.tile_size) * rhs.num_tuples
        bytes_transferred = dtype_size * (lhs_size + rhs_size)
        join_comm = self.cost_config.transfer_cost(bytes_transferred)

        # Join tuple count estimation (einsql lines 934-938)
        if info.join_keys:
            join_divisor = prod(
                max(lhs.value_count[key[0][1]], rhs.value_count[key[1][1]])
                for key in info.join_keys
            )
        else:
            join_divisor = 1
        join_num_tuples = lhs.num_tuples * rhs.num_tuples / join_divisor

        # Tile join cost (einsql lines 941-946)
        tile_join_cost = lhs.tile_size * rhs.tile_size
        for key in info.join_keys:
            if key[0][0] is lhs.node:
                tile_join_cost /= lhs.tile_shape[key[0][1]]
            elif key[0][0] is rhs.node:
                tile_join_cost /= rhs.tile_shape[key[0][1]]

        # Number of groups per tile (einsql lines 949-954)
        tile_num_group = 1.0
        for tensor, dim in info.aggregation_keys:
            if tensor is lhs.node:
                tile_num_group *= lhs.tile_shape[dim]
            elif tensor is rhs.node:
                tile_num_group *= rhs.tile_shape[dim]

        join_kernel_cost = 2 * tile_join_cost - tile_num_group
        join_flops = join_num_tuples * join_kernel_cost

        # Aggregation (einsql lines 959-969)
        agg_num_tuples = min(join_num_tuples / 2, agg_num_tuples) if join_num_tuples > 0 else 1

        agg_bytes = dtype_size * (
            (len(info.aggregation_keys) + len(info.join_keys) + scheme.tile_size)
            * join_num_tuples
        )
        agg_comm = self.cost_config.transfer_cost(agg_bytes)

        agg_kernel_cost = scheme.tile_size
        agg_flops = (
            (join_num_tuples / agg_num_tuples - 1)
            * agg_num_tuples
            * agg_kernel_cost
        ) if agg_num_tuples > 0 else 0.0

        # Total kernel cost via hardware model
        total_flops = join_flops + agg_flops
        kernel_time = self.cost_config.kernel_cost(total_flops)

        # Fixed cost (launch overhead)
        num_launches = int(join_num_tuples)
        if agg_num_tuples > 0 and join_num_tuples > 0:
            num_launches += int(
                (join_num_tuples / agg_num_tuples - 1) * agg_num_tuples
            )
        fixed_time = self.cost_config.fixed_cost(num_launches)

        # Multi-device communication: all-reduce for cross-device aggregation
        output_tile_size = prod(tile_shape)
        output_bytes = dtype_size * output_tile_size * join_num_tuples
        all_reduce = self.cost_config.all_reduce_cost(output_bytes)

        # Synchronization overhead
        sync = self.cost_config.parallelism_overhead()

        # Total cost for this operation
        cost = join_comm + agg_comm + all_reduce + kernel_time + fixed_time + sync

        # Accumulated cost including parent costs (einsql lines 977-988)
        dependencies = lhs.dependencies | rhs.dependencies | {lhs, rhs}
        accumulated_cost = cost + sum(s.cost for s in dependencies)

        if accumulated_cost < scheme.accumulated_cost:
            scheme.cost = cost
            scheme.comm = join_comm + agg_comm + all_reduce
            scheme.flops = total_flops
            scheme.accumulated_cost = accumulated_cost
            scheme.accumulated_comm = scheme.comm + sum(s.comm for s in dependencies)
            scheme.accumulated_flops = scheme.flops + sum(s.flops for s in dependencies)
            scheme.source = (lhs, rhs)
            scheme.dependencies = dependencies

    def get_best_scheme(
        self,
        tensor: Any,
    ) -> TilingScheme | None:
        """Return the scheme with the lowest accumulated cost.

        Args:
            tensor: A tensor whose schemes have been populated by
                optimize_reduction().

        Returns:
            The TilingScheme with the minimum accumulated_cost, or None
            if no scheme has finite cost.
        """
        best = None
        for scheme in tensor.schemes.values():
            if scheme.accumulated_cost < float("inf"):
                if best is None or scheme.accumulated_cost < best.accumulated_cost:
                    best = scheme
        return best

    def get_optimal_plan(
        self,
        tensor: Any,
    ) -> list[TilingScheme]:
        """Trace back through source pointers to build the full plan.

        Returns the list of TilingSchemes in dependency order (leaves first),
        representing the optimal execution plan from inputs to output.

        Args:
            tensor: The output tensor whose schemes have been optimized.

        Returns:
            Ordered list of TilingSchemes from inputs to output.
        """
        best = self.get_best_scheme(tensor)
        if best is None:
            return []

        plan: list[TilingScheme] = []
        visited: set[int] = set()
        self._trace_plan(best, plan, visited)
        return plan

    def _trace_plan(
        self,
        scheme: TilingScheme,
        plan: list[TilingScheme],
        visited: set[int],
    ) -> None:
        """Recursively trace source pointers to build execution plan."""
        scheme_id = id(scheme)
        if scheme_id in visited:
            return
        visited.add(scheme_id)

        if scheme.source:
            for parent in scheme.source:
                self._trace_plan(parent, plan, visited)

        plan.append(scheme)
