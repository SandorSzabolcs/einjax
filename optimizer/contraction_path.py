"""
Multi-tensor contraction path planning via opt_einsum integration.

For multi-tensor einsum expressions (3+ operands), uses opt_einsum to
determine the optimal binary contraction order, then runs DPOptimizer
on each step to select optimal tiling.

Ported from einsql/einsql.py lines 1393-1421. The key change is that
instead of calling einsql's ``einsum()`` recursively to build SQL
Reduction objects, we build a DAG of ReductionInfo nodes and run
DPOptimizer on each.

See PRD Section 5.4 for full specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import opt_einsum as oe

from einjax.core.notation import normalize_notation
from einjax.core.types import TilingScheme
from einjax.optimizer.cost_model import CostModelConfig
from einjax.optimizer.dp import DPOptimizer, ReductionInfo, infer_reduction_info
from einjax.tensor.base import BaseTensor, IndexedTerm


@dataclass
class ContractionStep:
    """A single binary contraction in the planned contraction path.

    Attributes:
        step_index: Position in the contraction sequence (0-based).
        formula: Einsum formula for this binary step (e.g., "ij,jk->ik").
        input_indices: Original operand indices consumed by this step.
        lhs_labels: Index labels for left operand.
        rhs_labels: Index labels for right operand.
        output_labels: Index labels for the result.
        output_tensor: The intermediate or final output BaseTensor.
        reduction_info: ReductionInfo for this binary contraction.
        best_scheme: Optimal TilingScheme selected by DPOptimizer (None
            if optimizer has not been run).
    """

    step_index: int
    formula: str
    input_indices: tuple[int, ...]
    lhs_labels: str
    rhs_labels: str
    output_labels: str
    output_tensor: BaseTensor
    reduction_info: ReductionInfo
    best_scheme: TilingScheme | None = None


@dataclass
class ContractionPlan:
    """Complete contraction plan for a multi-tensor einsum.

    Attributes:
        einsum_string: Original normalized einsum string.
        steps: Ordered list of binary contraction steps.
        total_cost: Sum of accumulated costs across all steps.
        tiling_schemes: Ordered list of TilingSchemes from the DP
            optimizer's optimal plan for the final output.
    """

    einsum_string: str
    steps: list[ContractionStep]
    total_cost: float
    tiling_schemes: list[TilingScheme]


def get_contraction_order(
    einsum_string: str,
    shapes: list[tuple[int, ...]],
    optimize: str = "dp",
) -> list[tuple[tuple[int, ...], str]]:
    """Use opt_einsum to determine binary contraction order.

    Args:
        einsum_string: Normalized einsum string (must contain '->').
        shapes: List of tensor shapes.
        optimize: opt_einsum optimization strategy. "dp" uses dynamic
            programming, "greedy" uses greedy heuristic, "optimal" uses
            exhaustive search. Note: "dp" here refers to opt_einsum's
            internal DP for contraction ordering, separate from our
            DPOptimizer for tile_shape selection.

    Returns:
        List of (input_indices, formula) pairs describing each binary
        contraction step. ``input_indices`` are into the *current*
        operand list (which shrinks as contractions are performed).
    """
    views = [np.empty(shape) for shape in shapes]
    _, path_info = oe.contract_path(
        einsum_string, *views, optimize=optimize,
    )

    steps: list[tuple[tuple[int, ...], str]] = []
    for contraction in path_info.contraction_list:
        indices = contraction[0]
        formula = contraction[2]
        steps.append((indices, formula))

    return steps


def plan_contraction(
    einsum_string: str,
    tensors: list[BaseTensor],
    cost_config: CostModelConfig,
    optimize: str = "dp",
) -> ContractionPlan:
    """Plan contraction order and tiling for a multi-tensor einsum.

    1. Normalize the einsum string.
    2. Use opt_einsum to find the optimal binary contraction order.
    3. Build intermediate BaseTensor nodes for each contraction step.
    4. Run DPOptimizer on each binary contraction.
    5. Return a ContractionPlan with the optimal tiling schemes.

    Args:
        einsum_string: Einsum notation (e.g., "ij,jk,kl->il").
        tensors: Input tensors.
        cost_config: Hardware cost model configuration.
        optimize: opt_einsum optimization strategy ("dp", "greedy",
            "optimal").

    Returns:
        ContractionPlan with ordered steps and optimal tiling schemes.

    Raises:
        ValueError: If fewer than 2 tensors are provided.
    """
    if len(tensors) < 2:
        raise ValueError(
            f"plan_contraction requires at least 2 tensors, got {len(tensors)}"
        )

    einsum_string = normalize_notation(einsum_string)
    shapes = [t.shape for t in tensors]
    contraction_order = get_contraction_order(einsum_string, shapes, optimize)

    optimizer = DPOptimizer(cost_config)
    operands: list[BaseTensor] = list(tensors)
    steps: list[ContractionStep] = []

    for step_idx, (indices, formula) in enumerate(contraction_order):
        lhs_operand = operands[indices[0]]
        rhs_operand = operands[indices[1]]

        # Parse formula into input labels and output labels
        input_part, output_labels = formula.split("->")
        subscript_labels = input_part.split(",")
        lhs_labels = subscript_labels[0]
        rhs_labels = subscript_labels[1]

        # Create IndexedTerms for the binary contraction
        lhs_term = IndexedTerm(lhs_operand, lhs_labels)
        rhs_term = IndexedTerm(rhs_operand, rhs_labels)

        # Infer contraction metadata
        info = infer_reduction_info(output_labels, lhs_term, rhs_term)

        # Create intermediate output tensor
        is_final = step_idx == len(contraction_order) - 1
        output_name = f"_t{step_idx}" if not is_final else "_output"
        output_tensor = BaseTensor(output_name, info.output_shape)

        # Run DP optimizer for this binary contraction
        optimizer.optimize_reduction(output_tensor, info)
        best_scheme = optimizer.get_best_scheme(output_tensor)

        step = ContractionStep(
            step_index=step_idx,
            formula=formula,
            input_indices=indices,
            lhs_labels=lhs_labels,
            rhs_labels=rhs_labels,
            output_labels=output_labels,
            output_tensor=output_tensor,
            reduction_info=info,
            best_scheme=best_scheme,
        )
        steps.append(step)

        # Remove contracted operands (in reverse order to preserve indices)
        for idx in sorted(indices, reverse=True):
            del operands[idx]

        # Add intermediate result to operand list
        operands.append(output_tensor)

    # Extract optimal tiling plan from the final output
    final_tensor = steps[-1].output_tensor if steps else None
    if final_tensor is not None:
        tiling_schemes = optimizer.get_optimal_plan(final_tensor)
    else:
        tiling_schemes = []

    total_cost = (
        steps[-1].best_scheme.accumulated_cost
        if steps and steps[-1].best_scheme is not None
        else float("inf")
    )

    return ContractionPlan(
        einsum_string=einsum_string,
        steps=steps,
        total_cost=total_cost,
        tiling_schemes=tiling_schemes,
    )
