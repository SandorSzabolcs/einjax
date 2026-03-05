"""Execution engine for tiled computation DAGs as sharded JAX operations.

Replaces einsql's SQLGenerator (lines 1428-1650). Instead of SQL CTEs,
produces a sequence of jax.jit-compiled stages ordered by topological sort.

Ported topological sort from einsql/einsql.py lines 1562-1610 (Kahn's
algorithm via _build_dependency_graph + _topological_sort).

See PRD Section 6 for full specification.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from einjax.core.types import TilingScheme
from einjax.execution.dense_kernels import execute_dense_einsum, execute_sharded_einsum
from einjax.optimizer.contraction_path import ContractionPlan, ContractionStep
from einjax.sharding.partition import tile_shape_to_partition_spec
from einjax.sharding.reshard import needs_reshard, reshard_dense


def build_dependency_graph(
    root_scheme: TilingScheme,
) -> dict[int, tuple[TilingScheme, set[int], set[int]]]:
    """Build adjacency list representation of the dependency graph.

    Walks the source pointers from the root scheme back to input schemes,
    building a directed acyclic graph.

    Ported from einsql/einsql.py _build_dependency_graph (lines 1562-1582).

    Args:
        root_scheme: The output tiling scheme to trace from.

    Returns:
        Dict mapping scheme id to (scheme, incoming_ids, outgoing_ids).
    """
    graph: dict[int, tuple[TilingScheme, set[int], set[int]]] = {}
    working_list = [root_scheme]

    while working_list:
        scheme = working_list.pop(0)
        sid = id(scheme)

        if sid not in graph:
            graph[sid] = (scheme, set(), set())

        for src in scheme.source:
            src_id = id(src)
            working_list.append(src)
            if src_id not in graph:
                graph[src_id] = (src, set(), set())
            graph[src_id][2].add(sid)   # outgoing from src to scheme
            graph[sid][1].add(src_id)   # incoming to scheme from src

    return graph


def topological_sort(
    root_scheme: TilingScheme,
) -> list[TilingScheme]:
    """Perform topological sort on the dependency graph.

    Uses Kahn's algorithm: repeatedly remove nodes with no incoming edges.
    Returns schemes in dependency order (inputs first, root last).

    Ported from einsql/einsql.py _topological_sort (lines 1584-1610).

    Args:
        root_scheme: The output tiling scheme to trace from.

    Returns:
        Ordered list of TilingSchemes from inputs to output.

    Raises:
        RuntimeError: If a cycle is detected in the dependency graph.
    """
    raw_graph = build_dependency_graph(root_scheme)

    # Deep copy the edge sets so we can mutate them
    graph: dict[int, tuple[TilingScheme, set[int], set[int]]] = {
        sid: (scheme, set(incoming), set(outgoing))
        for sid, (scheme, incoming, outgoing) in raw_graph.items()
    }

    ordered: list[TilingScheme] = []

    while graph:
        # Find node with no incoming edges
        ready_id = None
        for sid, (scheme, incoming, outgoing) in graph.items():
            if not incoming:
                ready_id = sid
                break

        if ready_id is None:
            raise RuntimeError("Cycle detected in dependency graph")

        ready_scheme, _, outgoing = graph.pop(ready_id)
        for out_id in outgoing:
            if out_id in graph:
                graph[out_id][1].discard(ready_id)

        ordered.append(ready_scheme)

    return ordered


class ExecutionEngine:
    """Execute a tiled computation DAG as sharded JAX operations.

    Replaces einsql's SQLGenerator: instead of SQL CTEs, produces a
    sequence of jax.jit-compiled stages ordered by topological sort.

    Per PRD Section 6.1:
    1. Topological sort (Kahn's algorithm)
    2. For each node in order:
       a. Determine path: dense sharded vs. sparse
       b. Execute via appropriate kernel
       c. Reshard output if next node needs different tiling
    3. Return final result

    Args:
        mesh: JAX device mesh for sharded execution. If None,
            runs on a single device without sharding.
        cost_config: Hardware cost model configuration (optional,
            used for resharding cost estimation).
    """

    def __init__(
        self,
        mesh: Any = None,
        cost_config: Any = None,
        mesh_axis_names: tuple[str, ...] | None = None,
    ):
        self.mesh = mesh
        self.cost_config = cost_config
        if mesh is not None and mesh_axis_names is None:
            self.mesh_axis_names = tuple(mesh.axis_names)
        else:
            self.mesh_axis_names = mesh_axis_names or ("x", "y", "z", "w")

    def _get_partition_spec(self, scheme: TilingScheme) -> tuple:
        """Derive a PartitionSpec tuple from a tiling scheme.

        Args:
            scheme: A TilingScheme with shape and tile_shape.

        Returns:
            Tuple suitable for jax.sharding.PartitionSpec construction.
        """
        return tile_shape_to_partition_spec(
            scheme.shape, scheme.tile_shape, self.mesh_axis_names
        )

    def _derive_output_spec_from_formula(
        self,
        formula: str,
        in_specs: list[tuple],
        operands: list[jnp.ndarray],
    ) -> tuple:
        """Derive an output PartitionSpec from the einsum formula and input specs.

        Maps each input index label to its sharding axis via the input specs,
        then builds the output spec by looking up each output label's axis.
        When an index is contracted (summed over), it doesn't appear in the
        output, so any all-reduce needed is handled by GSPMD automatically.

        Args:
            formula: Einsum formula (e.g., "ij,jk->ik").
            in_specs: PartitionSpec tuples for each input operand.
            operands: Input JAX arrays (used for shape validation).

        Returns:
            PartitionSpec tuple for the output.
        """
        input_part, output_labels = formula.split("->")
        input_labels_list = input_part.split(",")

        # Build mapping: label -> mesh axis (first occurrence wins)
        label_to_axis: dict[str, str | None] = {}
        for subscripts, spec in zip(input_labels_list, in_specs):
            for label, axis in zip(subscripts, spec):
                if label not in label_to_axis:
                    label_to_axis[label] = axis

        # Build output spec
        out_spec = tuple(label_to_axis.get(label) for label in output_labels)
        return out_spec

    def _execute_einsum(
        self,
        formula: str,
        source_arrays: list[jnp.ndarray],
        source_schemes: list[TilingScheme] | None = None,
        output_scheme: TilingScheme | None = None,
    ) -> jnp.ndarray:
        """Execute an einsum, using sharded execution when a mesh is available.

        Args:
            formula: Einsum formula string.
            source_arrays: Input JAX arrays.
            source_schemes: Tiling schemes for input arrays (needed for sharding).
            output_scheme: Tiling scheme for the output (if available).

        Returns:
            Result of the einsum contraction.
        """
        if self.mesh is None or source_schemes is None:
            return execute_dense_einsum(formula, *source_arrays)

        in_specs = [self._get_partition_spec(s) for s in source_schemes]

        # Derive output spec from the output scheme if available,
        # otherwise infer from formula and input specs
        if output_scheme is not None:
            out_spec = self._get_partition_spec(output_scheme)
        else:
            out_spec = self._derive_output_spec_from_formula(
                formula, in_specs, source_arrays
            )

        return execute_sharded_einsum(
            formula, *source_arrays,
            mesh=self.mesh, in_specs=in_specs, out_spec=out_spec,
        )

    def _shard_inputs(
        self,
        operands: list[jnp.ndarray],
        schemes: list[TilingScheme],
    ) -> list[jnp.ndarray]:
        """Place input tensors on the mesh with correct sharding.

        For each operand, derives a PartitionSpec from its tiling scheme
        and uses jax.device_put with NamedSharding to distribute the data
        across devices.

        Args:
            operands: Input JAX arrays.
            schemes: Tiling schemes corresponding to each operand.

        Returns:
            List of sharded JAX arrays.
        """
        if self.mesh is None:
            return operands

        sharded = []
        for arr, scheme in zip(operands, schemes):
            spec = tile_shape_to_partition_spec(
                scheme.shape, scheme.tile_shape, self.mesh_axis_names
            )
            sharding = jax.sharding.NamedSharding(
                self.mesh, jax.sharding.PartitionSpec(*spec)
            )
            sharded.append(jax.device_put(arr, sharding))
        return sharded

    def execute_plan(
        self,
        plan: ContractionPlan,
        operands: list[jnp.ndarray],
        gather_output: bool = False,
    ) -> jnp.ndarray:
        """Execute a ContractionPlan with actual JAX arrays.

        Follows the contraction steps in order, executing each binary
        einsum and resharding intermediates when necessary.

        Args:
            plan: ContractionPlan from plan_contraction().
            operands: Input JAX arrays matching the plan's input tensors.
            gather_output: If True, gather the result to a single device
                instead of leaving it sharded across the mesh.

        Returns:
            Result of the full contraction as a JAX array.
        """
        # Shard inputs if mesh is available and plan has tiling schemes
        if self.mesh is not None and plan.tiling_schemes:
            input_schemes = [
                s for s in plan.tiling_schemes
                if not s.source  # leaf nodes are inputs
            ]
            if len(input_schemes) == len(operands):
                operands = self._shard_inputs(operands, input_schemes)

        # Working list of intermediate results (mirrors plan_contraction)
        intermediates: list[jnp.ndarray] = list(operands)

        for step in plan.steps:
            lhs = intermediates[step.input_indices[0]]
            rhs = intermediates[step.input_indices[1]]

            # Use sharded execution when mesh is available and step has tiling info
            source_schemes = None
            output_scheme = None
            if self.mesh is not None and step.best_scheme is not None:
                # Source schemes come from the best_scheme's sources
                source_schemes = list(step.best_scheme.source) if step.best_scheme.source else None
                output_scheme = step.best_scheme

            result = self._execute_einsum(
                step.formula, [lhs, rhs],
                source_schemes=source_schemes,
                output_scheme=output_scheme,
            )

            # Remove consumed operands (reverse order to preserve indices)
            for idx in sorted(step.input_indices, reverse=True):
                del intermediates[idx]

            intermediates.append(result)

        final = intermediates[-1]
        if gather_output and self.mesh is not None:
            final = jax.device_get(final)
            final = jnp.asarray(final)
        return final

    def execute_schemes(
        self,
        root_scheme: TilingScheme,
        data: dict[str, jnp.ndarray],
        formulas: dict[str, str],
        gather_output: bool = False,
    ) -> jnp.ndarray:
        """Execute a DAG of tiling schemes with topological ordering.

        Low-level interface that takes a root TilingScheme, topologically
        sorts its dependency graph, and executes each node in order with
        resharding between consecutive steps when needed.

        Args:
            root_scheme: The output tiling scheme (from DPOptimizer).
            data: Mapping from tensor name to JAX array for input tensors.
            formulas: Mapping from output tensor name to einsum formula
                for each non-input node (e.g., {"result": "ij,jk->ik"}).
            gather_output: If True, gather the result to a single device
                instead of leaving it sharded across the mesh.

        Returns:
            Result of the final operation as a JAX array.
        """
        ordered = topological_sort(root_scheme)
        results: dict[str, jnp.ndarray] = dict(data)

        # Shard input tensors on mesh
        if self.mesh is not None:
            input_schemes = [s for s in ordered if s.node.name in data]
            input_arrays = [data[s.node.name] for s in input_schemes]
            sharded = self._shard_inputs(input_arrays, input_schemes)
            for s, arr in zip(input_schemes, sharded):
                results[s.node.name] = arr

        prev_scheme: TilingScheme | None = None

        for scheme in ordered:
            name = scheme.node.name

            if name in results:
                # Input tensor — already have data (possibly sharded)
                prev_scheme = scheme
                continue

            formula = formulas.get(name)
            if formula is None:
                raise ValueError(
                    f"No formula provided for non-input node '{name}'"
                )

            # Get operand arrays from source schemes
            source_arrays = []
            for src_scheme in scheme.source:
                src_name = src_scheme.node.name
                if src_name not in results:
                    raise ValueError(
                        f"Missing intermediate result for '{src_name}'"
                    )
                arr = results[src_name]

                # Reshard if needed
                if self.mesh is not None and prev_scheme is not None:
                    if needs_reshard(src_scheme, scheme):
                        arr = reshard_dense(
                            arr, src_scheme, scheme, mesh=self.mesh
                        )

                source_arrays.append(arr)

            source_schemes = list(scheme.source)
            result = self._execute_einsum(
                formula, source_arrays,
                source_schemes=source_schemes,
                output_scheme=scheme,
            )
            results[name] = result
            prev_scheme = scheme

        final = results[ordered[-1].node.name]
        if gather_output and self.mesh is not None:
            final = jax.device_get(final)
            final = jnp.asarray(final)
        return final

    def execute_sequence(
        self,
        schemes: list[TilingScheme],
        operands: list[jnp.ndarray],
        formulas: list[str],
        gather_output: bool = False,
    ) -> jnp.ndarray:
        """Execute a pre-ordered sequence of tiling schemes.

        Simplified interface for already-ordered execution plans (e.g.,
        from DPOptimizer.get_optimal_plan). Expects schemes in dependency
        order and formulas for each non-input step.

        The first ``len(operands)`` schemes are treated as inputs; the
        remaining schemes are executed in order using the provided formulas.

        Args:
            schemes: Ordered list of TilingSchemes (inputs first).
            operands: Input JAX arrays, one per input scheme.
            formulas: Einsum formulas for each non-input scheme, in order.
            gather_output: If True, gather the result to a single device
                instead of leaving it sharded across the mesh.

        Returns:
            Result of the final operation as a JAX array.

        Raises:
            ValueError: If the number of formulas doesn't match non-input
                schemes, or if the number of operands doesn't make sense.
        """
        num_inputs = len(operands)
        num_compute = len(schemes) - num_inputs

        if len(formulas) != num_compute:
            raise ValueError(
                f"Expected {num_compute} formulas for {num_compute} "
                f"compute steps, got {len(formulas)}"
            )

        # Shard inputs on mesh
        if self.mesh is not None:
            input_schemes = schemes[:num_inputs]
            operands = self._shard_inputs(operands, input_schemes)

        results: list[jnp.ndarray] = list(operands)

        for i, formula in enumerate(formulas):
            scheme = schemes[num_inputs + i]

            # Gather source operands from results
            source_arrays = []
            for src_scheme in scheme.source:
                # Find this source in our ordered schemes list
                for j, s in enumerate(schemes):
                    if id(s) == id(src_scheme) and j < len(results):
                        arr = results[j]
                        # Reshard if needed
                        if self.mesh is not None and needs_reshard(s, scheme):
                            arr = reshard_dense(
                                arr, s, scheme, mesh=self.mesh
                            )
                        source_arrays.append(arr)
                        break

            source_schemes = list(scheme.source) if scheme.source else None
            result = self._execute_einsum(
                formula, source_arrays,
                source_schemes=source_schemes,
                output_scheme=scheme,
            )
            results.append(result)

        final = results[-1]
        if gather_output and self.mesh is not None:
            final = jax.device_get(final)
            final = jnp.asarray(final)
        return final
