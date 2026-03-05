"""
Tests for einjax execution layer: dense_kernels, api, and engine.

Validates that einjax.einsum() produces correct results on a single
device, matching numpy.einsum to within 1e-5 relative tolerance
(PRD Section 15, Success Criterion 1). Also tests the ExecutionEngine
for multi-stage DAG execution with topological sort (PRD Section 6).
"""

import unittest

import numpy as np
import jax.numpy as jnp

from einjax.execution.dense_kernels import execute_dense_einsum
from einjax.execution.engine import (
    ExecutionEngine,
    build_dependency_graph,
    topological_sort,
)
from einjax.api import einsum, analyze, AnalysisResult
from einjax.core.types import TilingScheme
from einjax.optimizer.cost_model import CostModelConfig
from einjax.optimizer.contraction_path import plan_contraction
from einjax.tensor.base import BaseTensor


class TestDenseKernels(unittest.TestCase):
    """Tests for execute_dense_einsum (low-level kernel)."""

    def test_matmul(self):
        """Test matrix multiplication via jnp.einsum."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = execute_dense_einsum("ij,jk->ik", a, b)
        expected = np.einsum("ij,jk->ik", np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_trace(self):
        """Test trace (diagonal sum) via einsum."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = execute_dense_einsum("ii->", a)
        expected = np.einsum("ii->", np.asarray(a))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_outer_product(self):
        """Test outer product."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0])
        result = execute_dense_einsum("i,j->ij", a, b)
        expected = np.einsum("i,j->ij", np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_transpose(self):
        """Test matrix transpose."""
        a = jnp.arange(12.0).reshape(3, 4)
        result = execute_dense_einsum("ij->ji", a)
        expected = np.einsum("ij->ji", np.asarray(a))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_batch_matmul(self):
        """Test batched matrix multiplication."""
        a = jnp.ones((2, 3, 4))
        b = jnp.ones((2, 4, 5))
        result = execute_dense_einsum("bij,bjk->bik", a, b)
        expected = np.einsum("bij,bjk->bik", np.ones((2, 3, 4)), np.ones((2, 4, 5)))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_element_wise(self):
        """Test element-wise (Hadamard) product with contraction."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = execute_dense_einsum("ij,ij->", a, b)
        expected = np.einsum("ij,ij->", np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestEinsumAPI(unittest.TestCase):
    """Tests for the user-facing einsum() function."""

    def test_matmul_explicit(self):
        """Test matrix multiply with explicit output notation."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = einsum("ij,jk->ik", a, b)
        expected = np.einsum("ij,jk->ik", a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_matmul_implicit(self):
        """Test matrix multiply with implicit output (NumPy convention)."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = einsum("ij,jk", a, b)
        expected = np.einsum("ij,jk", a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_trace(self):
        """Test trace with scalar output."""
        a = np.eye(3)
        result = einsum("ii->", a)
        np.testing.assert_allclose(result, 3.0, rtol=1e-5)

    def test_diagonal(self):
        """Test diagonal extraction."""
        a = np.diag([1.0, 2.0, 3.0])
        result = einsum("ii->i", a)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_row_sum(self):
        """Test row-wise summation."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = einsum("ij->i", a)
        expected = np.array([6.0, 15.0])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_column_sum(self):
        """Test column-wise summation."""
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = einsum("ij->j", a)
        expected = np.array([9.0, 12.0])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_outer_product(self):
        """Test outer product via einsum."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0])
        result = einsum("i,j->ij", a, b)
        expected = np.outer(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_three_tensor_contraction(self):
        """Test multi-tensor contraction: ij,jk,kl->il."""
        a = np.random.default_rng(42).random((3, 4))
        b = np.random.default_rng(43).random((4, 5))
        c = np.random.default_rng(44).random((5, 2))
        result = einsum("ij,jk,kl->il", a, b, c)
        expected = np.einsum("ij,jk,kl->il", a, b, c)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_returns_jax_array(self):
        """Test that result is a JAX array, not NumPy."""
        a = np.ones((2, 2))
        b = np.ones((2, 2))
        result = einsum("ij,jk->ik", a, b)
        self.assertTrue(hasattr(result, "devices"), "Result should be a JAX array")

    def test_accepts_jax_arrays(self):
        """Test that JAX arrays are accepted as inputs."""
        a = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        b = jnp.array([[3.0, 4.0], [5.0, 6.0]])
        result = einsum("ij,jk->ik", a, b)
        expected = np.array([[3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_scalar_result(self):
        """Test that full contraction returns a scalar."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = einsum("ij,ij->", a, b)
        expected = np.einsum("ij,ij->", a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_validation_wrong_tensor_count(self):
        """Test that wrong tensor count raises ValueError."""
        with self.assertRaises(ValueError):
            einsum("ij,jk->ik", np.ones((2, 2)))

    def test_validation_dimension_mismatch(self):
        """Test that mismatched dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            einsum("ij,jk->ik", np.ones((2, 3)), np.ones((4, 5)))

    def test_validation_invalid_label(self):
        """Test that invalid output labels raise ValueError."""
        with self.assertRaises(ValueError):
            einsum("ij->z", np.ones((2, 2)))


class TestAnalyze(unittest.TestCase):
    """Tests for the analyze() function."""

    def test_matmul_analysis(self):
        """Test analysis of matrix multiplication."""
        a = np.ones((3, 4))
        b = np.ones((4, 5))
        result = analyze("ij,jk->ik", a, b)
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual("ij,jk->ik", result.einsum_string)
        self.assertEqual([(3, 4), (4, 5)], result.input_shapes)
        self.assertEqual((3, 5), result.output_shape)
        self.assertEqual(["j"], result.contracted_indices)
        self.assertEqual(["i", "k"], result.free_indices)
        self.assertEqual({"i": 3, "j": 4, "k": 5}, result.label_dimensions)

    def test_implicit_notation(self):
        """Test that analyze normalizes implicit notation."""
        a = np.ones((3, 4))
        b = np.ones((4, 5))
        result = analyze("ij,jk", a, b)
        self.assertEqual("ij,jk->ik", result.einsum_string)
        self.assertEqual((3, 5), result.output_shape)

    def test_trace_analysis(self):
        """Test analysis of trace operation."""
        a = np.ones((4, 4))
        result = analyze("ii->", a)
        self.assertEqual((), result.output_shape)
        self.assertEqual(["i"], result.contracted_indices)
        self.assertEqual([], result.free_indices)

    def test_transpose_analysis(self):
        """Test analysis of transpose operation."""
        a = np.ones((3, 4))
        result = analyze("ij->ji", a)
        self.assertEqual((4, 3), result.output_shape)
        self.assertEqual([], result.contracted_indices)
        self.assertEqual(["i", "j"], result.free_indices)

    def test_repr(self):
        """Test AnalysisResult repr is well-formed."""
        a = np.ones((2, 3))
        b = np.ones((3, 4))
        result = analyze("ij,jk->ik", a, b)
        r = repr(result)
        self.assertIn("ij,jk->ik", r)
        self.assertIn("contracted", r)


class TestBuildDependencyGraph(unittest.TestCase):
    """Tests for build_dependency_graph (Kahn's algorithm support)."""

    def test_single_node_no_deps(self):
        """A leaf node produces a graph with one entry and no edges."""
        t = BaseTensor("A", (4, 4))
        scheme = t.schemes[(4, 4)]
        graph = build_dependency_graph(scheme)
        self.assertEqual(1, len(graph))
        sid = id(scheme)
        self.assertEqual(0, len(graph[sid][1]))  # no incoming
        self.assertEqual(0, len(graph[sid][2]))  # no outgoing

    def test_binary_contraction_three_nodes(self):
        """A contraction of two inputs produces 3 nodes in the graph."""
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("C", (4, 4))

        lhs_scheme = a.schemes[(4, 4)]
        rhs_scheme = b.schemes[(4, 4)]
        out_scheme = out.schemes[(4, 4)]
        out_scheme.source = (lhs_scheme, rhs_scheme)

        graph = build_dependency_graph(out_scheme)
        self.assertEqual(3, len(graph))

    def test_edges_correct(self):
        """Verify incoming/outgoing edges are correct."""
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("C", (4, 4))

        lhs_scheme = a.schemes[(4, 4)]
        rhs_scheme = b.schemes[(4, 4)]
        out_scheme = out.schemes[(4, 4)]
        out_scheme.source = (lhs_scheme, rhs_scheme)

        graph = build_dependency_graph(out_scheme)
        out_id = id(out_scheme)
        # Output has 2 incoming edges
        self.assertEqual(2, len(graph[out_id][1]))
        # Output has 0 outgoing edges
        self.assertEqual(0, len(graph[out_id][2]))
        # Inputs have 0 incoming and 1 outgoing each
        for src in (lhs_scheme, rhs_scheme):
            sid = id(src)
            self.assertEqual(0, len(graph[sid][1]))
            self.assertEqual(1, len(graph[sid][2]))


class TestTopologicalSort(unittest.TestCase):
    """Tests for topological_sort (Kahn's algorithm)."""

    def test_single_node(self):
        """A single leaf node returns just that node."""
        t = BaseTensor("A", (4, 4))
        scheme = t.schemes[(4, 4)]
        ordered = topological_sort(scheme)
        self.assertEqual(1, len(ordered))
        self.assertIs(scheme, ordered[0])

    def test_binary_ordering(self):
        """Binary contraction: inputs come before output."""
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("C", (4, 4))

        lhs = a.schemes[(4, 4)]
        rhs = b.schemes[(4, 4)]
        out_scheme = out.schemes[(4, 4)]
        out_scheme.source = (lhs, rhs)

        ordered = topological_sort(out_scheme)
        self.assertEqual(3, len(ordered))
        # Output must be last
        self.assertIs(out_scheme, ordered[-1])
        # Both inputs must come before output
        out_idx = ordered.index(out_scheme)
        self.assertLess(ordered.index(lhs), out_idx)
        self.assertLess(ordered.index(rhs), out_idx)

    def test_chain_ordering(self):
        """Chain: A * B -> T1, T1 * C -> T2. All deps before dependents."""
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        c = BaseTensor("C", (4, 4))
        t1 = BaseTensor("T1", (4, 4))
        t2 = BaseTensor("T2", (4, 4))

        sa = a.schemes[(4, 4)]
        sb = b.schemes[(4, 4)]
        sc = c.schemes[(4, 4)]
        st1 = t1.schemes[(4, 4)]
        st2 = t2.schemes[(4, 4)]

        st1.source = (sa, sb)
        st2.source = (st1, sc)

        ordered = topological_sort(st2)
        self.assertEqual(5, len(ordered))
        # T2 must be last
        self.assertIs(st2, ordered[-1])
        # T1 before T2
        self.assertLess(ordered.index(st1), ordered.index(st2))
        # A and B before T1
        self.assertLess(ordered.index(sa), ordered.index(st1))
        self.assertLess(ordered.index(sb), ordered.index(st1))

    def test_no_duplicates(self):
        """Each node appears exactly once in the sorted output."""
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("C", (4, 4))

        lhs = a.schemes[(4, 4)]
        rhs = b.schemes[(4, 4)]
        out_scheme = out.schemes[(4, 4)]
        out_scheme.source = (lhs, rhs)

        ordered = topological_sort(out_scheme)
        ids = [id(s) for s in ordered]
        self.assertEqual(len(ids), len(set(ids)))


class TestExecutionEnginePlan(unittest.TestCase):
    """Tests for ExecutionEngine.execute_plan with ContractionPlan."""

    def _make_config(self):
        return CostModelConfig.from_device_type("cpu")

    def test_two_tensor_matmul(self):
        """execute_plan produces correct result for 2-tensor matmul."""
        config = self._make_config()
        a = BaseTensor("A", (3, 4))
        b = BaseTensor("B", (4, 5))
        plan = plan_contraction("ij,jk->ik", [a, b], config)

        a_data = jnp.array(np.random.default_rng(42).random((3, 4)).astype(np.float32))
        b_data = jnp.array(np.random.default_rng(43).random((4, 5)).astype(np.float32))

        engine = ExecutionEngine()
        result = engine.execute_plan(plan, [a_data, b_data])
        expected = np.einsum("ij,jk->ik", np.asarray(a_data), np.asarray(b_data))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_three_tensor_chain(self):
        """execute_plan produces correct result for 3-tensor chain."""
        config = self._make_config()
        a = BaseTensor("A", (3, 4))
        b = BaseTensor("B", (4, 5))
        c = BaseTensor("C", (5, 2))
        plan = plan_contraction("ij,jk,kl->il", [a, b, c], config)

        rng = np.random.default_rng(42)
        a_data = jnp.array(rng.random((3, 4)).astype(np.float32))
        b_data = jnp.array(rng.random((4, 5)).astype(np.float32))
        c_data = jnp.array(rng.random((5, 2)).astype(np.float32))

        engine = ExecutionEngine()
        result = engine.execute_plan(plan, [a_data, b_data, c_data])
        expected = np.einsum("ij,jk,kl->il",
                             np.asarray(a_data), np.asarray(b_data), np.asarray(c_data))
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_returns_jax_array(self):
        """execute_plan result is a JAX array."""
        config = self._make_config()
        a = BaseTensor("A", (2, 3))
        b = BaseTensor("B", (3, 2))
        plan = plan_contraction("ij,jk->ik", [a, b], config)

        a_data = jnp.ones((2, 3))
        b_data = jnp.ones((3, 2))

        engine = ExecutionEngine()
        result = engine.execute_plan(plan, [a_data, b_data])
        self.assertTrue(hasattr(result, "devices"), "Result should be a JAX array")

    def test_output_shape(self):
        """execute_plan output has correct shape."""
        config = self._make_config()
        a = BaseTensor("A", (3, 4))
        b = BaseTensor("B", (4, 5))
        plan = plan_contraction("ij,jk->ik", [a, b], config)

        a_data = jnp.ones((3, 4))
        b_data = jnp.ones((4, 5))

        engine = ExecutionEngine()
        result = engine.execute_plan(plan, [a_data, b_data])
        self.assertEqual((3, 5), result.shape)


class TestExecutionEngineSchemes(unittest.TestCase):
    """Tests for ExecutionEngine.execute_schemes with tiling DAG."""

    def test_simple_matmul_via_schemes(self):
        """execute_schemes produces correct matmul via scheme DAG."""
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("result", (4, 4))

        sa = a.schemes[(4, 4)]
        sb = b.schemes[(4, 4)]
        s_out = out.schemes[(4, 4)]
        s_out.source = (sa, sb)

        a_data = jnp.array(np.eye(4, dtype=np.float32))
        b_data = jnp.array(np.arange(16, dtype=np.float32).reshape(4, 4))

        engine = ExecutionEngine()
        result = engine.execute_schemes(
            s_out,
            data={"A": a_data, "B": b_data},
            formulas={"result": "ij,jk->ik"},
        )
        expected = np.einsum("ij,jk->ik", np.asarray(a_data), np.asarray(b_data))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_missing_formula_raises(self):
        """execute_schemes raises ValueError for missing formula."""
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("result", (4, 4))

        sa = a.schemes[(4, 4)]
        sb = b.schemes[(4, 4)]
        s_out = out.schemes[(4, 4)]
        s_out.source = (sa, sb)

        engine = ExecutionEngine()
        with self.assertRaises(ValueError):
            engine.execute_schemes(
                s_out,
                data={"A": jnp.ones((4, 4)), "B": jnp.ones((4, 4))},
                formulas={},  # missing formula
            )

    def test_missing_data_raises(self):
        """execute_schemes raises ValueError for missing input data."""
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("result", (4, 4))

        sa = a.schemes[(4, 4)]
        sb = b.schemes[(4, 4)]
        s_out = out.schemes[(4, 4)]
        s_out.source = (sa, sb)

        engine = ExecutionEngine()
        with self.assertRaises(ValueError):
            engine.execute_schemes(
                s_out,
                data={"A": jnp.ones((4, 4))},  # missing B
                formulas={"result": "ij,jk->ik"},
            )


class TestExecutionEngineSequence(unittest.TestCase):
    """Tests for ExecutionEngine.execute_sequence."""

    def test_matmul_sequence(self):
        """execute_sequence produces correct matmul."""
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("C", (4, 4))

        sa = a.schemes[(4, 4)]
        sb = b.schemes[(4, 4)]
        s_out = out.schemes[(4, 4)]
        s_out.source = (sa, sb)

        schemes = [sa, sb, s_out]
        a_data = jnp.eye(4, dtype=jnp.float32)
        b_data = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)

        engine = ExecutionEngine()
        result = engine.execute_sequence(schemes, [a_data, b_data], ["ij,jk->ik"])
        expected = np.einsum("ij,jk->ik", np.asarray(a_data), np.asarray(b_data))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_wrong_formula_count_raises(self):
        """execute_sequence raises ValueError for wrong formula count."""
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("C", (4, 4))

        sa = a.schemes[(4, 4)]
        sb = b.schemes[(4, 4)]
        s_out = out.schemes[(4, 4)]
        s_out.source = (sa, sb)

        schemes = [sa, sb, s_out]
        engine = ExecutionEngine()
        with self.assertRaises(ValueError):
            engine.execute_sequence(
                schemes,
                [jnp.ones((4, 4)), jnp.ones((4, 4))],
                ["ij,jk->ik", "extra"],  # too many formulas
            )


class TestPackageExecutionImports(unittest.TestCase):
    """Test that execution and API are importable from einjax."""

    def test_einsum_import(self):
        """Test that einsum is accessible from einjax."""
        import einjax
        self.assertTrue(callable(einjax.einsum))

    def test_analyze_import(self):
        """Test that analyze is accessible from einjax."""
        import einjax
        self.assertTrue(callable(einjax.analyze))

    def test_execute_dense_einsum_import(self):
        """Test that execute_dense_einsum is accessible from einjax."""
        import einjax
        self.assertTrue(callable(einjax.execute_dense_einsum))

    def test_execution_engine_import(self):
        """Test that ExecutionEngine is accessible from einjax."""
        import einjax
        self.assertIsNotNone(einjax.ExecutionEngine)

    def test_topological_sort_import(self):
        """Test that topological_sort is accessible from einjax."""
        import einjax
        self.assertTrue(callable(einjax.topological_sort))

    def test_build_dependency_graph_import(self):
        """Test that build_dependency_graph is accessible from einjax."""
        import einjax
        self.assertTrue(callable(einjax.build_dependency_graph))


class TestShardInputs(unittest.TestCase):
    """Tests for ExecutionEngine._shard_inputs (PRD 4.4)."""

    def test_no_mesh_returns_unchanged(self):
        """Without a mesh, _shard_inputs returns operands unchanged."""
        engine = ExecutionEngine(mesh=None)
        a = jnp.ones((8, 8))
        b = jnp.ones((8, 8))
        from einjax.tensor.base import BaseTensor

        sa = BaseTensor("A", (8, 8)).schemes[(8, 8)]
        sb = BaseTensor("B", (8, 8)).schemes[(8, 8)]

        result = engine._shard_inputs([a, b], [sa, sb])
        self.assertIs(result[0], a)
        self.assertIs(result[1], b)

    def test_with_mesh_returns_sharded_arrays(self):
        """With a mesh, _shard_inputs places arrays on the mesh."""
        import jax
        from einjax.sharding.mesh import create_mesh
        from einjax.tensor.base import BaseTensor

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)

        a = jnp.ones((8, 8))
        # tile_shape (4, 8): first dim sharded (UPPER), second local (LOWER)
        sa = TilingScheme(
            node=BaseTensor("A", (8, 8)),
            shape=(8, 8),
            tile_shape=(8, 8),
        )

        result = engine._shard_inputs([a], [sa])
        self.assertEqual(result[0].shape, (8, 8))
        # Verify array has a sharding attribute
        self.assertTrue(hasattr(result[0], "sharding"))

    def test_sharded_array_has_named_sharding(self):
        """Sharded array should have NamedSharding with correct spec."""
        import jax
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)

        a = jnp.ones((8, 8))
        sa = TilingScheme(
            node=BaseTensor("A", (8, 8)),
            shape=(8, 8),
            tile_shape=(8, 8),
        )

        result = engine._shard_inputs([a], [sa])
        sharding = result[0].sharding
        self.assertIsInstance(sharding, jax.sharding.NamedSharding)

    def test_preserves_data_values(self):
        """Sharding should not alter tensor values."""
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)

        a = jnp.arange(16.0).reshape(4, 4)
        sa = TilingScheme(
            node=BaseTensor("A", (4, 4)),
            shape=(4, 4),
            tile_shape=(4, 4),
        )

        result = engine._shard_inputs([a], [sa])
        np.testing.assert_allclose(result[0], a, rtol=1e-5)

    def test_multiple_operands(self):
        """_shard_inputs handles multiple operands correctly."""
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)

        a = jnp.ones((4, 4))
        b = jnp.ones((4, 4))
        sa = TilingScheme(
            node=BaseTensor("A", (4, 4)),
            shape=(4, 4),
            tile_shape=(4, 4),
        )
        sb = TilingScheme(
            node=BaseTensor("B", (4, 4)),
            shape=(4, 4),
            tile_shape=(4, 4),
        )

        result = engine._shard_inputs([a, b], [sa, sb])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (4, 4))
        self.assertEqual(result[1].shape, (4, 4))

    def test_mesh_axis_names_from_mesh(self):
        """Engine should derive mesh_axis_names from the mesh when not provided."""
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("data",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)

        self.assertEqual(engine.mesh_axis_names, ("data",))

    def test_mesh_axis_names_explicit(self):
        """Explicit mesh_axis_names override auto-detection."""
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh, mesh_axis_names=("custom",))

        self.assertEqual(engine.mesh_axis_names, ("custom",))

    def test_no_mesh_default_axis_names(self):
        """Without a mesh, default axis names are used."""
        engine = ExecutionEngine(mesh=None)
        self.assertEqual(engine.mesh_axis_names, ("x", "y", "z", "w"))


class TestShardedExecutePlan(unittest.TestCase):
    """Tests for execute_plan with mesh-based input sharding."""

    def test_execute_plan_with_mesh_correctness(self):
        """execute_plan with mesh produces correct results."""
        from einjax.sharding.mesh import create_mesh

        config = CostModelConfig.from_device_type("cpu")
        a = BaseTensor("A", (3, 4))
        b = BaseTensor("B", (4, 5))
        plan = plan_contraction("ij,jk->ik", [a, b], config)

        a_data = jnp.array(
            np.random.default_rng(42).random((3, 4)).astype(np.float32)
        )
        b_data = jnp.array(
            np.random.default_rng(43).random((4, 5)).astype(np.float32)
        )

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)
        result = engine.execute_plan(plan, [a_data, b_data])
        expected = np.einsum(
            "ij,jk->ik", np.asarray(a_data), np.asarray(b_data)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestShardedExecuteSchemes(unittest.TestCase):
    """Tests for execute_schemes with mesh-based input sharding."""

    def test_execute_schemes_with_mesh_correctness(self):
        """execute_schemes with mesh produces correct results."""
        from einjax.sharding.mesh import create_mesh

        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("result", (4, 4))

        sa = a.schemes[(4, 4)]
        sb = b.schemes[(4, 4)]
        s_out = out.schemes[(4, 4)]
        s_out.source = (sa, sb)

        a_data = jnp.array(np.eye(4, dtype=np.float32))
        b_data = jnp.array(np.arange(16, dtype=np.float32).reshape(4, 4))

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)
        result = engine.execute_schemes(
            s_out,
            data={"A": a_data, "B": b_data},
            formulas={"result": "ij,jk->ik"},
        )
        expected = np.einsum(
            "ij,jk->ik", np.asarray(a_data), np.asarray(b_data)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestShardedExecuteSequence(unittest.TestCase):
    """Tests for execute_sequence with mesh-based input sharding."""

    def test_execute_sequence_with_mesh_correctness(self):
        """execute_sequence with mesh produces correct results."""
        from einjax.sharding.mesh import create_mesh

        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("C", (4, 4))

        sa = a.schemes[(4, 4)]
        sb = b.schemes[(4, 4)]
        s_out = out.schemes[(4, 4)]
        s_out.source = (sa, sb)

        schemes = [sa, sb, s_out]
        a_data = jnp.eye(4, dtype=jnp.float32)
        b_data = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)
        result = engine.execute_sequence(
            schemes, [a_data, b_data], ["ij,jk->ik"]
        )
        expected = np.einsum(
            "ij,jk->ik", np.asarray(a_data), np.asarray(b_data)
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestExecuteShardedEinsum(unittest.TestCase):
    """Tests for execute_sharded_einsum (PRD 4.5)."""

    def test_no_mesh_falls_back_to_dense(self):
        """Without a mesh, execute_sharded_einsum behaves like jnp.einsum."""
        from einjax.execution.dense_kernels import execute_sharded_einsum

        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = execute_sharded_einsum("ij,jk->ik", a, b)
        expected = np.einsum("ij,jk->ik", np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_with_mesh_matmul_correct(self):
        """Sharded matmul with mesh produces correct result."""
        from einjax.execution.dense_kernels import execute_sharded_einsum
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        a = jnp.array(np.random.default_rng(42).random((4, 4)).astype(np.float32))
        b = jnp.array(np.random.default_rng(43).random((4, 4)).astype(np.float32))

        # All dims local (tile_shape == shape), so specs are all None
        result = execute_sharded_einsum(
            "ij,jk->ik", a, b,
            mesh=mesh,
            in_specs=[(None, None), (None, None)],
            out_spec=(None, None),
        )
        expected = np.einsum("ij,jk->ik", np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_with_mesh_returns_sharded_array(self):
        """Result from sharded einsum has NamedSharding."""
        import jax
        from einjax.execution.dense_kernels import execute_sharded_einsum
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        a = jnp.ones((4, 4))
        b = jnp.ones((4, 4))

        result = execute_sharded_einsum(
            "ij,jk->ik", a, b,
            mesh=mesh,
            in_specs=[(None, None), (None, None)],
            out_spec=(None, None),
        )
        self.assertIsInstance(result.sharding, jax.sharding.NamedSharding)

    def test_sharded_batch_matmul(self):
        """Sharded batched matmul produces correct result."""
        from einjax.execution.dense_kernels import execute_sharded_einsum
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        a = jnp.ones((2, 3, 4))
        b = jnp.ones((2, 4, 5))

        result = execute_sharded_einsum(
            "bij,bjk->bik", a, b,
            mesh=mesh,
            in_specs=[(None, None, None), (None, None, None)],
            out_spec=(None, None, None),
        )
        expected = np.einsum("bij,bjk->bik", np.ones((2, 3, 4)), np.ones((2, 4, 5)))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_sharded_trace(self):
        """Sharded trace operation produces correct result."""
        from einjax.execution.dense_kernels import execute_sharded_einsum
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        result = execute_sharded_einsum(
            "ii->", a,
            mesh=mesh,
            in_specs=[(None, None)],
            out_spec=(),
        )
        expected = np.einsum("ii->", np.asarray(a))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_sharded_outer_product(self):
        """Sharded outer product produces correct result."""
        from einjax.execution.dense_kernels import execute_sharded_einsum
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([4.0, 5.0])

        result = execute_sharded_einsum(
            "i,j->ij", a, b,
            mesh=mesh,
            in_specs=[(None,), (None,)],
            out_spec=(None, None),
        )
        expected = np.einsum("i,j->ij", np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestEngineExecuteEinsum(unittest.TestCase):
    """Tests for ExecutionEngine._execute_einsum (PRD 4.5)."""

    def test_no_mesh_uses_dense(self):
        """Without mesh, _execute_einsum uses plain dense einsum."""
        engine = ExecutionEngine(mesh=None)
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        result = engine._execute_einsum("ij,jk->ik", [a, b])
        expected = np.einsum("ij,jk->ik", np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_no_schemes_uses_dense(self):
        """With mesh but no schemes, _execute_einsum falls back to dense."""
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        result = engine._execute_einsum("ij,jk->ik", [a, b], source_schemes=None)
        expected = np.einsum("ij,jk->ik", np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_with_mesh_and_schemes_uses_sharded(self):
        """With mesh and schemes, _execute_einsum uses sharded execution."""
        import jax
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)

        a = jnp.array(np.random.default_rng(42).random((4, 4)).astype(np.float32))
        b = jnp.array(np.random.default_rng(43).random((4, 4)).astype(np.float32))

        sa = TilingScheme(
            node=BaseTensor("A", (4, 4)), shape=(4, 4), tile_shape=(4, 4),
        )
        sb = TilingScheme(
            node=BaseTensor("B", (4, 4)), shape=(4, 4), tile_shape=(4, 4),
        )
        s_out = TilingScheme(
            node=BaseTensor("C", (4, 4)), shape=(4, 4), tile_shape=(4, 4),
        )

        result = engine._execute_einsum(
            "ij,jk->ik", [a, b],
            source_schemes=[sa, sb],
            output_scheme=s_out,
        )
        expected = np.einsum("ij,jk->ik", np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        # Verify the result has NamedSharding
        self.assertIsInstance(result.sharding, jax.sharding.NamedSharding)


class TestDeriveOutputSpec(unittest.TestCase):
    """Tests for ExecutionEngine._derive_output_spec_from_formula."""

    def test_matmul_all_local(self):
        """All-local matmul derives all-None output spec."""
        engine = ExecutionEngine(mesh=None)
        out_spec = engine._derive_output_spec_from_formula(
            "ij,jk->ik",
            [(None, None), (None, None)],
            [],
        )
        self.assertEqual(out_spec, (None, None))

    def test_matmul_sharded_i(self):
        """Sharded first dim propagates to output."""
        engine = ExecutionEngine(mesh=None)
        out_spec = engine._derive_output_spec_from_formula(
            "ij,jk->ik",
            [("x", None), (None, None)],
            [],
        )
        self.assertEqual(out_spec, ("x", None))

    def test_contracted_dim_not_in_output(self):
        """Contracted dimension (j) is not in output spec."""
        engine = ExecutionEngine(mesh=None)
        out_spec = engine._derive_output_spec_from_formula(
            "ij,jk->ik",
            [(None, "x"), ("x", None)],
            [],
        )
        # i is None (first dim of first operand), k is None (second dim of second operand)
        self.assertEqual(out_spec, (None, None))

    def test_batch_dim_sharded(self):
        """Batch dimension sharding propagates to output."""
        engine = ExecutionEngine(mesh=None)
        out_spec = engine._derive_output_spec_from_formula(
            "bij,bjk->bik",
            [("x", None, None), ("x", None, None)],
            [],
        )
        self.assertEqual(out_spec, ("x", None, None))

    def test_trace_scalar_output(self):
        """Trace with scalar output produces empty spec."""
        engine = ExecutionEngine(mesh=None)
        out_spec = engine._derive_output_spec_from_formula(
            "ii->",
            [(None, None)],
            [],
        )
        self.assertEqual(out_spec, ())


class TestShardedExecutePlanWithSharding(unittest.TestCase):
    """Tests for execute_plan with full sharded dense execution (PRD 4.5)."""

    def test_plan_with_mesh_uses_sharded_einsum(self):
        """execute_plan with mesh invokes sharded einsum and produces correct result."""
        from einjax.sharding.mesh import create_mesh

        config = CostModelConfig.from_device_type("cpu")
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        plan = plan_contraction("ij,jk->ik", [a, b], config)

        a_data = jnp.array(np.random.default_rng(42).random((4, 4)).astype(np.float32))
        b_data = jnp.array(np.random.default_rng(43).random((4, 4)).astype(np.float32))

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)
        result = engine.execute_plan(plan, [a_data, b_data])
        expected = np.einsum("ij,jk->ik", np.asarray(a_data), np.asarray(b_data))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_schemes_with_mesh_uses_sharded_einsum(self):
        """execute_schemes with mesh invokes sharded einsum and produces correct result."""
        import jax
        from einjax.sharding.mesh import create_mesh

        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("result", (4, 4))

        sa = a.schemes[(4, 4)]
        sb = b.schemes[(4, 4)]
        s_out = out.schemes[(4, 4)]
        s_out.source = (sa, sb)

        a_data = jnp.array(np.eye(4, dtype=np.float32))
        b_data = jnp.array(np.arange(16, dtype=np.float32).reshape(4, 4))

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)
        result = engine.execute_schemes(
            s_out,
            data={"A": a_data, "B": b_data},
            formulas={"result": "ij,jk->ik"},
        )
        expected = np.einsum("ij,jk->ik", np.asarray(a_data), np.asarray(b_data))
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        # Result should have NamedSharding since we're running with a mesh
        self.assertIsInstance(result.sharding, jax.sharding.NamedSharding)

    def test_sequence_with_mesh_uses_sharded_einsum(self):
        """execute_sequence with mesh invokes sharded einsum and produces correct result."""
        import jax
        from einjax.sharding.mesh import create_mesh

        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("C", (4, 4))

        sa = a.schemes[(4, 4)]
        sb = b.schemes[(4, 4)]
        s_out = out.schemes[(4, 4)]
        s_out.source = (sa, sb)

        schemes = [sa, sb, s_out]
        a_data = jnp.eye(4, dtype=jnp.float32)
        b_data = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)

        mesh = create_mesh(axis_names=("x",), num_devices=1)
        engine = ExecutionEngine(mesh=mesh)
        result = engine.execute_sequence(
            schemes, [a_data, b_data], ["ij,jk->ik"]
        )
        expected = np.einsum("ij,jk->ik", np.asarray(a_data), np.asarray(b_data))
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        self.assertIsInstance(result.sharding, jax.sharding.NamedSharding)


class TestPackageShardedEinsumImports(unittest.TestCase):
    """Test that execute_sharded_einsum is importable from einjax."""

    def test_execute_sharded_einsum_import(self):
        """Test that execute_sharded_einsum is accessible from einjax."""
        import einjax
        self.assertTrue(callable(einjax.execute_sharded_einsum))


class TestOutputShardingPolicy(unittest.TestCase):
    """Tests for output sharding policy (PRD 4.9)."""

    def test_default_output_sharded_with_mesh(self):
        """By default, output of a sharded execution has NamedSharding."""
        import jax
        from einjax.sharding.mesh import create_mesh

        num = len(jax.devices())
        if num < 2:
            self.skipTest("Need >= 2 devices for sharded test")

        mesh = create_mesh(axis_names=("x",), num_devices=num)
        engine = ExecutionEngine(mesh=mesh)

        config = CostModelConfig.from_device_type("cpu")
        a = BaseTensor("A", (num * 4, num * 4))
        b = BaseTensor("B", (num * 4, num * 4))
        plan = plan_contraction("ij,jk->ik", [a, b], config)

        size = num * 4
        a_data = jnp.ones((size, size))
        b_data = jnp.ones((size, size))

        result = engine.execute_plan(plan, [a_data, b_data])
        self.assertIsInstance(result.sharding, jax.sharding.NamedSharding)

    def test_gather_output_plan(self):
        """With gather_output=True, execute_plan result is not NamedSharding."""
        import jax
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=len(jax.devices()))
        engine = ExecutionEngine(mesh=mesh)

        config = CostModelConfig.from_device_type("cpu")
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        plan = plan_contraction("ij,jk->ik", [a, b], config)

        a_data = jnp.array(np.eye(4, dtype=np.float32))
        b_data = jnp.array(np.arange(16, dtype=np.float32).reshape(4, 4))

        result = engine.execute_plan(plan, [a_data, b_data], gather_output=True)
        expected = np.einsum("ij,jk->ik", np.eye(4), np.arange(16).reshape(4, 4))
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        # After gather, result should not have NamedSharding
        self.assertNotIsInstance(
            getattr(result, "sharding", None),
            jax.sharding.NamedSharding,
        )

    def test_gather_output_schemes(self):
        """With gather_output=True, execute_schemes result is gathered."""
        import jax
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=len(jax.devices()))
        engine = ExecutionEngine(mesh=mesh)

        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("result", (4, 4))

        sa = a.schemes[(4, 4)]
        sb = b.schemes[(4, 4)]
        s_out = out.schemes[(4, 4)]
        s_out.source = (sa, sb)

        a_data = jnp.array(np.eye(4, dtype=np.float32))
        b_data = jnp.array(np.arange(16, dtype=np.float32).reshape(4, 4))

        result = engine.execute_schemes(
            s_out,
            data={"A": a_data, "B": b_data},
            formulas={"result": "ij,jk->ik"},
            gather_output=True,
        )
        expected = np.einsum("ij,jk->ik", np.eye(4), np.arange(16).reshape(4, 4))
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        self.assertNotIsInstance(
            getattr(result, "sharding", None),
            jax.sharding.NamedSharding,
        )

    def test_gather_output_sequence(self):
        """With gather_output=True, execute_sequence result is gathered."""
        import jax
        from einjax.sharding.mesh import create_mesh

        mesh = create_mesh(axis_names=("x",), num_devices=len(jax.devices()))
        engine = ExecutionEngine(mesh=mesh)

        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        out = BaseTensor("C", (4, 4))

        sa = a.schemes[(4, 4)]
        sb = b.schemes[(4, 4)]
        s_out = out.schemes[(4, 4)]
        s_out.source = (sa, sb)

        a_data = jnp.eye(4, dtype=jnp.float32)
        b_data = jnp.arange(16, dtype=jnp.float32).reshape(4, 4)

        result = engine.execute_sequence(
            [sa, sb, s_out], [a_data, b_data], ["ij,jk->ik"],
            gather_output=True,
        )
        expected = np.einsum("ij,jk->ik", np.eye(4), np.arange(16).reshape(4, 4))
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        self.assertNotIsInstance(
            getattr(result, "sharding", None),
            jax.sharding.NamedSharding,
        )

    def test_gather_false_no_mesh_is_noop(self):
        """gather_output=False with no mesh returns normal result."""
        engine = ExecutionEngine(mesh=None)

        config = CostModelConfig.from_device_type("cpu")
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        plan = plan_contraction("ij,jk->ik", [a, b], config)

        a_data = jnp.array(np.eye(4, dtype=np.float32))
        b_data = jnp.array(np.arange(16, dtype=np.float32).reshape(4, 4))

        result = engine.execute_plan(plan, [a_data, b_data], gather_output=False)
        expected = np.einsum("ij,jk->ik", np.eye(4), np.arange(16).reshape(4, 4))
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_gather_true_no_mesh_is_noop(self):
        """gather_output=True with no mesh still works (no-op)."""
        engine = ExecutionEngine(mesh=None)

        config = CostModelConfig.from_device_type("cpu")
        a = BaseTensor("A", (4, 4))
        b = BaseTensor("B", (4, 4))
        plan = plan_contraction("ij,jk->ik", [a, b], config)

        a_data = jnp.array(np.eye(4, dtype=np.float32))
        b_data = jnp.array(np.arange(16, dtype=np.float32).reshape(4, 4))

        result = engine.execute_plan(plan, [a_data, b_data], gather_output=True)
        expected = np.einsum("ij,jk->ik", np.eye(4), np.arange(16).reshape(4, 4))
        np.testing.assert_allclose(result, expected, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
