"""
Tests for the DPOptimizer and ReductionInfo.

Tests DP optimizer correctness: scheme enumeration, cost scoring,
optimal scheme selection, and plan tracing for binary tensor contractions.
"""

from __future__ import annotations

import unittest
from math import prod

import numpy as np

from einjax.core.types import TilingScheme
from einjax.optimizer.cost_model import CostModelConfig
from einjax.optimizer.dp import DPOptimizer, ReductionInfo, infer_reduction_info
from einjax.tensor.base import BaseTensor, IndexedTerm
from einjax.tensor.dense import DenseTensor


class TestInferReductionInfo(unittest.TestCase):
    """Test infer_reduction_info for various contraction patterns."""

    def test_matmul(self):
        """ij,jk->ik: j is join key, i and k are aggregation keys."""
        A = DenseTensor("A", np.ones((3, 4)))
        B = DenseTensor("B", np.ones((4, 5)))
        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")

        info = infer_reduction_info("ik", lhs, rhs)

        self.assertEqual(info.output_shape, (3, 5))
        self.assertEqual(len(info.join_keys), 1)
        # Join key connects A dim 1 to B dim 0 (both are 'j')
        self.assertEqual(info.join_keys[0][0], (A, 1))
        self.assertEqual(info.join_keys[0][1], (B, 0))
        self.assertEqual(len(info.aggregation_keys), 2)
        self.assertEqual(info.lhs_key_indices, (1,))
        self.assertEqual(info.rhs_key_indices, (0,))

    def test_outer_product(self):
        """i,j->ij: no join keys, both dims are aggregation keys."""
        A = DenseTensor("A", np.ones((3,)))
        B = DenseTensor("B", np.ones((4,)))
        lhs = IndexedTerm(A, "i")
        rhs = IndexedTerm(B, "j")

        info = infer_reduction_info("ij", lhs, rhs)

        self.assertEqual(info.output_shape, (3, 4))
        self.assertEqual(len(info.join_keys), 0)
        self.assertEqual(len(info.aggregation_keys), 2)
        self.assertEqual(info.lhs_key_indices, ())
        self.assertEqual(info.rhs_key_indices, ())

    def test_dot_product(self):
        """i,i->: i is join key, no aggregation keys (scalar output)."""
        A = DenseTensor("A", np.ones((4,)))
        B = DenseTensor("B", np.ones((4,)))
        lhs = IndexedTerm(A, "i")
        rhs = IndexedTerm(B, "i")

        info = infer_reduction_info("", lhs, rhs)

        self.assertEqual(info.output_shape, ())
        self.assertEqual(len(info.join_keys), 1)
        self.assertEqual(len(info.aggregation_keys), 0)

    def test_batch_matmul(self):
        """bij,bjk->bik: j is join key, b/i/k are aggregation keys."""
        A = DenseTensor("A", np.ones((2, 3, 4)))
        B = DenseTensor("B", np.ones((2, 4, 5)))
        lhs = IndexedTerm(A, "bij")
        rhs = IndexedTerm(B, "bjk")

        info = infer_reduction_info("bik", lhs, rhs)

        self.assertEqual(info.output_shape, (2, 3, 5))
        self.assertEqual(len(info.join_keys), 2)  # b and j are shared
        self.assertEqual(len(info.aggregation_keys), 3)

    def test_elementwise(self):
        """ij,ij->ij: i and j are join keys, i and j are aggregation keys."""
        A = DenseTensor("A", np.ones((3, 4)))
        B = DenseTensor("B", np.ones((3, 4)))
        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "ij")

        info = infer_reduction_info("ij", lhs, rhs)

        self.assertEqual(info.output_shape, (3, 4))
        self.assertEqual(len(info.join_keys), 2)  # both i and j
        self.assertEqual(len(info.aggregation_keys), 2)


class TestDPOptimizerBasic(unittest.TestCase):
    """Test DPOptimizer on simple contractions."""

    def setUp(self):
        self.config = CostModelConfig.from_device_type("cpu", 1)
        self.optimizer = DPOptimizer(self.config)

    def test_matmul_produces_finite_costs(self):
        """DP should produce at least one scheme with finite cost."""
        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))
        C = DenseTensor("C", np.zeros((4, 4)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        schemes = self.optimizer.optimize_reduction(C, info)

        # At least one scheme should have finite cost
        finite_schemes = [s for s in schemes.values() if s.cost < float("inf")]
        self.assertGreater(len(finite_schemes), 0)

    def test_matmul_best_scheme_exists(self):
        """get_best_scheme should return a valid scheme."""
        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))
        C = DenseTensor("C", np.zeros((4, 4)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        self.optimizer.optimize_reduction(C, info)
        best = self.optimizer.get_best_scheme(C)

        self.assertIsNotNone(best)
        self.assertLess(best.cost, float("inf"))
        self.assertLess(best.accumulated_cost, float("inf"))
        self.assertGreater(best.cost, 0.0)

    def test_matmul_scheme_has_source(self):
        """Best scheme should track parent (lhs, rhs) sources."""
        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))
        C = DenseTensor("C", np.zeros((4, 4)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        self.optimizer.optimize_reduction(C, info)
        best = self.optimizer.get_best_scheme(C)

        self.assertEqual(len(best.source), 2)
        self.assertIs(best.source[0].node, A)
        self.assertIs(best.source[1].node, B)

    def test_matmul_dependencies_tracked(self):
        """Best scheme dependencies should include lhs and rhs schemes."""
        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))
        C = DenseTensor("C", np.zeros((4, 4)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        self.optimizer.optimize_reduction(C, info)
        best = self.optimizer.get_best_scheme(C)

        self.assertIn(best.source[0], best.dependencies)
        self.assertIn(best.source[1], best.dependencies)

    def test_outer_product(self):
        """Outer product i,j->ij should produce finite costs."""
        A = DenseTensor("A", np.ones((3,)))
        B = DenseTensor("B", np.ones((4,)))
        C = DenseTensor("C", np.zeros((3, 4)))

        lhs = IndexedTerm(A, "i")
        rhs = IndexedTerm(B, "j")
        info = infer_reduction_info("ij", lhs, rhs)

        self.optimizer.optimize_reduction(C, info)
        best = self.optimizer.get_best_scheme(C)

        self.assertIsNotNone(best)
        self.assertLess(best.cost, float("inf"))


class TestDPOptimizerCostRanking(unittest.TestCase):
    """Test that the optimizer correctly ranks tiling choices."""

    def test_faster_hardware_lower_cost(self):
        """Same contraction should have lower cost on faster hardware."""
        A = DenseTensor("A", np.ones((8, 8)))
        B = DenseTensor("B", np.ones((8, 8)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        # CPU
        C_cpu = DenseTensor("C_cpu", np.zeros((8, 8)))
        cpu_opt = DPOptimizer(CostModelConfig.from_device_type("cpu", 1))
        cpu_opt.optimize_reduction(C_cpu, info)
        cpu_best = cpu_opt.get_best_scheme(C_cpu)

        # GPU
        C_gpu = DenseTensor("C_gpu", np.zeros((8, 8)))
        gpu_opt = DPOptimizer(CostModelConfig.from_device_type("gpu:a100", 1))
        gpu_opt.optimize_reduction(C_gpu, info)
        gpu_best = gpu_opt.get_best_scheme(C_gpu)

        self.assertGreater(cpu_best.accumulated_cost, gpu_best.accumulated_cost)

    def test_sparse_lower_cost_than_dense(self):
        """Sparse matrix should have lower cost (fewer non-zero tiles)."""
        # Dense matrix: all ones
        dense_data = np.ones((6, 6))
        A_dense = DenseTensor("A_dense", dense_data)

        # Sparse matrix: only diagonal
        sparse_data = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        A_sparse = DenseTensor("A_sparse", sparse_data)

        B = DenseTensor("B", np.ones((6, 6)))

        config = CostModelConfig.from_device_type("cpu", 1)

        # Dense path
        lhs_d = IndexedTerm(A_dense, "ij")
        rhs_d = IndexedTerm(B, "jk")
        info_d = infer_reduction_info("ik", lhs_d, rhs_d)
        C_dense = DenseTensor("C_dense", np.zeros((6, 6)))
        opt_d = DPOptimizer(config)
        opt_d.optimize_reduction(C_dense, info_d)
        dense_best = opt_d.get_best_scheme(C_dense)

        # Sparse path
        lhs_s = IndexedTerm(A_sparse, "ij")
        rhs_s = IndexedTerm(B, "jk")
        info_s = infer_reduction_info("ik", lhs_s, rhs_s)
        C_sparse = DenseTensor("C_sparse", np.zeros((6, 6)))
        opt_s = DPOptimizer(config)
        opt_s.optimize_reduction(C_sparse, info_s)
        sparse_best = opt_s.get_best_scheme(C_sparse)

        # Sparse input should lead to lower or equal cost
        self.assertLessEqual(
            sparse_best.accumulated_cost, dense_best.accumulated_cost
        )

    def test_accumulated_cost_includes_parents(self):
        """Accumulated cost should be >= the operation's own cost."""
        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))
        C = DenseTensor("C", np.zeros((4, 4)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        opt = DPOptimizer(CostModelConfig.from_device_type("cpu", 1))
        opt.optimize_reduction(C, info)
        best = opt.get_best_scheme(C)

        # Accumulated cost >= operation cost since parents have cost >= 0
        self.assertGreaterEqual(best.accumulated_cost, best.cost)


class TestDPOptimizerPlan(unittest.TestCase):
    """Test plan tracing and optimal plan extraction."""

    def setUp(self):
        self.config = CostModelConfig.from_device_type("cpu", 1)
        self.optimizer = DPOptimizer(self.config)

    def test_plan_nonempty(self):
        """Plan should contain at least the output + input schemes."""
        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))
        C = DenseTensor("C", np.zeros((4, 4)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        self.optimizer.optimize_reduction(C, info)
        plan = self.optimizer.get_optimal_plan(C)

        # Plan should have 3 entries: lhs scheme, rhs scheme, output scheme
        self.assertEqual(len(plan), 3)

    def test_plan_order(self):
        """Plan should end with the output scheme (dependencies first)."""
        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))
        C = DenseTensor("C", np.zeros((4, 4)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        self.optimizer.optimize_reduction(C, info)
        plan = self.optimizer.get_optimal_plan(C)

        # Last entry should be the output scheme
        best = self.optimizer.get_best_scheme(C)
        self.assertIs(plan[-1], best)

        # First two should be input schemes (lhs and rhs, in some order)
        input_nodes = {plan[0].node, plan[1].node}
        self.assertIn(A, input_nodes)
        self.assertIn(B, input_nodes)

    def test_plan_empty_for_unoptimized(self):
        """Plan should be empty if no scheme has finite cost."""
        # Create tensor but don't optimize — all costs remain at default 0
        # which is finite, so let's test with a manually created tensor
        A = BaseTensor("A", (4, 4))
        for scheme in A.schemes.values():
            scheme.cost = float("inf")
            scheme.accumulated_cost = float("inf")

        plan = self.optimizer.get_optimal_plan(A)
        self.assertEqual(len(plan), 0)


class TestDPOptimizerSchemeCompatibility(unittest.TestCase):
    """Test that DP correctly matches compatible scheme pairs."""

    def test_compatible_join_keys(self):
        """Only schemes with matching join key tile sizes should pair."""
        # A (4,6) × B (6,8) -> C (4,8) with j=6 as join key
        A = DenseTensor("A", np.ones((4, 6)))
        B = DenseTensor("B", np.ones((6, 8)))
        C = DenseTensor("C", np.zeros((4, 8)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        opt = DPOptimizer(CostModelConfig.from_device_type("cpu", 1))
        opt.optimize_reduction(C, info)
        best = opt.get_best_scheme(C)

        # The join key dimension (j=dim 1 of A, dim 0 of B) must have
        # the same tile size in both source schemes
        lhs_scheme, rhs_scheme = best.source
        self.assertEqual(lhs_scheme.tile_shape[1], rhs_scheme.tile_shape[0])

    def test_nonsquare_matmul(self):
        """DP should work for non-square matrices."""
        A = DenseTensor("A", np.ones((3, 6)))
        B = DenseTensor("B", np.ones((6, 2)))
        C = DenseTensor("C", np.zeros((3, 2)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        opt = DPOptimizer(CostModelConfig.from_device_type("cpu", 1))
        opt.optimize_reduction(C, info)
        best = opt.get_best_scheme(C)

        self.assertIsNotNone(best)
        self.assertLess(best.cost, float("inf"))

    def test_prime_dimension(self):
        """Prime-sized dimensions (only factors 1 and p) should work."""
        A = DenseTensor("A", np.ones((5, 7)))
        B = DenseTensor("B", np.ones((7, 3)))
        C = DenseTensor("C", np.zeros((5, 3)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        opt = DPOptimizer(CostModelConfig.from_device_type("cpu", 1))
        opt.optimize_reduction(C, info)
        best = opt.get_best_scheme(C)

        self.assertIsNotNone(best)
        # With prime dims, tile_shape must be 1 or the dim itself
        for ts, s in zip(best.tile_shape, C.shape):
            self.assertTrue(ts == 1 or ts == s)


class TestDPOptimizerMetrics(unittest.TestCase):
    """Test that cost metrics are correctly populated."""

    def setUp(self):
        self.config = CostModelConfig.from_device_type("cpu", 1)
        self.optimizer = DPOptimizer(self.config)

    def test_comm_positive(self):
        """Communication cost should be positive for contractions."""
        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))
        C = DenseTensor("C", np.zeros((4, 4)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        self.optimizer.optimize_reduction(C, info)
        best = self.optimizer.get_best_scheme(C)

        self.assertGreater(best.comm, 0.0)

    def test_flops_positive(self):
        """FLOPs should be positive for contractions."""
        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))
        C = DenseTensor("C", np.zeros((4, 4)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        self.optimizer.optimize_reduction(C, info)
        best = self.optimizer.get_best_scheme(C)

        self.assertGreater(best.flops, 0.0)

    def test_accumulated_metrics(self):
        """Accumulated comm/flops should be >= local comm/flops."""
        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))
        C = DenseTensor("C", np.zeros((4, 4)))

        lhs = IndexedTerm(A, "ij")
        rhs = IndexedTerm(B, "jk")
        info = infer_reduction_info("ik", lhs, rhs)

        self.optimizer.optimize_reduction(C, info)
        best = self.optimizer.get_best_scheme(C)

        self.assertGreaterEqual(best.accumulated_comm, best.comm)
        self.assertGreaterEqual(best.accumulated_flops, best.flops)


class TestPackageOptimizerImports(unittest.TestCase):
    """Verify optimizer exports from the einjax package."""

    def test_import_from_optimizer(self):
        from einjax.optimizer import DPOptimizer, infer_reduction_info, ReductionInfo
        self.assertIsNotNone(DPOptimizer)
        self.assertIsNotNone(infer_reduction_info)
        self.assertIsNotNone(ReductionInfo)

    def test_import_from_top_level(self):
        import einjax
        self.assertTrue(hasattr(einjax, "DPOptimizer"))
        self.assertTrue(hasattr(einjax, "infer_reduction_info"))
        self.assertTrue(hasattr(einjax, "ReductionInfo"))


# =========================================================================
# Contraction Path Tests
# =========================================================================


class TestGetContractionOrder(unittest.TestCase):
    """Test opt_einsum-based contraction ordering."""

    def test_two_tensor_single_step(self):
        """Two-tensor contraction should produce exactly one step."""
        from einjax.optimizer.contraction_path import get_contraction_order

        steps = get_contraction_order("ij,jk->ik", [(3, 4), (4, 5)])
        self.assertEqual(len(steps), 1)
        indices, formula = steps[0]
        self.assertEqual(len(indices), 2)
        self.assertIn("->", formula)

    def test_three_tensor_two_steps(self):
        """Three-tensor contraction should produce two binary steps."""
        from einjax.optimizer.contraction_path import get_contraction_order

        steps = get_contraction_order("ij,jk,kl->il", [(3, 4), (4, 5), (5, 6)])
        self.assertEqual(len(steps), 2)
        for indices, formula in steps:
            self.assertEqual(len(indices), 2)
            self.assertIn("->", formula)

    def test_four_tensor_three_steps(self):
        """Four-tensor contraction should produce three binary steps."""
        from einjax.optimizer.contraction_path import get_contraction_order

        steps = get_contraction_order(
            "ij,jk,kl,lm->im", [(2, 3), (3, 4), (4, 5), (5, 6)]
        )
        self.assertEqual(len(steps), 3)

    def test_formula_labels_valid(self):
        """Each step formula should have valid input and output labels."""
        from einjax.optimizer.contraction_path import get_contraction_order

        steps = get_contraction_order("ij,jk,kl->il", [(3, 4), (4, 5), (5, 6)])
        for _, formula in steps:
            parts = formula.split("->")
            self.assertEqual(len(parts), 2)
            inputs = parts[0].split(",")
            self.assertEqual(len(inputs), 2)
            # All output labels should appear in at least one input
            for label in parts[1]:
                self.assertTrue(
                    label in inputs[0] or label in inputs[1],
                    f"Output label '{label}' not in inputs of '{formula}'",
                )

    def test_implicit_notation(self):
        """Implicit notation (no ->) should be handled."""
        from einjax.optimizer.contraction_path import get_contraction_order

        steps = get_contraction_order("ij,jk,kl", [(3, 4), (4, 5), (5, 6)])
        self.assertEqual(len(steps), 2)


class TestPlanContraction(unittest.TestCase):
    """Test plan_contraction for multi-tensor einsum planning."""

    def setUp(self):
        self.config = CostModelConfig.from_device_type("cpu", 1)

    def test_two_tensor_matmul(self):
        """Two-tensor matmul should produce a single-step plan."""
        from einjax.optimizer.contraction_path import plan_contraction

        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))

        plan = plan_contraction("ij,jk->ik", [A, B], self.config)

        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.einsum_string, "ij,jk->ik")
        self.assertLess(plan.total_cost, float("inf"))
        self.assertGreater(plan.total_cost, 0.0)

    def test_three_tensor_chain(self):
        """Three-tensor chain ij,jk,kl->il should produce two steps."""
        from einjax.optimizer.contraction_path import plan_contraction

        A = DenseTensor("A", np.ones((3, 4)))
        B = DenseTensor("B", np.ones((4, 5)))
        C = DenseTensor("C", np.ones((5, 6)))

        plan = plan_contraction("ij,jk,kl->il", [A, B, C], self.config)

        self.assertEqual(len(plan.steps), 2)
        self.assertLess(plan.total_cost, float("inf"))

    def test_step_formulas_valid(self):
        """Each step should have a valid binary einsum formula."""
        from einjax.optimizer.contraction_path import plan_contraction

        A = DenseTensor("A", np.ones((3, 4)))
        B = DenseTensor("B", np.ones((4, 5)))
        C = DenseTensor("C", np.ones((5, 6)))

        plan = plan_contraction("ij,jk,kl->il", [A, B, C], self.config)

        for step in plan.steps:
            self.assertIn("->", step.formula)
            self.assertIsNotNone(step.output_tensor)
            self.assertIsNotNone(step.reduction_info)
            self.assertEqual(len(step.lhs_labels), len(step.formula.split("->")[0].split(",")[0]))
            self.assertEqual(len(step.rhs_labels), len(step.formula.split("->")[0].split(",")[1]))

    def test_best_scheme_per_step(self):
        """Each step should have a best_scheme with finite cost."""
        from einjax.optimizer.contraction_path import plan_contraction

        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))
        C = DenseTensor("C", np.ones((4, 4)))

        plan = plan_contraction("ij,jk,kl->il", [A, B, C], self.config)

        for step in plan.steps:
            self.assertIsNotNone(step.best_scheme)
            self.assertLess(step.best_scheme.cost, float("inf"))

    def test_tiling_schemes_nonempty(self):
        """Plan should have at least one tiling scheme."""
        from einjax.optimizer.contraction_path import plan_contraction

        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))

        plan = plan_contraction("ij,jk->ik", [A, B], self.config)

        self.assertGreater(len(plan.tiling_schemes), 0)

    def test_output_shape_correct(self):
        """Final step output shape should match expected einsum result."""
        from einjax.optimizer.contraction_path import plan_contraction

        A = DenseTensor("A", np.ones((3, 4)))
        B = DenseTensor("B", np.ones((4, 5)))
        C = DenseTensor("C", np.ones((5, 6)))

        plan = plan_contraction("ij,jk,kl->il", [A, B, C], self.config)

        final_step = plan.steps[-1]
        self.assertEqual(final_step.output_tensor.shape, (3, 6))

    def test_intermediate_naming(self):
        """Intermediate tensors should be named _t0, _t1, ...; final _output."""
        from einjax.optimizer.contraction_path import plan_contraction

        A = DenseTensor("A", np.ones((3, 4)))
        B = DenseTensor("B", np.ones((4, 5)))
        C = DenseTensor("C", np.ones((5, 6)))

        plan = plan_contraction("ij,jk,kl->il", [A, B, C], self.config)

        self.assertEqual(plan.steps[0].output_tensor.name, "_t0")
        self.assertEqual(plan.steps[-1].output_tensor.name, "_output")

    def test_fewer_than_two_tensors_raises(self):
        """plan_contraction with <2 tensors should raise ValueError."""
        from einjax.optimizer.contraction_path import plan_contraction

        A = DenseTensor("A", np.ones((4, 4)))
        with self.assertRaises(ValueError):
            plan_contraction("ij->ij", [A], self.config)

    def test_implicit_notation_normalized(self):
        """Implicit notation should be normalized in the plan."""
        from einjax.optimizer.contraction_path import plan_contraction

        A = DenseTensor("A", np.ones((3, 4)))
        B = DenseTensor("B", np.ones((4, 5)))

        plan = plan_contraction("ij,jk", [A, B], self.config)

        self.assertEqual(plan.einsum_string, "ij,jk->ik")


class TestContractionPlanCostConsistency(unittest.TestCase):
    """Test cost consistency across contraction plans."""

    def setUp(self):
        self.config = CostModelConfig.from_device_type("cpu", 1)

    def test_total_cost_equals_last_step_accumulated(self):
        """Plan total_cost should match the final step's accumulated cost."""
        from einjax.optimizer.contraction_path import plan_contraction

        A = DenseTensor("A", np.ones((4, 4)))
        B = DenseTensor("B", np.ones((4, 4)))
        C = DenseTensor("C", np.ones((4, 4)))

        plan = plan_contraction("ij,jk,kl->il", [A, B, C], self.config)

        self.assertEqual(plan.total_cost, plan.steps[-1].best_scheme.accumulated_cost)

    def test_faster_hardware_lower_total_cost(self):
        """Total cost should be lower on faster hardware."""
        from einjax.optimizer.contraction_path import plan_contraction

        A = DenseTensor("A", np.ones((6, 6)))
        B = DenseTensor("B", np.ones((6, 6)))
        C = DenseTensor("C", np.ones((6, 6)))

        cpu_config = CostModelConfig.from_device_type("cpu", 1)
        gpu_config = CostModelConfig.from_device_type("gpu:a100", 1)

        cpu_plan = plan_contraction("ij,jk,kl->il", [A, B, C], cpu_config)
        gpu_plan = plan_contraction("ij,jk,kl->il", [A, B, C], gpu_config)

        self.assertGreater(cpu_plan.total_cost, gpu_plan.total_cost)


class TestPackageContractionPathImports(unittest.TestCase):
    """Verify contraction path exports from the einjax package."""

    def test_import_from_optimizer(self):
        from einjax.optimizer import (
            ContractionStep,
            ContractionPlan,
            get_contraction_order,
            plan_contraction,
        )
        self.assertIsNotNone(ContractionStep)
        self.assertIsNotNone(ContractionPlan)
        self.assertIsNotNone(get_contraction_order)
        self.assertIsNotNone(plan_contraction)

    def test_import_from_top_level(self):
        import einjax
        self.assertTrue(hasattr(einjax, "ContractionStep"))
        self.assertTrue(hasattr(einjax, "ContractionPlan"))
        self.assertTrue(hasattr(einjax, "get_contraction_order"))
        self.assertTrue(hasattr(einjax, "plan_contraction"))


if __name__ == "__main__":
    unittest.main()
