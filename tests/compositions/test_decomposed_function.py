"""
Unit tests for DecomposedFunction composition.

Tests cooperative decomposition of optimization landscapes into separable,
m-dimensional non-separable, and overlapping sub-components.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.utils.benchmark_validation import BenchmarkValidator

from pyMOFL.compositions.decomposed_function import DecomposedFunction
from pyMOFL.functions.benchmark import RastriginFunction, RosenbrockFunction, SphereFunction
from pyMOFL.functions.transformations.decomposition import GroupingTransform
from pyMOFL.registry import get


class TestDecomposedFunction:
    """Tests for DecomposedFunction."""

    def test_registry_lookup(self):
        assert get("DecomposedFunction") is DecomposedFunction
        assert get("decomposed_function") is DecomposedFunction

    def test_decomposed_function_evaluation(self):
        """Test evaluation with 2 rotated non-separable blocks and 1 separable block."""
        R1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        R2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)

        gt = GroupingTransform.from_sizes(
            dimension=8,
            block_sizes=[3, 3],
            block_rotations=[R1, R2],
            separable_size=2,
        )

        f_non_sep1 = SphereFunction(dimension=3)
        f_non_sep2 = SphereFunction(dimension=3)
        f_sep = SphereFunction(dimension=2)

        func = DecomposedFunction(
            grouping_transform=gt,
            non_separable_function=[f_non_sep1, f_non_sep2],
            separable_function=f_sep,
            weights=[1.0, 2.0, 0.5],
            bias=100.0,
        )

        x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        # rotated non-sep 1: R1 @ [1, 2, 3] = [2, 3, 1] -> Sphere = 4 + 9 + 1 = 14
        # rotated non-sep 2: R2 @ [4, 5, 6] = [-4, 5, -6] -> Sphere = 16 + 25 + 36 = 77
        # separable: [7, 8] -> Sphere = 49 + 64 = 113
        # total = 100 + 1.0 * 14 + 2.0 * 77 + 0.5 * 113 = 100 + 14 + 154 + 56.5 = 324.5
        val = func.evaluate(x)
        assert np.isclose(val, 324.5)

    def test_decomposed_function_evaluate_batch(self):
        """Test vectorized batch evaluation matches individual evaluations."""
        R = np.array([[0, 1], [-1, 0]], dtype=float)
        gt = GroupingTransform.from_sizes(
            dimension=4,
            block_sizes=[2],
            block_rotations=[R],
            separable_size=2,
        )

        f_non_sep = RosenbrockFunction(dimension=2)
        f_sep = SphereFunction(dimension=2)

        func = DecomposedFunction(
            grouping_transform=gt,
            non_separable_function=f_non_sep,
            separable_function=f_sep,
            bias=25.0,
        )

        rng = np.random.default_rng(42)
        X = rng.uniform(-5.0, 5.0, size=(10, 4))

        batch_vals = func.evaluate_batch(X)
        single_vals = np.array([func.evaluate(row) for row in X])

        np.testing.assert_allclose(batch_vals, single_vals, atol=1e-12)

    def test_decomposed_function_overlapping(self):
        """Test evaluation with overlapping sub-components."""
        gt = GroupingTransform.from_overlapping_window(
            dimension=6,
            group_size=3,
            overlap_size=1,
            num_groups=2,
        )
        # Groups: [0, 1, 2], [2, 3, 4] -> variable 2 is shared
        f1 = SphereFunction(dimension=3)
        f2 = SphereFunction(dimension=3)

        func = DecomposedFunction(
            grouping_transform=gt,
            overlapping_function=[f1, f2],
            bias=-50.0,
        )

        x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        # group 1: [1, 2, 3] -> Sphere = 1 + 4 + 9 = 14
        # group 2: [3, 4, 5] -> Sphere = 9 + 16 + 25 = 50
        # total = -50 + 14 + 50 = 14
        assert np.isclose(func.evaluate(x), 14.0)

        # Overlapping global minimum should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="overlapping"):
            func.get_global_minimum()

    def test_decomposed_function_global_minimum_recovery(self):
        """Test analytical recovery of global optimum in non-overlapping partition."""
        # Non-trivial rotation matrix: 90 degree 2D rotation
        R = np.array([[0.0, -1.0], [1.0, 0.0]])
        P = np.array([3, 0, 2, 1])  # Permutation

        gt = GroupingTransform(
            dimension=4,
            permutation=P,
            non_separable_groups=[[0, 1]],
            block_rotations=[R],
            separable_indices=[2, 3],
        )

        # Sphere minimum is at origin (0, 0) with value 0.0
        # Rastrigin minimum is at origin (0, 0) with value 0.0
        f_non_sep = SphereFunction(dimension=2)
        f_sep = RastriginFunction(dimension=2)

        func = DecomposedFunction(
            grouping_transform=gt,
            non_separable_function=f_non_sep,
            separable_function=f_sep,
            bias=-123.45,
        )

        opt_x, opt_f = func.get_global_minimum()
        assert np.isclose(opt_f, -123.45)
        # Value at opt_x must evaluate to opt_f
        assert np.isclose(func.evaluate(opt_x), opt_f, atol=1e-10)

    def test_benchmark_contract(self):
        """Test full BenchmarkValidator contract on a partitioned DecomposedFunction."""
        gt = GroupingTransform.from_sizes(
            dimension=4,
            block_sizes=[2],
            block_rotations=[np.eye(2)],
            separable_size=2,
        )
        func = DecomposedFunction(
            grouping_transform=gt,
            non_separable_function=SphereFunction(dimension=2),
            separable_function=SphereFunction(dimension=2),
        )
        BenchmarkValidator.assert_contract(func, check_global_minimum=True)

    def test_factory_construction(self):
        """Test constructing DecomposedFunction via FunctionFactory from nested dictionary config."""
        from pyMOFL.factories.function_factory import FunctionFactory

        factory = FunctionFactory()
        cfg = {
            "type": "bias",
            "parameters": {"value": 50.0},
            "function": {
                "type": "decomposed",
                "parameters": {
                    "dimension": 4,
                    "grouping": {
                        "block_sizes": [2],
                        "separable_size": 2,
                    },
                    "weights": [2.0, 1.0],
                },
                "function": {
                    "type": "shift",
                    "parameters": {"vector": [1.0, 1.0, 2.0, 2.0]},
                },
                "functions": [
                    {"type": "sphere", "parameters": {"dimension": 2}},
                    {"type": "sphere", "parameters": {"dimension": 2}},
                ],
            },
        }
        func = factory.create_function(cfg)
        assert func.dimension == 4
        # At x = [1, 1, 2, 2], shift -> 0 -> spheres are 0 -> total = 50.0
        val = func.evaluate(np.array([1.0, 1.0, 2.0, 2.0]))
        assert np.isclose(val, 50.0)
