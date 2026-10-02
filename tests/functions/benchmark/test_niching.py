"""
Tests for multimodal niching benchmark functions (CEC 2013 / CEC 2015 Niching).

Validates mathematical correctness, boundary handling, batch evaluation,
and registry integration for all 8 niching base functions.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.niching import (
    ExpandedDecreasingMinimaFunction,
    ExpandedEqualMinimaFunction,
    ExpandedFiveUnevenPeakTrapFunction,
    ExpandedHimmelblauFunction,
    ExpandedSixHumpCamelFunction,
    ExpandedTwoPeakTrapFunction,
    ExpandedUnevenMinimaFunction,
    ModifiedVincentFunction,
)
from tests.utils.benchmark_validation import BenchmarkValidator


class TestNichingFunctions:
    """Contract and evaluation tests for niching functions."""

    def test_two_peak_trap(self):
        func = ExpandedTwoPeakTrapFunction(dimension=2)
        BenchmarkValidator.assert_contract(func, check_global_minimum=True, global_min_atol=1e-8)
        # Global optimum at x = 20 gives trap(20) = 200, so f(x) = 0
        assert abs(func.evaluate(np.array([20.0, 20.0]))) < 1e-10

    def test_five_uneven_peak_trap(self):
        func = ExpandedFiveUnevenPeakTrapFunction(dimension=2)
        BenchmarkValidator.assert_contract(func, check_global_minimum=True, global_min_atol=1e-8)
        # Global peaks (minima in minimization f(x) = 0) at x = 0 and x = 30
        assert abs(func.evaluate(np.zeros(2))) < 1e-10
        assert abs(func.evaluate(np.full(2, 30.0))) < 1e-10

        # Local peaks (minima in minimization):
        # x = 5.0 -> trap = 160 -> f(x) = 200 - 160 = 40.0 per dimension -> 80.0 for 2D
        np.testing.assert_allclose(func.evaluate(np.full(2, 5.0)), 80.0, rtol=1e-10)
        # x = 12.5 -> trap = 140 -> f(x) = 200 - 140 = 60.0 per dimension -> 120.0 for 2D
        np.testing.assert_allclose(func.evaluate(np.full(2, 12.5)), 120.0, rtol=1e-10)
        # x = 22.5 -> trap = 160 -> f(x) = 200 - 160 = 40.0 per dimension -> 80.0 for 2D
        np.testing.assert_allclose(func.evaluate(np.full(2, 22.5)), 80.0, rtol=1e-10)

        # Point inside (5, 7.5):
        # x = 6.0: linear piece is 200 - 64 * (7.5 - 6.0) = 200 - 96 = 104.0 per dimension -> 208.0 for 2D
        np.testing.assert_allclose(func.evaluate(np.full(2, 6.0)), 208.0, rtol=1e-10)
        # 1D check as well
        func1 = ExpandedFiveUnevenPeakTrapFunction(dimension=1)
        np.testing.assert_allclose(func1.evaluate(np.array([5.0])), 40.0, rtol=1e-10)
        np.testing.assert_allclose(func1.evaluate(np.array([6.0])), 104.0, rtol=1e-10)
        np.testing.assert_allclose(func1.evaluate(np.array([12.5])), 60.0, rtol=1e-10)
        np.testing.assert_allclose(func1.evaluate(np.array([22.5])), 40.0, rtol=1e-10)

    def test_equal_minima(self):
        func = ExpandedEqualMinimaFunction(dimension=3)
        BenchmarkValidator.assert_contract(func, check_global_minimum=True, global_min_atol=1e-8)
        # Optima at x_i in {0.1, 0.3, 0.5, 0.7, 0.9}
        for opt_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(func.evaluate(np.full(3, opt_val))) < 1e-10

    def test_decreasing_minima(self):
        func = ExpandedDecreasingMinimaFunction(dimension=2)
        BenchmarkValidator.assert_contract(func, check_global_minimum=False)
        x_sample = np.array([0.5, 0.5])
        assert func.evaluate(x_sample) >= 0.0

    def test_uneven_minima(self):
        func = ExpandedUnevenMinimaFunction(dimension=2)
        BenchmarkValidator.assert_contract(func, check_global_minimum=True, global_min_atol=1e-6)
        x_opt, _ = func.get_global_minimum()
        assert abs(func.evaluate(x_opt)) < 1e-6

    def test_expanded_himmelblau(self):
        func = ExpandedHimmelblauFunction(dimension=2)
        BenchmarkValidator.assert_contract(func, check_global_minimum=True, global_min_atol=1e-8)
        # Himmelblau optimum at (3, 2)
        assert abs(func.evaluate(np.array([3.0, 2.0]))) < 1e-10

        # Scalable even dimension >= 2
        func4 = ExpandedHimmelblauFunction(dimension=4)
        BenchmarkValidator.assert_contract(func4, check_global_minimum=True, global_min_atol=1e-8)
        with pytest.raises(ValueError, match="requires even dimension >= 2"):
            ExpandedHimmelblauFunction(dimension=1)
        with pytest.raises(ValueError, match="requires even dimension >= 2"):
            ExpandedHimmelblauFunction(dimension=3)
        with pytest.raises(ValueError, match="requires even dimension >= 2"):
            ExpandedHimmelblauFunction(dimension=5)

    def test_expanded_six_hump_camel(self):
        func = ExpandedSixHumpCamelFunction(dimension=2)
        BenchmarkValidator.assert_contract(func, check_global_minimum=True, global_min_atol=1e-3)
        with pytest.raises(ValueError):
            ExpandedSixHumpCamelFunction(dimension=1)

    def test_modified_vincent(self):
        func = ModifiedVincentFunction(dimension=3)
        BenchmarkValidator.assert_contract(func, check_global_minimum=True, global_min_atol=1e-8)
        x_opt, _ = func.get_global_minimum()
        assert abs(func.evaluate(x_opt)) < 1e-10

        # Non-optimum point evaluated in full float64 precision
        func1 = ModifiedVincentFunction(dimension=1)
        x_test = np.array([3.0])
        expected_val = float(-np.sin(10.0 * np.log(3.0)) + 1.0)
        np.testing.assert_allclose(func1.evaluate(x_test), expected_val, rtol=1e-12, atol=1e-12)


class TestNichingBatchEvaluation:
    """Verify batch evaluation across all niching functions."""

    @pytest.mark.parametrize(
        "func_cls,dim",
        [
            (ExpandedTwoPeakTrapFunction, 3),
            (ExpandedFiveUnevenPeakTrapFunction, 3),
            (ExpandedEqualMinimaFunction, 3),
            (ExpandedDecreasingMinimaFunction, 3),
            (ExpandedUnevenMinimaFunction, 3),
            (ExpandedHimmelblauFunction, 4),
            (ExpandedSixHumpCamelFunction, 4),
            (ModifiedVincentFunction, 3),
        ],
    )
    def test_batch_matches_single(self, func_cls, dim):
        func = func_cls(dimension=dim)
        low = func.operational_bounds.low
        high = func.operational_bounds.high
        rng = np.random.default_rng(42)
        X = rng.uniform(low, high, size=(8, dim))
        batch_res = func.evaluate_batch(X)
        single_res = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch_res, single_res, rtol=1e-12, atol=1e-12)


class TestNichingRegistry:
    """Verify all niching functions are discoverable via registry."""

    @pytest.mark.parametrize(
        "alias,expected_cls",
        [
            ("ExpandedTwoPeakTrap", ExpandedTwoPeakTrapFunction),
            ("expanded_two_peak_trap", ExpandedTwoPeakTrapFunction),
            ("TwoPeakTrap", ExpandedTwoPeakTrapFunction),
            ("ExpandedFiveUnevenPeakTrap", ExpandedFiveUnevenPeakTrapFunction),
            ("ExpandedEqualMinima", ExpandedEqualMinimaFunction),
            ("ExpandedDecreasingMinima", ExpandedDecreasingMinimaFunction),
            ("ExpandedUnevenMinima", ExpandedUnevenMinimaFunction),
            ("ExpandedHimmelblau", ExpandedHimmelblauFunction),
            ("ExpandedSixHumpCamel", ExpandedSixHumpCamelFunction),
            ("ModifiedVincent", ModifiedVincentFunction),
        ],
    )
    def test_registry_resolution(self, alias, expected_cls):
        from pyMOFL.registry import get

        cls_found = get(alias)
        assert cls_found is expected_cls
