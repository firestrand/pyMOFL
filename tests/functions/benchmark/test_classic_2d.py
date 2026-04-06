"""
Tests for classic 2D optimization functions.

Following TDD approach with comprehensive test coverage for all classic 2D variants:
- Himmelblau function
- Matyas function
- Easom function
- McCormick function
- Goldstein-Price function

Tests validate mathematical correctness, bounds handling, and edge cases.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.easom import EasomFunction
from pyMOFL.functions.benchmark.goldstein_price import GoldsteinPriceFunction
from pyMOFL.functions.benchmark.himmelblau import HimmelblauFunction
from pyMOFL.functions.benchmark.matyas import MatyasFunction
from pyMOFL.functions.benchmark.mccormick import McCormickFunction


class TestHimmelblauFunction:
    """Test Himmelblau function (2D non-scalable with 4 global minima)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = HimmelblauFunction()
        assert func.dimension == 2
        assert func.initialization_bounds.low.tolist() == [-5.0, -5.0]
        assert func.initialization_bounds.high.tolist() == [5.0, 5.0]

    def test_dimension_validation(self):
        """Test that non-2D initialization raises ValueError."""
        with pytest.raises(
            ValueError, match="Himmelblau function is Non-Scalable and requires dimension=2"
        ):
            HimmelblauFunction(dimension=3)

    def test_global_minimum(self):
        """Test global minimum - Himmelblau has 4 global minima all with value 0."""
        func = HimmelblauFunction()
        min_point, min_value = func.get_global_minimum()
        assert min_value == 0.0

        # Should return one of the four known global minima
        known_minima = [
            np.array([3.0, 2.0]),
            np.array([-2.805118, 3.131312]),
            np.array([-3.779310, -3.283186]),
            np.array([3.584428, -1.848126]),
        ]

        # Check if returned point is close to one of the known minima
        is_close_to_known = False
        for expected_min in known_minima:
            if np.allclose(min_point, expected_min, atol=1e-3):
                is_close_to_known = True
                break

        assert is_close_to_known, f"Returned minimum {min_point} not close to any known minimum"

    def test_evaluate_at_global_minima(self):
        """Test function evaluates to 0 at all global minima."""
        func = HimmelblauFunction()

        global_minima = [
            np.array([3.0, 2.0]),
            np.array([-2.805118, 3.131312]),
            np.array([-3.779310, -3.283186]),
            np.array([3.584428, -1.848126]),
        ]

        for min_point in global_minima:
            result = func.evaluate(min_point)
            assert abs(result) < 1e-5, f"Function value {result} at {min_point} should be ~0"

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = HimmelblauFunction()

        # Test origin [0, 0]
        # f(x,y) = (x² + y - 11)² + (x + y² - 7)²
        # f(0,0) = (0 + 0 - 11)² + (0 + 0 - 7)² = 121 + 49 = 170
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        expected = (0**2 + 0 - 11) ** 2 + (0 + 0**2 - 7) ** 2
        assert abs(result - expected) < 1e-10

        # Test [1, 1]
        # f(1,1) = (1 + 1 - 11)² + (1 + 1 - 7)² = 81 + 25 = 106
        x = np.array([1.0, 1.0])
        result = func.evaluate(x)
        expected = (1**2 + 1 - 11) ** 2 + (1 + 1**2 - 7) ** 2
        assert abs(result - expected) < 1e-10


class TestMatyasFunction:
    """Test Matyas function (2D non-scalable unimodal)."""

    def test_initialization(self):
        """Test proper initialization."""
        func = MatyasFunction()
        assert func.dimension == 2
        assert func.initialization_bounds.low.tolist() == [-10.0, -10.0]
        assert func.initialization_bounds.high.tolist() == [10.0, 10.0]

    def test_dimension_validation(self):
        """Test dimension validation."""
        with pytest.raises(
            ValueError, match="Matyas function is Non-Scalable and requires dimension=2"
        ):
            MatyasFunction(dimension=3)

    def test_global_minimum(self):
        """Test global minimum at origin."""
        func = MatyasFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [0.0, 0.0])
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at origin."""
        func = MatyasFunction()
        result = func.evaluate(np.array([0.0, 0.0]))
        assert abs(result) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = MatyasFunction()

        # Test [1, 1]
        # f(x,y) = 0.26(x² + y²) - 0.48xy
        # f(1,1) = 0.26(1 + 1) - 0.48(1)(1) = 0.52 - 0.48 = 0.04
        x = np.array([1.0, 1.0])
        result = func.evaluate(x)
        expected = 0.26 * (1**2 + 1**2) - 0.48 * 1 * 1
        assert abs(result - expected) < 1e-10

        # Test [2, -1]
        # f(2,-1) = 0.26(4 + 1) - 0.48(2)(-1) = 1.3 + 0.96 = 2.26
        x = np.array([2.0, -1.0])
        result = func.evaluate(x)
        expected = 0.26 * (4 + 1) - 0.48 * 2 * (-1)
        assert abs(result - expected) < 1e-10


class TestEasomFunction:
    """Test Easom function (2D non-scalable with very small global minimum area)."""

    def test_initialization(self):
        """Test proper initialization."""
        func = EasomFunction()
        assert func.dimension == 2
        assert func.initialization_bounds.low.tolist() == [-100.0, -100.0]
        assert func.initialization_bounds.high.tolist() == [100.0, 100.0]

    def test_dimension_validation(self):
        """Test dimension validation."""
        with pytest.raises(
            ValueError, match="Easom function is Non-Scalable and requires dimension=2"
        ):
            EasomFunction(dimension=3)

    def test_global_minimum(self):
        """Test global minimum at (π, π)."""
        func = EasomFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [np.pi, np.pi])
        assert min_value == -1.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to -1 at (π, π)."""
        func = EasomFunction()
        result = func.evaluate(np.array([np.pi, np.pi]))
        assert abs(result - (-1.0)) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = EasomFunction()

        # Test origin [0, 0]
        # f(x,y) = -cos(x)cos(y)exp(-(x-π)² - (y-π)²)
        # f(0,0) = -cos(0)cos(0)exp(-(0-π)² - (0-π)²) = -1 * exp(-π² - π²) = -exp(-2π²)
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        expected = -np.cos(0) * np.cos(0) * np.exp(-((0 - np.pi) ** 2) - (0 - np.pi) ** 2)
        assert abs(result - expected) < 1e-10

        # Test [1, 1]
        x = np.array([1.0, 1.0])
        result = func.evaluate(x)
        expected = -np.cos(1) * np.cos(1) * np.exp(-((1 - np.pi) ** 2) - (1 - np.pi) ** 2)
        assert abs(result - expected) < 1e-10


class TestMcCormickFunction:
    """Test McCormick function (2D non-scalable multimodal)."""

    def test_initialization(self):
        """Test proper initialization."""
        func = McCormickFunction()
        assert func.dimension == 2
        assert func.initialization_bounds.low.tolist() == [-1.5, -3.0]
        assert func.initialization_bounds.high.tolist() == [4.0, 4.0]

    def test_dimension_validation(self):
        """Test dimension validation."""
        with pytest.raises(
            ValueError, match="McCormick function is Non-Scalable and requires dimension=2"
        ):
            McCormickFunction(dimension=3)

    def test_global_minimum(self):
        """Test global minimum."""
        func = McCormickFunction()
        min_point, min_value = func.get_global_minimum()
        expected_point = np.array([-0.54719, -1.54719])
        expected_value = -1.9133

        np.testing.assert_array_almost_equal(min_point, expected_point, decimal=4)
        assert abs(min_value - expected_value) < 1e-3

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates correctly at global minimum."""
        func = McCormickFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-3

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = McCormickFunction()

        # Test origin [0, 0]
        # f(x,y) = sin(x + y) + (x - y)² - 1.5x + 2.5y + 1
        # f(0,0) = sin(0) + 0 - 0 + 0 + 1 = 1
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        expected = np.sin(0 + 0) + (0 - 0) ** 2 - 1.5 * 0 + 2.5 * 0 + 1
        assert abs(result - expected) < 1e-10


class TestGoldsteinPriceFunction:
    """Test Goldstein-Price function (2D non-scalable with scaling challenges)."""

    def test_initialization(self):
        """Test proper initialization."""
        func = GoldsteinPriceFunction()
        assert func.dimension == 2
        assert func.initialization_bounds.low.tolist() == [-2.0, -2.0]
        assert func.initialization_bounds.high.tolist() == [2.0, 2.0]

    def test_dimension_validation(self):
        """Test dimension validation."""
        with pytest.raises(
            ValueError, match="Goldstein-Price function is Non-Scalable and requires dimension=2"
        ):
            GoldsteinPriceFunction(dimension=3)

    def test_global_minimum(self):
        """Test global minimum."""
        func = GoldsteinPriceFunction()
        min_point, min_value = func.get_global_minimum()
        expected_point = np.array([0.0, -1.0])
        expected_value = 3.0

        np.testing.assert_array_almost_equal(min_point, expected_point)
        assert min_value == expected_value

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 3 at (0, -1)."""
        func = GoldsteinPriceFunction()
        result = func.evaluate(np.array([0.0, -1.0]))
        assert abs(result - 3.0) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = GoldsteinPriceFunction()

        # Test origin [0, 0]
        # f(x,y) = [1 + (x + y + 1)²(19 - 14x + 3x² - 14y + 6xy + 3y²)] ×
        #          [30 + (2x - 3y)²(18 - 32x + 12x² + 48y - 36xy + 27y²)]
        # At [0,0]:
        # First bracket: [1 + (0 + 0 + 1)²(19 - 0 + 0 - 0 + 0 + 0)] = [1 + 1×19] = 20
        # Second bracket: [30 + (0 - 0)²(18 - 0 + 0 + 0 - 0 + 0)] = [30 + 0×18] = 30
        # f(0,0) = 20 × 30 = 600
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)

        # Calculate expected value step by step
        x1, x2 = 0.0, 0.0
        term1 = 1 + (x1 + x2 + 1) ** 2 * (
            19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        )
        term2 = 30 + (2 * x1 - 3 * x2) ** 2 * (
            18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        )
        expected = term1 * term2

        assert abs(result - expected) < 1e-10


class TestClassic2DFunctionIntegration:
    """Integration tests for all classic 2D functions."""

    @pytest.mark.parametrize(
        "func_class",
        [
            HimmelblauFunction,
            MatyasFunction,
            EasomFunction,
            McCormickFunction,
            GoldsteinPriceFunction,
        ],
    )
    def test_all_functions_instantiate(self, func_class):
        """Test all functions can be instantiated."""
        func = func_class()
        assert hasattr(func, "evaluate")
        assert hasattr(func, "evaluate_batch")
        assert hasattr(func, "get_global_minimum")
        assert func.dimension == 2

    @pytest.mark.parametrize(
        "func_class",
        [
            HimmelblauFunction,
            MatyasFunction,
            EasomFunction,
            McCormickFunction,
            GoldsteinPriceFunction,
        ],
    )
    def test_batch_evaluation_consistency(self, func_class):
        """Test batch evaluation matches individual evaluation."""
        func = func_class()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)

    def test_registry_integration(self):
        """Test functions are properly registered."""
        from pyMOFL.registry import get

        # Test registry names work
        himmelblau = get("Himmelblau")()
        matyas = get("Matyas")()
        easom = get("Easom")()
        mccormick = get("McCormick")()
        goldstein = get("Goldstein_Price")()

        assert isinstance(himmelblau, HimmelblauFunction)
        assert isinstance(matyas, MatyasFunction)
        assert isinstance(easom, EasomFunction)
        assert isinstance(mccormick, McCormickFunction)
        assert isinstance(goldstein, GoldsteinPriceFunction)

    @pytest.mark.parametrize(
        "func_class",
        [
            HimmelblauFunction,
            MatyasFunction,
            EasomFunction,
            McCormickFunction,
            GoldsteinPriceFunction,
        ],
    )
    def test_bounds_consistency(self, func_class):
        """Test bounds are consistently handled."""
        func = func_class()
        assert func.initialization_bounds is not None
        assert func.operational_bounds is not None
        assert len(func.initialization_bounds.low) == 2
        assert len(func.initialization_bounds.high) == 2
        assert func.dimension == 2

    @pytest.mark.parametrize(
        "func_class",
        [
            HimmelblauFunction,
            MatyasFunction,
            EasomFunction,
            McCormickFunction,
            GoldsteinPriceFunction,
        ],
    )
    def test_global_minimum_consistency(self, func_class):
        """Test global minimum methods are consistent."""
        func = func_class()
        min_point, min_value = func.get_global_minimum()
        evaluated_value = func.evaluate(min_point)

        # Allow some tolerance for numerical precision
        tolerance = 1e-3 if func_class == McCormickFunction else 1e-10
        assert abs(evaluated_value - min_value) < tolerance, (
            f"Global minimum inconsistent for {func_class.__name__}"
        )

    @pytest.mark.parametrize(
        "func_class",
        [
            HimmelblauFunction,
            MatyasFunction,
            EasomFunction,
            McCormickFunction,
            GoldsteinPriceFunction,
        ],
    )
    def test_input_validation(self, func_class):
        """Test input validation for all functions."""
        func = func_class()

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0]))  # Wrong dimension

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension

    @pytest.mark.parametrize(
        "func_class",
        [
            HimmelblauFunction,
            MatyasFunction,
            EasomFunction,
            McCormickFunction,
            GoldsteinPriceFunction,
        ],
    )
    def test_mathematical_consistency(self, func_class):
        """Test mathematical consistency."""
        func = func_class()
        test_point = np.array([0.5, -0.5])

        result = func.evaluate(test_point)
        assert isinstance(result, (int, float))
        assert not np.isnan(result)
        assert not np.isinf(result)
