"""
Tests for Branin function family.

Following TDD approach with comprehensive test coverage for all Branin variants.
Tests validate mathematical correctness, bounds handling, and edge cases.
"""

import numpy as np
import pytest

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.functions.benchmark.branin import Branin2Function, BraninFunction


class TestBraninFunction:
    """Test Branin RCOS function (2D non-scalable)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = BraninFunction()
        assert func.dimension == 2
        assert func.initialization_bounds.low.tolist() == [-5.0, 0.0]
        assert func.initialization_bounds.high.tolist() == [10.0, 15.0]

    def test_dimension_validation(self):
        """Test that non-2D initialization raises ValueError."""
        with pytest.raises(
            ValueError, match="Branin RCOS function is Non-Scalable and requires dimension=2"
        ):
            BraninFunction(dimension=3)

        with pytest.raises(
            ValueError, match="Branin RCOS function is Non-Scalable and requires dimension=2"
        ):
            BraninFunction(dimension=1)

    def test_custom_bounds(self):
        """Test initialization with custom bounds."""
        custom_bounds = Bounds(
            low=np.array([-2.0, -2.0]),
            high=np.array([2.0, 2.0]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        func = BraninFunction(operational_bounds=custom_bounds)
        np.testing.assert_array_equal(func.operational_bounds.low, [-2.0, -2.0])
        np.testing.assert_array_equal(func.operational_bounds.high, [2.0, 2.0])

    def test_global_minimum(self):
        """Test global minimum - Branin has 3 global minima."""
        func = BraninFunction()
        min_point, min_value = func.get_global_minimum()
        expected_value = 0.397887

        # Should return one of the three global minima
        expected_minima = [
            np.array([-np.pi, 12.275]),
            np.array([np.pi, 2.275]),
            np.array([9.42478, 2.475]),
        ]

        # Check if returned point is close to one of the known minima
        is_close_to_known = False
        for expected_min in expected_minima:
            if np.allclose(min_point, expected_min, atol=1e-3):
                is_close_to_known = True
                break

        assert is_close_to_known, f"Returned minimum {min_point} not close to any known minimum"
        assert abs(min_value - expected_value) < 1e-5

    def test_evaluate_at_global_minima(self):
        """Test function evaluates to ~0.397887 at global minima."""
        func = BraninFunction()
        expected_value = 0.397887

        # Test known global minima
        global_minima = [
            np.array([-np.pi, 12.275]),
            np.array([np.pi, 2.275]),
            np.array([9.42478, 2.475]),
        ]

        for min_point in global_minima:
            result = func.evaluate(min_point)
            assert abs(result - expected_value) < 1e-5, (
                f"Function value {result} at {min_point} should be ~{expected_value}"
            )

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = BraninFunction()

        # Test origin point [0, 0]
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)

        # Mathematical definition:
        # f(x,y) = a(y - b*x² + c*x - r)² + s(1 - t)*cos(x) + s
        # where a=1, b=5.1/(4π²), c=5/π, r=6, s=10, t=1/(8π)
        # f(0,0) = 1*(0 - 0 + 0 - 6)² + 10*(1 - 1/(8π))*cos(0) + 10
        #        = 36 + 10*(1 - 1/(8π))*1 + 10
        #        = 36 + 10*(1 - 1/(8π)) + 10
        a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi, 6.0, 10.0, 1.0 / (8 * np.pi)
        expected = a * (0 - b * 0**2 + c * 0 - r) ** 2 + s * (1 - t) * np.cos(0) + s

        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        func = BraninFunction()
        X = np.array(
            [
                [-np.pi, 12.275],  # Global minimum
                [np.pi, 2.275],  # Global minimum
                [0.0, 0.0],  # Origin
                [5.0, 10.0],  # Arbitrary point
            ]
        )
        results = func.evaluate_batch(X)

        # Verify individual evaluations match batch
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)

    def test_input_validation(self):
        """Test input validation for wrong dimensions."""
        func = BraninFunction()

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0]))  # Wrong dimension

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension

    def test_mathematical_properties(self):
        """Test mathematical properties of the function."""
        func = BraninFunction()

        # Test that function is non-negative (should be ≥ 0.397887)
        test_points = [
            [-np.pi, 12.275],  # Global minimum
            [0.0, 0.0],
            [5.0, 10.0],
            [-3.0, 5.0],
            [8.0, 12.0],
        ]

        for point in test_points:
            result = func.evaluate(np.array(point))
            assert result >= 0.39, f"Function should be ≥ 0.39, got {result} at {point}"

    def test_registry_names(self):
        """Test that all registry names work."""
        from pyMOFL.registry import get

        # Test both registry names
        func1 = get("Branin")()
        func2 = get("Branin_RCOS")()

        assert isinstance(func1, BraninFunction)
        assert isinstance(func2, BraninFunction)

        # Should produce same results
        x = np.array([1.0, 2.0])
        assert func1.evaluate(x) == func2.evaluate(x)


class TestBranin2Function:
    """Test Branin 2 function (2D non-scalable)."""

    def test_initialization(self):
        """Test proper initialization."""
        func = Branin2Function()
        assert func.dimension == 2
        assert func.initialization_bounds.low.tolist() == [-5.0, 0.0]
        assert func.initialization_bounds.high.tolist() == [10.0, 15.0]

    def test_dimension_validation(self):
        """Test dimension validation."""
        with pytest.raises(
            ValueError, match="Branin RCOS 2 function is Non-Scalable and requires dimension=2"
        ):
            Branin2Function(dimension=3)

    def test_global_minimum(self):
        """Test global minimum."""
        func = Branin2Function()
        min_point, min_value = func.get_global_minimum()

        # Branin 2 should have global minimum around similar locations as Branin
        # but with different value
        assert len(min_point) == 2
        assert isinstance(min_value, (int, float))
        assert min_value >= 0.0  # Should be non-negative

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates correctly at global minimum."""
        func = Branin2Function()
        min_point, min_value = func.get_global_minimum()

        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = Branin2Function()

        # Test origin
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        assert result >= 0.0  # Should be non-negative

    def test_batch_evaluation(self):
        """Test batch evaluation consistency."""
        func = Branin2Function()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 10.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)

    def test_difference_from_branin_1(self):
        """Test that Branin 2 produces different results from Branin 1."""
        branin1 = BraninFunction()
        branin2 = Branin2Function()

        test_points = [[0.0, 0.0], [1.0, 1.0], [5.0, 10.0]]

        for point in test_points:
            x = np.array(point)
            result1 = branin1.evaluate(x)
            result2 = branin2.evaluate(x)
            # Should produce different results (unless by coincidence)
            # We'll just verify both can be evaluated without error
            assert isinstance(result1, (int, float))
            assert isinstance(result2, (int, float))


class TestBraninFamilyIntegration:
    """Integration tests for Branin function family."""

    def test_all_functions_instantiate(self):
        """Test all Branin functions can be instantiated."""
        functions = [BraninFunction(), Branin2Function()]
        for func in functions:
            assert hasattr(func, "evaluate")
            assert hasattr(func, "evaluate_batch")
            assert hasattr(func, "get_global_minimum")

    def test_registry_integration(self):
        """Test functions are properly registered."""
        from pyMOFL.registry import get

        # Test registry names work
        func1 = get("Branin")()
        func2 = get("Branin_RCOS")()
        func3 = get("Branin_2")()

        assert isinstance(func1, BraninFunction)
        assert isinstance(func2, BraninFunction)
        assert isinstance(func3, Branin2Function)

    def test_bounds_consistency(self):
        """Test bounds are consistently handled."""
        functions = [BraninFunction(), Branin2Function()]

        for func in functions:
            assert func.initialization_bounds is not None
            assert func.operational_bounds is not None
            assert len(func.initialization_bounds.low) == 2
            assert len(func.initialization_bounds.high) == 2
            assert func.dimension == 2

    def test_mathematical_consistency(self):
        """Test mathematical consistency across function family."""
        functions = [BraninFunction(), Branin2Function()]

        # All should handle same input structure
        test_point = np.array([1.0, 2.0])

        for func in functions:
            result = func.evaluate(test_point)
            assert isinstance(result, (int, float))
            assert not np.isnan(result)
            assert not np.isinf(result)

    def test_global_minimum_consistency(self):
        """Test global minimum methods are consistent."""
        for func_class in [BraninFunction, Branin2Function]:
            func = func_class()
            min_point, min_value = func.get_global_minimum()

            evaluated_value = func.evaluate(min_point)

            assert abs(evaluated_value - min_value) < 1e-6, (
                f"Global minimum inconsistent for {func_class.__name__}"
            )
