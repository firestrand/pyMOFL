"""
Tests for Powell function family.

Following TDD approach with comprehensive test coverage for all Powell variants.
Tests validate mathematical correctness, bounds handling, and edge cases.
"""

import numpy as np
import pytest

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.functions.benchmark.powell import (
    PowellSingular2Function,
    PowellSingularFunction,
    PowellSumFunction,
)


class TestPowellSingularFunction:
    """Test Powell Singular function (4D non-scalable)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = PowellSingularFunction()
        assert func.dimension == 4
        assert func.initialization_bounds.low.tolist() == [-4.0, -4.0, -4.0, -4.0]
        assert func.initialization_bounds.high.tolist() == [5.0, 5.0, 5.0, 5.0]

    def test_dimension_validation(self):
        """Test that non-4D initialization raises ValueError."""
        with pytest.raises(
            ValueError, match="Powell Singular function is Non-Scalable and requires dimension=4"
        ):
            PowellSingularFunction(dimension=3)

        with pytest.raises(
            ValueError, match="Powell Singular function is Non-Scalable and requires dimension=4"
        ):
            PowellSingularFunction(dimension=5)

    def test_custom_bounds(self):
        """Test initialization with custom bounds."""
        custom_bounds = Bounds(
            low=np.array([-2.0, -2.0, -2.0, -2.0]),
            high=np.array([2.0, 2.0, 2.0, 2.0]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        func = PowellSingularFunction(operational_bounds=custom_bounds)
        np.testing.assert_array_equal(func.operational_bounds.low, [-2.0, -2.0, -2.0, -2.0])
        np.testing.assert_array_equal(func.operational_bounds.high, [2.0, 2.0, 2.0, 2.0])

    def test_global_minimum(self):
        """Test global minimum is at origin with value 0."""
        func = PowellSingularFunction()
        min_point, min_value = func.get_global_minimum()
        expected_point = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(min_point, expected_point)
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at global minimum."""
        func = PowellSingularFunction()
        x = np.array([0.0, 0.0, 0.0, 0.0])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = PowellSingularFunction()

        # Test point: [1, 1, 1, 1]
        # f(x) = (x1 + 10*x2)² + 5*(x3 - x4)² + (x2 - 2*x3)^4 + 10*(x1 - x4)^4
        # f([1,1,1,1]) = (1 + 10)² + 5*(1-1)² + (1-2)^4 + 10*(1-1)^4 = 121 + 0 + 1 + 0 = 122
        x = np.array([1.0, 1.0, 1.0, 1.0])
        result = func.evaluate(x)
        expected = 122.0
        assert abs(result - expected) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        func = PowellSingularFunction()
        X = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.5, -0.5, 0.2, -0.2]])
        results = func.evaluate_batch(X)

        # Verify individual evaluations match batch
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)

    def test_input_validation(self):
        """Test input validation for wrong dimensions."""
        func = PowellSingularFunction()

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))  # Wrong dimension

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension

    def test_mathematical_properties(self):
        """Test mathematical properties of the function."""
        func = PowellSingularFunction()

        # Test symmetry properties - Powell Singular is not symmetric
        x1 = np.array([1.0, 2.0, 3.0, 4.0])
        x2 = np.array([2.0, 1.0, 4.0, 3.0])
        result1 = func.evaluate(x1)
        result2 = func.evaluate(x2)
        assert result1 != result2  # Not symmetric

        # Test non-negativity
        test_points = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [0.5, -0.5, 0.2, -0.2],
        ]
        for point in test_points:
            result = func.evaluate(np.array(point))
            assert result >= 0.0, f"Function should be non-negative, got {result} at {point}"


class TestPowellSingular2Function:
    """Test Powell Singular 2 function (4D non-scalable)."""

    def test_initialization(self):
        """Test proper initialization."""
        func = PowellSingular2Function()
        assert func.dimension == 4

    def test_dimension_validation(self):
        """Test dimension validation."""
        with pytest.raises(
            ValueError, match="Powell Singular 2 function is Non-Scalable and requires dimension=4"
        ):
            PowellSingular2Function(dimension=3)

    def test_global_minimum(self):
        """Test global minimum."""
        func = PowellSingular2Function()
        min_point, min_value = func.get_global_minimum()
        expected_point = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(min_point, expected_point)
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at global minimum."""
        func = PowellSingular2Function()
        x = np.array([0.0, 0.0, 0.0, 0.0])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = PowellSingular2Function()

        # Test point: [1, 1, 1, 1]
        # f(x) = (x1 + 10*x2)² + 5*(x3 - x4)² + (x2 - 2*x3)^4 + 10*(x1 - x4)^4
        # Different formulation than Powell Singular - implementation will define exact formula
        x = np.array([1.0, 1.0, 1.0, 1.0])
        result = func.evaluate(x)
        assert result >= 0.0  # Should be non-negative

    def test_batch_evaluation(self):
        """Test batch evaluation consistency."""
        func = PowellSingular2Function()
        X = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)


class TestPowellSumFunction:
    """Test Powell Sum function (scalable)."""

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [2, 5, 10, 30]:
            func = PowellSumFunction(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim
            assert len(func.initialization_bounds.high) == dim

    def test_default_bounds(self):
        """Test default bounds are correct."""
        func = PowellSumFunction(dimension=5)
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(5, -1.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(5, 1.0))

    def test_global_minimum_various_dimensions(self):
        """Test global minimum for various dimensions."""
        for dim in [2, 5, 10]:
            func = PowellSumFunction(dimension=dim)
            min_point, min_value = func.get_global_minimum()
            expected_point = np.zeros(dim)
            np.testing.assert_array_almost_equal(min_point, expected_point)
            assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at global minimum."""
        for dim in [2, 5, 10]:
            func = PowellSumFunction(dimension=dim)
            x = np.zeros(dim)
            result = func.evaluate(x)
            assert abs(result) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = PowellSumFunction(dimension=3)

        # Test point: [1, 1, 1]
        # f(x) = sum(|x_i|^(i+2)) for i=1 to n
        # f([1,1,1]) = |1|^2 + |1|^3 + |1|^4 = 1 + 1 + 1 = 3
        x = np.array([1.0, 1.0, 1.0])
        result = func.evaluate(x)
        expected = 3.0
        assert abs(result - expected) < 1e-10

        # Test point: [0.5, -0.5, 0.5]
        # f([0.5,-0.5,0.5]) = |0.5|^2 + |-0.5|^3 + |0.5|^4 = 0.25 + 0.125 + 0.0625 = 0.4375
        x = np.array([0.5, -0.5, 0.5])
        result = func.evaluate(x)
        expected = 0.4375
        assert abs(result - expected) < 1e-10

    def test_batch_evaluation(self):
        """Test batch evaluation consistency."""
        func = PowellSumFunction(dimension=4)
        X = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.5, -0.5, 0.5, -0.5]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)

    def test_mathematical_properties(self):
        """Test mathematical properties."""
        func = PowellSumFunction(dimension=5)

        # Test non-negativity
        test_points = [np.zeros(5), np.ones(5), -np.ones(5), np.random.randn(5) * 0.5]
        for point in test_points:
            result = func.evaluate(point)
            assert result >= 0.0, f"Function should be non-negative, got {result} at {point}"

    def test_input_validation(self):
        """Test input validation."""
        func = PowellSumFunction(dimension=4)

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))  # Wrong dimension

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension


class TestPowellFamilyIntegration:
    """Integration tests for Powell function family."""

    def test_all_functions_instantiate(self):
        """Test all Powell functions can be instantiated."""
        functions = [
            PowellSingularFunction(),
            PowellSingular2Function(),
            PowellSumFunction(dimension=5),
        ]
        for func in functions:
            assert hasattr(func, "evaluate")
            assert hasattr(func, "evaluate_batch")
            assert hasattr(func, "get_global_minimum")

    def test_registry_integration(self):
        """Test functions are properly registered."""
        from pyMOFL.registry import get

        # Test registry names work
        func1 = get("Powell_Singular")(dimension=4)
        func2 = get("PowellSingular")(dimension=4)
        func3 = get("Powell_Sum")(dimension=5)

        assert isinstance(func1, PowellSingularFunction)
        assert isinstance(func2, PowellSingularFunction)
        assert isinstance(func3, PowellSumFunction)

    def test_bounds_consistency(self):
        """Test bounds are consistently handled."""
        functions = [
            PowellSingularFunction(),
            PowellSingular2Function(),
            PowellSumFunction(dimension=3),
        ]

        for func in functions:
            assert func.initialization_bounds is not None
            assert func.operational_bounds is not None
            assert len(func.initialization_bounds.low) == func.dimension
            assert len(func.initialization_bounds.high) == func.dimension
