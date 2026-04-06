"""
Tests for Alpine function family.

Following TDD approach with comprehensive test coverage for all Alpine variants.
Tests validate mathematical correctness, bounds handling, and edge cases.
"""

import numpy as np
import pytest

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.functions.benchmark.alpine import Alpine1Function, Alpine2Function


class TestAlpine1Function:
    """Test Alpine 1 function (2D non-scalable, non-differentiable)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = Alpine1Function()
        assert func.dimension == 2
        assert func.initialization_bounds.low.tolist() == [-10.0, -10.0]
        assert func.initialization_bounds.high.tolist() == [10.0, 10.0]

    def test_dimension_validation(self):
        """Test that non-2D initialization raises ValueError."""
        with pytest.raises(
            ValueError, match="Alpine 1 function is Non-Scalable and requires dimension=2"
        ):
            Alpine1Function(dimension=3)

        with pytest.raises(
            ValueError, match="Alpine 1 function is Non-Scalable and requires dimension=2"
        ):
            Alpine1Function(dimension=1)

    def test_custom_bounds(self):
        """Test initialization with custom bounds."""
        custom_bounds = Bounds(
            low=np.array([-5.0, -5.0]),
            high=np.array([5.0, 5.0]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        func = Alpine1Function(operational_bounds=custom_bounds)
        np.testing.assert_array_equal(func.operational_bounds.low, [-5.0, -5.0])
        np.testing.assert_array_equal(func.operational_bounds.high, [5.0, 5.0])

    def test_global_minimum(self):
        """Test global minimum is at origin with value 0."""
        func = Alpine1Function()
        min_point, min_value = func.get_global_minimum()
        expected_point = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(min_point, expected_point)
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at global minimum."""
        func = Alpine1Function()
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = Alpine1Function()

        # Test point: [1, 1]
        # f(x) = sum(|x_i * sin(x_i) + 0.1 * x_i|) for i=1 to n
        # f([1,1]) = |1 * sin(1) + 0.1 * 1| + |1 * sin(1) + 0.1 * 1|
        #          = 2 * |sin(1) + 0.1| = 2 * |0.8414... + 0.1| ≈ 2 * 0.9415
        x = np.array([1.0, 1.0])
        result = func.evaluate(x)
        expected = 2.0 * abs(np.sin(1.0) + 0.1)
        assert abs(result - expected) < 1e-10

        # Test point: [-1, 1]
        # f([-1,1]) = |-1 * sin(-1) + 0.1 * (-1)| + |1 * sin(1) + 0.1 * 1|
        #           = |(-1) * sin(-1) + 0.1 * (-1)| + |1 * sin(1) + 0.1 * 1|
        #           = |1 * sin(1) - 0.1| + |sin(1) + 0.1|  # since sin(-1) = -sin(1)
        x = np.array([-1.0, 1.0])
        result = func.evaluate(x)
        expected = abs(1.0 * np.sin(1.0) - 0.1) + abs(np.sin(1.0) + 0.1)
        assert abs(result - expected) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        func = Alpine1Function()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [0.5, -0.5]])
        results = func.evaluate_batch(X)

        # Verify individual evaluations match batch
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)

    def test_input_validation(self):
        """Test input validation for wrong dimensions."""
        func = Alpine1Function()

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0]))  # Wrong dimension

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension

    def test_mathematical_properties(self):
        """Test mathematical properties of the function."""
        func = Alpine1Function()

        # Test non-negativity
        test_points = [[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [0.5, -0.5], [5.0, -5.0]]
        for point in test_points:
            result = func.evaluate(np.array(point))
            assert result >= 0.0, f"Function should be non-negative, got {result} at {point}"

    def test_registry_names(self):
        """Test that all registry names work."""
        from pyMOFL.registry import get

        # Test both registry names
        func1 = get("Alpine_1")()
        func2 = get("Alpine1")()

        assert isinstance(func1, Alpine1Function)
        assert isinstance(func2, Alpine1Function)

        # Should produce same results
        x = np.array([1.0, 2.0])
        assert func1.evaluate(x) == func2.evaluate(x)


class TestAlpine2Function:
    """Test Alpine 2 function (scalable, differentiable)."""

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [2, 5, 10, 30]:
            func = Alpine2Function(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim
            assert len(func.initialization_bounds.high) == dim

    def test_default_bounds(self):
        """Test default bounds are correct."""
        func = Alpine2Function(dimension=5)
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(5, 0.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(5, 10.0))

    def test_custom_bounds(self):
        """Test initialization with custom bounds."""
        custom_bounds = Bounds(
            low=np.array([1.0, 1.0, 1.0]),
            high=np.array([5.0, 5.0, 5.0]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        func = Alpine2Function(dimension=3, operational_bounds=custom_bounds)
        np.testing.assert_array_equal(func.operational_bounds.low, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(func.operational_bounds.high, [5.0, 5.0, 5.0])

    def test_global_minimum_various_dimensions(self):
        """Test global minimum for various dimensions."""

        for dim in [2, 5, 10]:
            func = Alpine2Function(dimension=dim)
            min_point, min_value = func.get_global_minimum()

            # Global minimum should be around [7.917, 7.917, ...]
            expected_point = np.full(dim, 7.917)
            np.testing.assert_array_almost_equal(min_point, expected_point, decimal=2)

            # Global minimum value should be (sqrt(x_i) * sin(x_i))^dim
            component_value = np.sqrt(7.917) * np.sin(7.917)
            expected_min = component_value**dim
            assert abs(min_value - expected_min) < 0.1  # Some tolerance due to approximation

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates correctly at global minimum."""
        for dim in [2, 5, 10]:
            func = Alpine2Function(dimension=dim)
            min_point, min_value = func.get_global_minimum()
            result = func.evaluate(min_point)
            assert abs(result - min_value) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = Alpine2Function(dimension=3)

        # Test point: [1, 1, 1]
        # f(x) = prod(sqrt(x_i) * sin(x_i)) for i=1 to n
        # f([1,1,1]) = sqrt(1) * sin(1) * sqrt(1) * sin(1) * sqrt(1) * sin(1)
        #            = (sin(1))^3 ≈ (0.8414)^3 ≈ 0.596
        x = np.array([1.0, 1.0, 1.0])
        result = func.evaluate(x)
        expected = (np.sqrt(1.0) * np.sin(1.0)) ** 3
        assert abs(result - expected) < 1e-10

        # Test point: [4, 9, 16] (perfect squares)
        # f([4,9,16]) = sqrt(4)*sin(4) * sqrt(9)*sin(9) * sqrt(16)*sin(16)
        #             = 2*sin(4) * 3*sin(9) * 4*sin(16)
        x = np.array([4.0, 9.0, 16.0])
        result = func.evaluate(x)
        expected = (2.0 * np.sin(4.0)) * (3.0 * np.sin(9.0)) * (4.0 * np.sin(16.0))
        assert abs(result - expected) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        func = Alpine2Function(dimension=4)
        X = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [4.0, 4.0, 4.0, 4.0],
                [7.917, 7.917, 7.917, 7.917],  # Near global minimum
                [2.0, 3.0, 4.0, 5.0],
            ]
        )
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_array_almost_equal(results, expected)

    def test_mathematical_properties(self):
        """Test mathematical properties."""
        func = Alpine2Function(dimension=5)

        # Test that function can handle positive values
        test_points = [
            np.ones(5),
            np.full(5, 2.0),
            np.full(5, 7.917),  # Near optimum
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        ]

        for point in test_points:
            result = func.evaluate(point)
            assert isinstance(result, (int, float))
            assert not np.isnan(result)
            assert not np.isinf(result)

    def test_zero_handling(self):
        """Test handling of zero values (should give zero result)."""
        func = Alpine2Function(dimension=3)

        # Any zero component should make the product zero
        test_points = [
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 0.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
        ]

        for point in test_points:
            result = func.evaluate(point)
            assert result == 0.0, f"Expected 0.0 for {point}, got {result}"

    def test_input_validation(self):
        """Test input validation."""
        func = Alpine2Function(dimension=4)

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))  # Wrong dimension

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension


class TestAlpineFamilyIntegration:
    """Integration tests for Alpine function family."""

    def test_all_functions_instantiate(self):
        """Test all Alpine functions can be instantiated."""
        functions = [Alpine1Function(), Alpine2Function(dimension=5)]
        for func in functions:
            assert hasattr(func, "evaluate")
            assert hasattr(func, "evaluate_batch")
            assert hasattr(func, "get_global_minimum")

    def test_registry_integration(self):
        """Test functions are properly registered."""
        from pyMOFL.registry import get

        # Test registry names work
        func1 = get("Alpine_1")()
        func2 = get("Alpine1")()
        func3 = get("Alpine_2")(dimension=5)
        func4 = get("Alpine2")(dimension=5)

        assert isinstance(func1, Alpine1Function)
        assert isinstance(func2, Alpine1Function)
        assert isinstance(func3, Alpine2Function)
        assert isinstance(func4, Alpine2Function)

    def test_bounds_consistency(self):
        """Test bounds are consistently handled."""
        functions = [Alpine1Function(), Alpine2Function(dimension=3)]

        for func in functions:
            assert func.initialization_bounds is not None
            assert func.operational_bounds is not None
            assert len(func.initialization_bounds.low) == func.dimension
            assert len(func.initialization_bounds.high) == func.dimension

    def test_mathematical_consistency(self):
        """Test mathematical consistency across function family."""
        # Test that both functions can handle their respective domains
        alpine1 = Alpine1Function()
        alpine2 = Alpine2Function(dimension=2)

        # Alpine1 can handle negative values
        x1 = np.array([-1.0, 1.0])
        result1 = alpine1.evaluate(x1)
        assert isinstance(result1, (int, float))

        # Alpine2 is typically defined for positive domain
        x2 = np.array([1.0, 2.0])
        result2 = alpine2.evaluate(x2)
        assert isinstance(result2, (int, float))

    def test_global_minimum_consistency(self):
        """Test global minimum methods are consistent."""
        for func_class in [Alpine1Function, Alpine2Function]:
            if func_class == Alpine1Function:
                func = func_class()
            else:
                func = func_class(dimension=3)

            min_point, min_value = func.get_global_minimum()
            evaluated_value = func.evaluate(min_point)
            assert abs(evaluated_value - min_value) < 1e-10, (
                f"Global minimum inconsistent for {func_class.__name__}"
            )
