"""
Tests for the Griewank function.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal import GriewankFunction
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.decorators import Biased

class TestGriewankFunction:
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        func = GriewankFunction(dimension=3)
        assert func.dimension == 3
        np.testing.assert_allclose(func.initialization_bounds.low, [-600, -600, -600])
        np.testing.assert_allclose(func.initialization_bounds.high, [600, 600, 600])
        np.testing.assert_allclose(func.operational_bounds.low, [-600, -600, -600])
        np.testing.assert_allclose(func.operational_bounds.high, [600, 600, 600])

        custom_init_bounds = Bounds(low=np.array([-10, -20, -30]), high=np.array([10, 20, 30]), mode=BoundModeEnum.INITIALIZATION)
        custom_oper_bounds = Bounds(low=np.array([-5, -15, -25]), high=np.array([5, 15, 25]), mode=BoundModeEnum.OPERATIONAL)
        func = GriewankFunction(dimension=3, initialization_bounds=custom_init_bounds, operational_bounds=custom_oper_bounds)
        np.testing.assert_allclose(func.initialization_bounds.low, [-10, -20, -30])
        np.testing.assert_allclose(func.initialization_bounds.high, [10, 20, 30])
        np.testing.assert_allclose(func.operational_bounds.low, [-5, -15, -25])
        np.testing.assert_allclose(func.operational_bounds.high, [5, 15, 25])

    def test_global_minimum(self):
        """Test the function at its global minimum."""
        for dim in [2, 3, 5]:
            min_point, min_value = GriewankFunction.get_global_minimum(dim)
            func = GriewankFunction(dimension=dim)
            np.testing.assert_allclose(min_point, np.zeros(dim))
            assert min_value == 0.0
            np.testing.assert_allclose(func.evaluate(min_point), min_value)
            bias_value = 7.0
            biased_func = Biased(func, bias=bias_value)
            np.testing.assert_allclose(biased_func.evaluate(min_point), min_value + bias_value)

    def test_function_values(self):
        """Test function values at specific points."""
        func = GriewankFunction(dimension=2)
        # At origin
        assert np.isclose(func.evaluate(np.zeros(2)), 0.0)
        # At [1, 1]
        x = np.array([1.0, 1.0])
        expected = 1.0 + (1.0 + 1.0) * 0.00025 - np.cos(1.0/1.0) * np.cos(1.0/np.sqrt(2))
        assert np.isclose(func.evaluate(x), expected)
        # With bias
        bias_value = 2.5
        biased_func = Biased(func, bias=bias_value)
        assert np.isclose(biased_func.evaluate(x), expected + bias_value)

    def test_evaluate_batch(self):
        """Test the batch evaluation method."""
        func = GriewankFunction(dimension=2)
        X = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 3.0]
        ])
        expected = np.array([
            func.evaluate(X[0]),
            func.evaluate(X[1]),
            func.evaluate(X[2])
        ])
        np.testing.assert_allclose(func.evaluate_batch(X), expected)
        # With bias
        bias_value = 1.5
        biased_func = Biased(func, bias=bias_value)
        biased_results = biased_func.evaluate_batch(X)
        np.testing.assert_allclose(biased_results, expected + bias_value)

    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        func = GriewankFunction(dimension=2)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]])) 