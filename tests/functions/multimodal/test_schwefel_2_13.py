"""
Tests for the Schwefel's Problem 2.13 function.
"""

import pytest
import numpy as np
from pyMOFL.functions.multimodal.schwefel_2_13 import SchwefelFunction213
from pyMOFL.decorators import Biased
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum

class TestSchwefelFunction213:
    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        func = SchwefelFunction213(dimension=3)
        assert func.dimension == 3
        np.testing.assert_allclose(func.initialization_bounds.low, [-np.pi, -np.pi, -np.pi])
        np.testing.assert_allclose(func.initialization_bounds.high, [np.pi, np.pi, np.pi])
        np.testing.assert_allclose(func.operational_bounds.low, [-np.pi, -np.pi, -np.pi])
        np.testing.assert_allclose(func.operational_bounds.high, [np.pi, np.pi, np.pi])

        custom_init_bounds = Bounds(low=np.array([-1, -2, -3]), high=np.array([1, 2, 3]), mode=BoundModeEnum.INITIALIZATION)
        custom_oper_bounds = Bounds(low=np.array([-0.1, -0.2, -0.3]), high=np.array([0.1, 0.2, 0.3]), mode=BoundModeEnum.OPERATIONAL)
        func = SchwefelFunction213(dimension=3, initialization_bounds=custom_init_bounds, operational_bounds=custom_oper_bounds)
        np.testing.assert_allclose(func.initialization_bounds.low, [-1, -2, -3])
        np.testing.assert_allclose(func.initialization_bounds.high, [1, 2, 3])
        np.testing.assert_allclose(func.operational_bounds.low, [-0.1, -0.2, -0.3])
        np.testing.assert_allclose(func.operational_bounds.high, [0.1, 0.2, 0.3])

    def test_global_minimum(self):
        """Test the function at its global minimum."""
        dim = 3
        # Use a fixed random seed for reproducibility
        rng = np.random.default_rng(42)
        a = rng.integers(-100, 101, (dim, dim))
        b = rng.integers(-100, 101, (dim, dim))
        alpha = rng.uniform(-np.pi, np.pi, dim)
        func = SchwefelFunction213(dimension=dim, a=a, b=b, alpha=alpha)
        min_point, min_value = SchwefelFunction213.get_global_minimum(dim, a, b, alpha)
        np.testing.assert_allclose(min_point, alpha)
        assert min_value == 0.0
        np.testing.assert_allclose(func.evaluate(min_point), min_value, atol=1e-10)
        bias_value = 7.0
        biased_func = Biased(func, bias=bias_value)
        np.testing.assert_allclose(biased_func.evaluate(min_point), min_value + bias_value, atol=1e-10)

    def test_function_values(self):
        """Test function values at specific points."""
        dim = 2
        rng = np.random.default_rng(123)
        a = rng.integers(-100, 101, (dim, dim))
        b = rng.integers(-100, 101, (dim, dim))
        alpha = rng.uniform(-np.pi, np.pi, dim)
        func = SchwefelFunction213(dimension=dim, a=a, b=b, alpha=alpha)
        # At global minimum
        assert np.isclose(func.evaluate(alpha), 0.0, atol=1e-10)
        # At a different point
        x = alpha + 0.1
        val = func.evaluate(x)
        assert isinstance(val, float)
        # With bias
        bias_value = 2.5
        biased_func = Biased(func, bias=bias_value)
        assert np.isclose(biased_func.evaluate(x), val + bias_value)

    def test_evaluate_batch(self):
        """Test the batch evaluation method."""
        dim = 2
        rng = np.random.default_rng(1234)
        a = rng.integers(-100, 101, (dim, dim))
        b = rng.integers(-100, 101, (dim, dim))
        alpha = rng.uniform(-np.pi, np.pi, dim)
        func = SchwefelFunction213(dimension=dim, a=a, b=b, alpha=alpha)
        X = np.array([
            alpha,
            alpha + 0.1,
            alpha - 0.2
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
        func = SchwefelFunction213(dimension=2)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]])) 