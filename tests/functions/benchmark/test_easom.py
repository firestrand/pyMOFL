"""
Tests for EasomFunction.

f(x,y) = -cos(x)cos(y)exp(-(x-π)² - (y-π)²)
Domain: [-100, 100]², Global minimum: f(π, π) = -1
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.easom import EasomFunction


class TestEasomFunction:
    """Tests for EasomFunction."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = EasomFunction()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, np.array([-100.0, -100.0]))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.array([100.0, 100.0]))

    def test_dimension_must_be_2(self):
        """Test that non-2D dimensions raise ValueError."""
        with pytest.raises(ValueError, match="dimension=2"):
            EasomFunction(dimension=3)
        with pytest.raises(ValueError, match="dimension=2"):
            EasomFunction(dimension=1)

    def test_global_minimum(self):
        """Test function value at global minimum (π, π)."""
        func = EasomFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)
        assert min_value == -1.0

    def test_get_global_minimum(self):
        """Test get_global_minimum returns correct point and value."""
        func = EasomFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, np.array([np.pi, np.pi]))
        assert min_value == -1.0

    def test_value_at_origin(self):
        """Test function value at origin — nearly zero due to exp decay."""
        func = EasomFunction()
        result = func.evaluate(np.array([0.0, 0.0]))
        # cos(0)*cos(0)*exp(-(π²+π²)) = exp(-2π²) ≈ 3.5e-9
        expected = -np.exp(-(np.pi**2 + np.pi**2))
        assert result == pytest.approx(expected, abs=1e-12)

    def test_value_far_from_minimum(self):
        """Test that function is nearly zero far from (π, π)."""
        func = EasomFunction()
        # At (50, 50): exp decay makes value extremely close to 0
        result = func.evaluate(np.array([50.0, 50.0]))
        assert abs(result) < 1e-100

    def test_known_values(self):
        """Test at specific known points."""
        func = EasomFunction()
        # At (π, 0): cos(π)cos(0)exp(-(0)-(π²)) = -(-1)(1)exp(-π²) = exp(-π²)
        result = func.evaluate(np.array([np.pi, 0.0]))
        expected = -np.cos(np.pi) * np.cos(0.0) * np.exp(-(np.pi**2))
        assert result == pytest.approx(expected, abs=1e-12)

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual."""
        func = EasomFunction()
        rng = np.random.default_rng(42)
        X = rng.uniform(-10, 10, size=(10, 2))
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test batch output shape."""
        func = EasomFunction()
        X = np.zeros((5, 2))
        results = func.evaluate_batch(X)
        assert results.shape == (5,)

    def test_dimension_validation(self):
        """Test wrong input shape raises error."""
        func = EasomFunction()
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))

    def test_negative_near_minimum(self):
        """Test function is negative near the global minimum."""
        func = EasomFunction()
        # Near (π, π), function should be negative
        x = np.array([np.pi + 0.1, np.pi - 0.1])
        assert func.evaluate(x) < 0

    def test_registry_name(self):
        """Test registry name works."""
        from pyMOFL.registry import get

        func = get("Easom")(dimension=2)
        assert isinstance(func, EasomFunction)
