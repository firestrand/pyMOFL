"""
Tests for GoldsteinPriceFunction.

f(x,y) = [1 + (x+y+1)²(19-14x+3x²-14y+6xy+3y²)] ×
         [30 + (2x-3y)²(18-32x+12x²+48y-36xy+27y²)]
Domain: [-2, 2]², Global minimum: f(0, -1) = 3
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.goldstein_price import GoldsteinPriceFunction


class TestGoldsteinPriceFunction:
    """Tests for GoldsteinPriceFunction."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = GoldsteinPriceFunction()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, np.array([-2.0, -2.0]))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.array([2.0, 2.0]))

    def test_dimension_must_be_2(self):
        """Test that non-2D dimensions raise ValueError."""
        with pytest.raises(ValueError, match="dimension=2"):
            GoldsteinPriceFunction(dimension=3)
        with pytest.raises(ValueError, match="dimension=2"):
            GoldsteinPriceFunction(dimension=1)

    def test_global_minimum(self):
        """Test function value at global minimum (0, -1) = 3."""
        func = GoldsteinPriceFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)
        assert min_value == 3.0

    def test_get_global_minimum(self):
        """Test get_global_minimum returns correct point and value."""
        func = GoldsteinPriceFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.array([0.0, -1.0]))
        assert min_value == 3.0

    def test_value_at_origin(self):
        """Test function value at origin — hand-computed."""
        func = GoldsteinPriceFunction()
        # At (0, 0):
        # term1 = 1 + (0+0+1)²(19-0+0-0+0+0) = 1 + 1*19 = 20
        # term2 = 30 + (0-0)²(18-0+0+0-0+0) = 30 + 0 = 30
        # f = 20 * 30 = 600
        result = func.evaluate(np.array([0.0, 0.0]))
        assert result == pytest.approx(600.0, abs=1e-10)

    def test_known_value_at_local_minima(self):
        """Test function at known local minimum (1.2, 0.8)."""
        func = GoldsteinPriceFunction()
        # (1.2, 0.8) is a known local minimum with f ≈ 840
        result = func.evaluate(np.array([1.2, 0.8]))
        assert result == pytest.approx(840.0, rel=0.01)

    def test_known_value_hand_computed(self):
        """Test at (1, 1) with hand computation."""
        func = GoldsteinPriceFunction()
        x1, x2 = 1.0, 1.0
        inner1 = 19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        term1 = 1 + (x1 + x2 + 1) ** 2 * inner1
        inner2 = 18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        term2 = 30 + (2 * x1 - 3 * x2) ** 2 * inner2
        expected = term1 * term2
        result = func.evaluate(np.array([x1, x2]))
        assert result == pytest.approx(expected, abs=1e-10)

    def test_positive_definite(self):
        """Test that function is always positive (minimum is 3 > 0)."""
        func = GoldsteinPriceFunction()
        rng = np.random.default_rng(42)
        X = rng.uniform(-2, 2, size=(50, 2))
        for x in X:
            assert func.evaluate(x) > 0

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual."""
        func = GoldsteinPriceFunction()
        rng = np.random.default_rng(42)
        X = rng.uniform(-2, 2, size=(10, 2))
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test batch output shape."""
        func = GoldsteinPriceFunction()
        X = np.zeros((5, 2))
        results = func.evaluate_batch(X)
        assert results.shape == (5,)

    def test_dimension_validation(self):
        """Test wrong input shape raises error."""
        func = GoldsteinPriceFunction()
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))

    def test_registry_names(self):
        """Test all registry names work."""
        from pyMOFL.registry import get

        func1 = get("GoldsteinPrice")(dimension=2)
        func2 = get("Goldstein_Price")(dimension=2)
        assert isinstance(func1, GoldsteinPriceFunction)
        assert isinstance(func2, GoldsteinPriceFunction)

        x = np.array([0.5, -0.5])
        assert func1.evaluate(x) == func2.evaluate(x)
