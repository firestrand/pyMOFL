"""
Tests for McCormickFunction.

f(x,y) = sin(x + y) + (x - y)² - 1.5x + 2.5y + 1
Domain: [-1.5, 4] × [-3, 4], Global minimum: f(-0.54719, -1.54719) ≈ -1.9133
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.mccormick import McCormickFunction


class TestMcCormickFunction:
    """Tests for McCormickFunction."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = McCormickFunction()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, np.array([-1.5, -3.0]))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.array([4.0, 4.0]))

    def test_asymmetric_bounds(self):
        """Test that bounds are asymmetric (different per dimension)."""
        func = McCormickFunction()
        assert func.operational_bounds.low[0] != func.operational_bounds.low[1]
        assert func.operational_bounds.high[0] == func.operational_bounds.high[1]

    def test_dimension_must_be_2(self):
        """Test that non-2D dimensions raise ValueError."""
        with pytest.raises(ValueError, match="dimension=2"):
            McCormickFunction(dimension=3)
        with pytest.raises(ValueError, match="dimension=2"):
            McCormickFunction(dimension=1)

    def test_global_minimum(self):
        """Test function value at global minimum."""
        func = McCormickFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-3)

    def test_get_global_minimum(self):
        """Test get_global_minimum returns correct point and value."""
        func = McCormickFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, np.array([-0.54719, -1.54719]), decimal=4)
        assert min_value == pytest.approx(-1.9133, abs=1e-3)

    def test_value_at_origin(self):
        """Test function value at origin — hand-computed."""
        func = McCormickFunction()
        # At (0, 0): sin(0) + 0² - 0 + 0 + 1 = 1
        result = func.evaluate(np.array([0.0, 0.0]))
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_known_value_hand_computed(self):
        """Test at (1, 1) with hand computation."""
        func = McCormickFunction()
        x1, x2 = 1.0, 1.0
        expected = np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1
        result = func.evaluate(np.array([x1, x2]))
        assert result == pytest.approx(expected, abs=1e-10)

    def test_known_value_negative_point(self):
        """Test at (-1, -2) with hand computation."""
        func = McCormickFunction()
        x1, x2 = -1.0, -2.0
        expected = np.sin(x1 + x2) + (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 1
        result = func.evaluate(np.array([x1, x2]))
        assert result == pytest.approx(expected, abs=1e-10)

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual."""
        func = McCormickFunction()
        rng = np.random.default_rng(42)
        X = rng.uniform(-1.5, 4, size=(10, 2))
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test batch output shape."""
        func = McCormickFunction()
        X = np.zeros((5, 2))
        results = func.evaluate_batch(X)
        assert results.shape == (5,)

    def test_dimension_validation(self):
        """Test wrong input shape raises error."""
        func = McCormickFunction()
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))

    def test_registry_name(self):
        """Test registry name works."""
        from pyMOFL.registry import get

        func = get("McCormick")(dimension=2)
        assert isinstance(func, McCormickFunction)
