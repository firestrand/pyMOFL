"""
Tests for KatsuuraFunction.

Validates mathematical correctness, bounds handling, batch evaluation,
and structural properties of the Katsuura benchmark function.

f(x) = (10/D^2) * prod(1 + i * sum(|2^j * x_i - round(2^j * x_i)| / 2^j))^(10/D^1.2) - 10/D^2
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.katsuura import KatsuuraFunction


class TestKatsuuraFunction:
    """Tests for KatsuuraFunction."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = KatsuuraFunction(dimension=10)
        assert func.dimension == 10
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(10, -100.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(10, 100.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(10, -100.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(10, 100.0))

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [1, 2, 5, 10]:
            func = KatsuuraFunction(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim

    def test_precomputed_constants(self):
        """Test that precomputed constants are correct."""
        dim = 5
        func = KatsuuraFunction(dimension=dim)
        assert func._norm == pytest.approx(10.0 / 25.0)
        assert func._exp == pytest.approx(10.0 / 5.0**1.2)
        assert len(func._pow2) == 32
        assert func._pow2[0] == pytest.approx(2.0)
        assert func._pow2[31] == pytest.approx(2.0**32)
        assert len(func._indices) == 5
        np.testing.assert_array_equal(func._indices, [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_global_minimum(self):
        """Test that function evaluates to 0 at the origin.

        At x=0: 2^j * 0 = 0, round(0) = 0, so inner sums are all 0.
        Each factor in product = (1 + i*0)^exp = 1.
        Product = 1. f = norm * 1 - norm = 0.
        """
        for dim in [1, 2, 5, 10]:
            func = KatsuuraFunction(dimension=dim)
            x_opt = np.zeros(dim)
            result = func.evaluate(x_opt)
            assert abs(result) < 1e-12, f"Expected 0.0 at origin for dim={dim}, got {result}"

    def test_get_global_minimum(self):
        """Test get_global_minimum static method returns correct values."""
        for dim in [1, 2, 5, 10, 30]:
            min_point, min_value = KatsuuraFunction.get_global_minimum(dim)
            np.testing.assert_array_equal(min_point, np.zeros(dim))
            assert min_value == 0.0

    def test_get_global_minimum_consistency(self):
        """Test that evaluating at the returned global minimum gives the stated value."""
        for dim in [2, 5, 10]:
            func = KatsuuraFunction(dimension=dim)
            min_point, min_value = KatsuuraFunction.get_global_minimum(dim)
            result = func.evaluate(min_point)
            assert abs(result - min_value) < 1e-12

    def test_known_value_at_integer(self):
        """Test that integer inputs give f=0.

        For any integer x_i, 2^j * x_i is an integer, so
        |2^j * x_i - round(2^j * x_i)| = 0 for all j.
        Therefore each inner sum = 0, product = 1, f = norm - norm = 0.
        """
        func = KatsuuraFunction(dimension=3)
        for val in [1.0, -2.0, 5.0, -10.0]:
            x = np.full(3, val)
            result = func.evaluate(x)
            assert abs(result) < 1e-10, f"Expected ~0 at integer point {val}, got {result}"

    def test_known_value_non_integer(self):
        """Test that non-integer inputs give positive values."""
        func = KatsuuraFunction(dimension=3)
        x = np.array([0.5, 0.5, 0.5])
        result = func.evaluate(x)
        # 0.5 is a dyadic rational: 2^1 * 0.5 = 1 (integer), 2^j * 0.5 is integer for j>=1
        # So actually this should also give 0
        # Let's use a non-dyadic-rational value instead
        x = np.array([0.3, 0.7, 0.1])
        result = func.evaluate(x)
        assert result > 0.0, f"Expected positive value for non-integer input, got {result}"

    def test_evaluate_batch(self):
        """Test that batch evaluation matches individual evaluation."""
        func = KatsuuraFunction(dimension=5)
        rng = np.random.default_rng(42)
        X = rng.uniform(-5, 5, size=(10, 5))
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test that batch output has correct shape."""
        func = KatsuuraFunction(dimension=5)
        X = np.zeros((7, 5))
        results = func.evaluate_batch(X)
        assert results.shape == (7,), f"Expected shape (7,), got {results.shape}"

    def test_dimension_validation(self):
        """Test that wrong input shape raises an error."""
        func = KatsuuraFunction(dimension=5)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))  # Wrong dimension

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension

    def test_non_negativity(self):
        """Test that function values are always non-negative for random inputs.

        The product of factors >= 1 means the product >= 1, so
        norm * product - norm >= norm * 1 - norm = 0.
        """
        func = KatsuuraFunction(dimension=5)
        rng = np.random.default_rng(123)
        for _ in range(50):
            x = rng.uniform(-5, 5, size=5)
            result = func.evaluate(x)
            assert result >= -1e-10, f"Function should be non-negative, got {result}"

    def test_high_dimensionality(self):
        """Test function works correctly in higher dimensions."""
        func = KatsuuraFunction(dimension=30)
        x = np.zeros(30)
        result = func.evaluate(x)
        assert abs(result) < 1e-10

        # Non-zero point should produce finite result
        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, size=30)
        result = func.evaluate(x)
        assert np.isfinite(result), f"Expected finite result, got {result}"

    def test_registry_names(self):
        """Test that all registry names work."""
        from pyMOFL.registry import get

        func1 = get("katsuura")(dimension=3)
        func2 = get("Katsuura")(dimension=3)
        assert isinstance(func1, KatsuuraFunction)
        assert isinstance(func2, KatsuuraFunction)

        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) == func2.evaluate(x)
