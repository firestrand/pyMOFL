"""
Tests for DifferentPowersFunction.

Validates mathematical correctness, bounds handling, batch evaluation,
and structural properties of the Different Powers benchmark function.

f(x) = sqrt(sum(|x_i|^(2 + 4*(i-1)/(D-1)) for i=1..D))
Exponents range from 2 (i=1) to 6 (i=D). For D=1, exponent is 2.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.different_powers import DifferentPowersFunction


class TestDifferentPowersFunction:
    """Tests for DifferentPowersFunction."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = DifferentPowersFunction(dimension=10)
        assert func.dimension == 10
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(10, -100.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(10, 100.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(10, -100.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(10, 100.0))

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [1, 2, 5, 10, 30]:
            func = DifferentPowersFunction(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim

    def test_global_minimum(self):
        """Test that function evaluates to 0 at the origin."""
        for dim in [1, 2, 5, 10]:
            func = DifferentPowersFunction(dimension=dim)
            x_opt = np.zeros(dim)
            result = func.evaluate(x_opt)
            assert result == 0.0, f"Expected 0.0 at origin for dim={dim}, got {result}"

    def test_get_global_minimum(self):
        """Test get_global_minimum method returns correct values."""
        for dim in [1, 2, 5, 10, 30]:
            func = DifferentPowersFunction(dimension=dim)
            min_point, min_value = func.get_global_minimum()
            np.testing.assert_array_equal(min_point, np.zeros(dim))
            assert min_value == 0.0

    def test_get_global_minimum_consistency(self):
        """Test that evaluating at the returned global minimum gives the stated value."""
        for dim in [2, 5, 10]:
            func = DifferentPowersFunction(dimension=dim)
            min_point, min_value = func.get_global_minimum()
            result = func.evaluate(min_point)
            assert abs(result - min_value) < 1e-12

    def test_exponents_dim2(self):
        """Test that exponents for D=2 are [2, 6]."""
        func = DifferentPowersFunction(dimension=2)
        expected = np.array([2.0, 6.0])
        np.testing.assert_array_almost_equal(func._exponents, expected)

    def test_exponents_dim3(self):
        """Test that exponents for D=3 are [2, 4, 6]."""
        func = DifferentPowersFunction(dimension=3)
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_array_almost_equal(func._exponents, expected)

    def test_exponents_dim1(self):
        """Test that exponent for D=1 is [2]."""
        func = DifferentPowersFunction(dimension=1)
        expected = np.array([2.0])
        np.testing.assert_array_almost_equal(func._exponents, expected)

    def test_known_values_all_ones_dim2(self):
        """Test f(1,1) for D=2: sqrt(|1|^2 + |1|^6) = sqrt(1+1) = sqrt(2)."""
        func = DifferentPowersFunction(dimension=2)
        x = np.ones(2)
        result = func.evaluate(x)
        expected = np.sqrt(2.0)
        assert abs(result - expected) < 1e-12, f"Expected {expected}, got {result}"

    def test_known_values_all_ones_dim3(self):
        """Test f(1,1,1) for D=3: sqrt(|1|^2 + |1|^4 + |1|^6) = sqrt(3)."""
        func = DifferentPowersFunction(dimension=3)
        x = np.ones(3)
        result = func.evaluate(x)
        expected = np.sqrt(3.0)
        assert abs(result - expected) < 1e-12, f"Expected {expected}, got {result}"

    def test_known_values_single_component(self):
        """Test with only one non-zero component at a time for D=3."""
        func = DifferentPowersFunction(dimension=3)

        # Only x1 = 2: sqrt(|2|^2) = sqrt(4) = 2
        x = np.array([2.0, 0.0, 0.0])
        result = func.evaluate(x)
        expected = np.sqrt(2.0**2)
        assert abs(result - expected) < 1e-12

        # Only x2 = 2: sqrt(|2|^4) = sqrt(16) = 4
        x = np.array([0.0, 2.0, 0.0])
        result = func.evaluate(x)
        expected = np.sqrt(2.0**4)
        assert abs(result - expected) < 1e-12

        # Only x3 = 2: sqrt(|2|^6) = sqrt(64) = 8
        x = np.array([0.0, 0.0, 2.0])
        result = func.evaluate(x)
        expected = np.sqrt(2.0**6)
        assert abs(result - expected) < 1e-12

    def test_known_values_negative_inputs(self):
        """Test that absolute value is correctly applied."""
        func = DifferentPowersFunction(dimension=2)
        x_pos = np.array([2.0, 3.0])
        x_neg = np.array([-2.0, -3.0])
        # |x| is used, so sign should not matter
        assert func.evaluate(x_pos) == pytest.approx(func.evaluate(x_neg))

    def test_evaluate_batch(self):
        """Test that batch evaluation matches individual evaluation."""
        func = DifferentPowersFunction(dimension=5)
        rng = np.random.default_rng(42)
        X = rng.uniform(-10, 10, size=(10, 5))
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test that batch output has correct shape."""
        func = DifferentPowersFunction(dimension=5)
        X = np.zeros((7, 5))
        results = func.evaluate_batch(X)
        assert results.shape == (7,), f"Expected shape (7,), got {results.shape}"

    def test_dimension_validation(self):
        """Test that wrong input shape raises an error."""
        func = DifferentPowersFunction(dimension=5)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))  # Wrong dimension

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension

    def test_non_negativity(self):
        """Test that function values are always non-negative."""
        func = DifferentPowersFunction(dimension=5)
        rng = np.random.default_rng(123)
        for _ in range(50):
            x = rng.uniform(-10, 10, size=5)
            result = func.evaluate(x)
            assert result >= 0.0, f"Function should be non-negative, got {result}"

    def test_asymmetric_sensitivity(self):
        """Test that later dimensions are more sensitive (higher exponents)."""
        func = DifferentPowersFunction(dimension=3)
        val = 2.0

        # Perturb each dimension independently
        results = []
        for i in range(3):
            x = np.zeros(3)
            x[i] = val
            results.append(func.evaluate(x))

        # With exponents [2, 4, 6], perturbing later dims with val > 1
        # gives larger values: sqrt(4) < sqrt(16) < sqrt(64)
        assert results[0] < results[1] < results[2], f"Expected increasing sensitivity: {results}"

    def test_registry_names(self):
        """Test that all registry names work."""
        from pyMOFL.registry import get

        func1 = get("different_powers")(dimension=3)
        func2 = get("DifferentPowers")(dimension=3)
        assert isinstance(func1, DifferentPowersFunction)
        assert isinstance(func2, DifferentPowersFunction)

        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) == func2.evaluate(x)
