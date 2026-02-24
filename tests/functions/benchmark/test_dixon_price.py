"""
Tests for Dixon-Price function.

Following TDD approach with comprehensive test coverage.
Tests validate mathematical correctness, bounds handling, and edge cases.

f(x) = (x_1 - 1)^2 + sum_{i=2}^{D} i*(2*x_i^2 - x_{i-1})^2
Domain: [-10, 10]^D
Global minimum: f* = 0 at x_i = 2^(-(2^i - 2)/2^i) for i = 1..D (1-based)
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.dixon_price import DixonPriceFunction


class TestDixonPriceFunction:
    """Tests for Dixon-Price benchmark function."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = DixonPriceFunction(dimension=5)
        assert func.dimension == 5
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(5, -10.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(5, 10.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(5, -10.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(5, 10.0))

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [2, 5, 10, 30]:
            func = DixonPriceFunction(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim
            assert len(func.initialization_bounds.high) == dim

    def test_global_minimum(self):
        """Test function evaluates to 0 at the global minimum."""
        for dim in [2, 5, 10]:
            func = DixonPriceFunction(dimension=dim)
            min_point, _min_value = DixonPriceFunction.get_global_minimum(dim)
            result = func.evaluate(min_point)
            assert abs(result) < 1e-8, f"Expected ~0 at optimum for dim={dim}, got {result}"

    def test_get_global_minimum(self):
        """Test static get_global_minimum method returns correct point and value."""
        for dim in [2, 5, 10]:
            min_point, min_value = DixonPriceFunction.get_global_minimum(dim)
            assert min_value == 0.0
            assert len(min_point) == dim

        # D=2: x_1 = 2^(-(2^1 - 2)/2^1) = 2^0 = 1
        #       x_2 = 2^(-(2^2 - 2)/2^2) = 2^(-2/4) = 2^(-0.5)
        min_point_2, _ = DixonPriceFunction.get_global_minimum(2)
        assert abs(min_point_2[0] - 1.0) < 1e-10
        assert abs(min_point_2[1] - 2.0 ** (-0.5)) < 1e-10

    def test_get_global_minimum_consistency(self):
        """Test that evaluating at the returned global minimum gives the returned value."""
        for dim in [2, 3, 5]:
            func = DixonPriceFunction(dimension=dim)
            min_point, min_value = DixonPriceFunction.get_global_minimum(dim)
            evaluated = func.evaluate(min_point)
            assert abs(evaluated - min_value) < 1e-8

    def test_known_values(self):
        """Test function evaluation at hand-computed known points."""
        func2 = DixonPriceFunction(dimension=2)

        # f(1,1,...,1) for D=2:
        # term1 = (1-1)^2 = 0
        # term2 = 2*(2*1^2 - 1)^2 = 2*(2-1)^2 = 2*1 = 2
        # total = 0 + 2 = 2
        result_ones = func2.evaluate(np.ones(2))
        assert abs(result_ones - 2.0) < 1e-10, f"f(1,1) expected 2.0, got {result_ones}"

        # f(1,1,1) for D=3:
        # term1 = 0
        # i=2: 2*(2*1-1)^2 = 2
        # i=3: 3*(2*1-1)^2 = 3
        # total = 0 + 2 + 3 = 5
        func3 = DixonPriceFunction(dimension=3)
        result_ones_3 = func3.evaluate(np.ones(3))
        assert abs(result_ones_3 - 5.0) < 1e-10, f"f(1,1,1) expected 5.0, got {result_ones_3}"

        # f(0,...,0) for D=2:
        # term1 = (0-1)^2 = 1
        # i=2: 2*(2*0-0)^2 = 2*0 = 0
        # total = 1
        result_zeros = func2.evaluate(np.zeros(2))
        assert abs(result_zeros - 1.0) < 1e-10, f"f(0,0) expected 1.0, got {result_zeros}"

        # f at optimum for D=2: x=(1, 1/sqrt(2)) should give 0
        x_opt = np.array([1.0, 2.0 ** (-0.5)])
        result_opt = func2.evaluate(x_opt)
        assert abs(result_opt) < 1e-10, f"f at optimum expected 0, got {result_opt}"

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluations."""
        func = DixonPriceFunction(dimension=3)
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 2.0 ** (-0.5), 2.0 ** (-0.75)],
                [0.5, -0.5, 0.2],
            ]
        )
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test that batch evaluation returns correct shape."""
        func = DixonPriceFunction(dimension=4)
        X = np.random.uniform(-10, 10, size=(7, 4))
        results = func.evaluate_batch(X)
        assert results.shape == (7,), f"Expected shape (7,), got {results.shape}"

    def test_dimension_validation(self):
        """Test input validation rejects wrong-shaped inputs."""
        func = DixonPriceFunction(dimension=3)

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0]))  # Too few

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0, 4.0]))  # Too many

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0]]))  # Wrong columns

    def test_non_negativity(self):
        """Test that Dixon-Price function is always non-negative (sum of squares)."""
        func = DixonPriceFunction(dimension=5)
        rng = np.random.default_rng(42)
        for _ in range(50):
            x = rng.uniform(-10, 10, size=5)
            result = func.evaluate(x)
            assert result >= -1e-10, f"Expected non-negative, got {result} at {x}"

    def test_non_separability(self):
        """Test that Dixon-Price is non-separable (x_{i-1} appears in term for i)."""
        func = DixonPriceFunction(dimension=3)
        # f(1, a, 0) vs f(2, a, 0) should differ due to cross-term 2*(2*a^2 - x1)^2
        x1 = np.array([1.0, 0.5, 0.0])
        x2 = np.array([2.0, 0.5, 0.0])
        # In x1: i=2 term = 2*(2*0.25 - 1)^2 = 2*(0.5-1)^2 = 2*0.25 = 0.5
        # In x2: i=2 term = 2*(2*0.25 - 2)^2 = 2*(0.5-2)^2 = 2*2.25 = 4.5
        # So they must differ
        f1 = func.evaluate(x1)
        f2 = func.evaluate(x2)
        assert f1 != f2, "Dixon-Price should be non-separable"

    def test_optimum_location_formula(self):
        """Test the specific optimum formula x_i = 2^(-(2^i-2)/2^i)."""
        for dim in [2, 3, 4, 5]:
            i = np.arange(1, dim + 1, dtype=float)
            x_opt = 2.0 ** (-(2.0**i - 2.0) / 2.0**i)
            func = DixonPriceFunction(dimension=dim)
            result = func.evaluate(x_opt)
            assert abs(result) < 1e-8, f"Optimum formula failed for dim={dim}, f={result}"

    def test_scalability(self):
        """Test function works across multiple dimensions."""
        for dim in [2, 10, 50]:
            func = DixonPriceFunction(dimension=dim)
            min_point, _ = DixonPriceFunction.get_global_minimum(dim)
            result = func.evaluate(min_point)
            assert abs(result) < 1e-6


class TestDixonPriceRegistry:
    """Test Dixon-Price registry integration."""

    def test_registry_names(self):
        """Test that registered names resolve to DixonPriceFunction."""
        from pyMOFL.registry import get

        func1 = get("DixonPrice")(dimension=3)
        func2 = get("dixon_price")(dimension=3)
        assert isinstance(func1, DixonPriceFunction)
        assert isinstance(func2, DixonPriceFunction)

        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) == func2.evaluate(x)
