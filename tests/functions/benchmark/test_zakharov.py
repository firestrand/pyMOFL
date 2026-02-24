"""
Tests for Zakharov function.

Following TDD approach with comprehensive test coverage.
Tests validate mathematical correctness, bounds handling, and edge cases.

f(x) = sum(x_i^2) + (sum(0.5*i*x_i))^2 + (sum(0.5*i*x_i))^4   (1-based i)
Domain: [-5, 10]^D
Global minimum: f(0, ..., 0) = 0
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.zakharov import ZakharovFunction


class TestZakharovFunction:
    """Tests for Zakharov benchmark function."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = ZakharovFunction(dimension=5)
        assert func.dimension == 5
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(5, -5.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(5, 10.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(5, -5.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(5, 10.0))

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [2, 5, 10, 30]:
            func = ZakharovFunction(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim
            assert len(func.initialization_bounds.high) == dim

    def test_global_minimum(self):
        """Test function evaluates to 0 at the global minimum (origin)."""
        for dim in [2, 5, 10]:
            func = ZakharovFunction(dimension=dim)
            x_opt = np.zeros(dim)
            result = func.evaluate(x_opt)
            assert abs(result) < 1e-10, f"Expected 0 at origin for dim={dim}, got {result}"

    def test_get_global_minimum(self):
        """Test static get_global_minimum method returns correct point and value."""
        for dim in [2, 5, 10]:
            min_point, min_value = ZakharovFunction.get_global_minimum(dim)
            np.testing.assert_array_equal(min_point, np.zeros(dim))
            assert min_value == 0.0

    def test_get_global_minimum_consistency(self):
        """Test that evaluating at the returned global minimum gives the returned value."""
        for dim in [2, 5, 10]:
            func = ZakharovFunction(dimension=dim)
            min_point, min_value = ZakharovFunction.get_global_minimum(dim)
            evaluated = func.evaluate(min_point)
            assert abs(evaluated - min_value) < 1e-10

    def test_known_values(self):
        """Test function evaluation at hand-computed known points."""
        func2 = ZakharovFunction(dimension=2)

        # f(1,0) for D=2:
        # sum_sq = 1, weighted = 0.5*1*1 + 0.5*2*0 = 0.5
        # result = 1 + 0.5^2 + 0.5^4 = 1 + 0.25 + 0.0625 = 1.3125
        result_10 = func2.evaluate(np.array([1.0, 0.0]))
        assert abs(result_10 - 1.3125) < 1e-10, f"f(1,0) expected 1.3125, got {result_10}"

        # f(0,1) for D=2:
        # sum_sq = 1, weighted = 0.5*1*0 + 0.5*2*1 = 1.0
        # result = 1 + 1^2 + 1^4 = 1 + 1 + 1 = 3.0
        result_01 = func2.evaluate(np.array([0.0, 1.0]))
        assert abs(result_01 - 3.0) < 1e-10, f"f(0,1) expected 3.0, got {result_01}"

        # f(1,1) for D=2:
        # sum_sq = 2, weighted = 0.5*1*1 + 0.5*2*1 = 0.5 + 1.0 = 1.5
        # result = 2 + 1.5^2 + 1.5^4 = 2 + 2.25 + 5.0625 = 9.3125
        result_11 = func2.evaluate(np.ones(2))
        assert abs(result_11 - 9.3125) < 1e-10, f"f(1,1) expected 9.3125, got {result_11}"

        # f(2,0) for D=2:
        # sum_sq = 4, weighted = 0.5*1*2 = 1.0
        # result = 4 + 1 + 1 = 6.0
        result_20 = func2.evaluate(np.array([2.0, 0.0]))
        assert abs(result_20 - 6.0) < 1e-10, f"f(2,0) expected 6.0, got {result_20}"

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluations."""
        func = ZakharovFunction(dimension=3)
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.5, -0.5, 0.2],
            ]
        )
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test that batch evaluation returns correct shape."""
        func = ZakharovFunction(dimension=4)
        X = np.random.uniform(-5, 10, size=(7, 4))
        results = func.evaluate_batch(X)
        assert results.shape == (7,), f"Expected shape (7,), got {results.shape}"

    def test_dimension_validation(self):
        """Test input validation rejects wrong-shaped inputs."""
        func = ZakharovFunction(dimension=3)

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0]))  # Too few

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0, 4.0]))  # Too many

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0]]))  # Wrong columns

    def test_non_negativity(self):
        """Test that Zakharov function is always non-negative."""
        func = ZakharovFunction(dimension=5)
        rng = np.random.default_rng(42)
        for _ in range(50):
            x = rng.uniform(-5, 10, size=5)
            result = func.evaluate(x)
            assert result >= -1e-10, f"Expected non-negative, got {result} at {x}"

    def test_scalability(self):
        """Test function works across multiple dimensions with correct minimum."""
        for dim in [2, 10, 50]:
            func = ZakharovFunction(dimension=dim)
            x_opt = np.zeros(dim)
            result = func.evaluate(x_opt)
            assert abs(result) < 1e-10

    def test_unimodality_gradient_towards_origin(self):
        """Test that function value decreases as we move toward the origin."""
        func = ZakharovFunction(dimension=3)
        x_far = np.array([3.0, 3.0, 3.0])
        x_mid = np.array([1.5, 1.5, 1.5])
        x_near = np.array([0.5, 0.5, 0.5])
        x_opt = np.zeros(3)

        f_far = func.evaluate(x_far)
        f_mid = func.evaluate(x_mid)
        f_near = func.evaluate(x_near)
        f_opt = func.evaluate(x_opt)

        assert f_far > f_mid > f_near > f_opt


class TestZakharovRegistry:
    """Test Zakharov registry integration."""

    def test_registry_names(self):
        """Test that registered names resolve to ZakharovFunction."""
        from pyMOFL.registry import get

        func1 = get("Zakharov")(dimension=3)
        func2 = get("zakharov")(dimension=3)
        assert isinstance(func1, ZakharovFunction)
        assert isinstance(func2, ZakharovFunction)

        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) == func2.evaluate(x)
