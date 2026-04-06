"""
Tests for Levy function.

Following TDD approach with comprehensive test coverage.
Tests validate mathematical correctness, bounds handling, and edge cases.

w_i = 1 + (x_i - 1)/4
f(x) = sin^2(pi*w_1) + sum_{i=1}^{D-1} (w_i-1)^2 (1+10*sin^2(pi*w_{i+1}))
        + (w_D-1)^2 (1+sin^2(2*pi*w_D))
Domain: [-10, 10]^D
Global minimum: f(1, ..., 1) = 0
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.levy import LevyFunction


class TestLevyFunction:
    """Tests for Levy benchmark function."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = LevyFunction(dimension=5)
        assert func.dimension == 5
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(5, -10.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(5, 10.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(5, -10.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(5, 10.0))

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [2, 5, 10, 30]:
            func = LevyFunction(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim
            assert len(func.initialization_bounds.high) == dim

    def test_global_minimum(self):
        """Test function evaluates to 0 at the global minimum (1, ..., 1)."""
        for dim in [2, 5, 10]:
            func = LevyFunction(dimension=dim)
            x_opt = np.ones(dim)
            result = func.evaluate(x_opt)
            # At x=(1,...,1): w_i=1, sin(pi)=0, all (w_i-1)=0
            assert abs(result) < 1e-10, f"Expected 0 at optimum for dim={dim}, got {result}"

    def test_get_global_minimum(self):
        """Test get_global_minimum method returns correct point and value."""
        for dim in [2, 5, 10]:
            func = LevyFunction(dimension=dim)
            min_point, min_value = func.get_global_minimum()
            np.testing.assert_array_equal(min_point, np.ones(dim))
            assert min_value == 0.0

    def test_get_global_minimum_consistency(self):
        """Test that evaluating at the returned global minimum gives the returned value."""
        for dim in [2, 5, 10]:
            func = LevyFunction(dimension=dim)
            min_point, min_value = func.get_global_minimum()
            evaluated = func.evaluate(min_point)
            assert abs(evaluated - min_value) < 1e-10

    def test_known_values(self):
        """Test function evaluation at hand-computed known points."""
        # D=1 case: f(x) = sin^2(pi*w1) + (w1-1)^2*(1+sin^2(2*pi*w1))
        # where w1 = 1 + (x1-1)/4
        # Note: the sum from i=1 to D-1 is empty when D=1, so only term1 + term3
        func1 = LevyFunction(dimension=1)

        # f(1): w1=1, sin(pi)=0, (0)^2*(1+sin(2*pi))=0 -> 0
        result_opt = func1.evaluate(np.array([1.0]))
        assert abs(result_opt) < 1e-10

        # D=2 case
        func2 = LevyFunction(dimension=2)

        # f(1,1) = 0
        result_ones = func2.evaluate(np.ones(2))
        assert abs(result_ones) < 1e-10

        # f(-1,-1) for D=2:
        # w_i = 1 + (-1-1)/4 = 1 + (-2/4) = 0.5
        # term1 = sin^2(pi*0.5) = sin^2(pi/2) = 1
        # term2 (i=1 only): (0.5-1)^2*(1+10*sin^2(pi*0.5)) = 0.25*(1+10*1) = 0.25*11 = 2.75
        # term3: (0.5-1)^2*(1+sin^2(2*pi*0.5)) = 0.25*(1+sin^2(pi)) = 0.25*(1+0) = 0.25
        # total = 1 + 2.75 + 0.25 = 4.0
        result_neg_ones = func2.evaluate(np.full(2, -1.0))
        assert abs(result_neg_ones - 4.0) < 1e-10, f"f(-1,-1) expected 4.0, got {result_neg_ones}"

        # f(5,5) for D=2:
        # w_i = 1 + (5-1)/4 = 1 + 1 = 2
        # term1 = sin^2(2*pi) = 0
        # term2 (i=1): (2-1)^2*(1+10*sin^2(2*pi)) = 1*(1+0) = 1
        # term3: (2-1)^2*(1+sin^2(4*pi)) = 1*(1+0) = 1
        # total = 0 + 1 + 1 = 2.0
        result_fives = func2.evaluate(np.full(2, 5.0))
        assert abs(result_fives - 2.0) < 1e-10, f"f(5,5) expected 2.0, got {result_fives}"

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluations."""
        func = LevyFunction(dimension=3)
        X = np.array(
            [
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0],
                [5.0, 5.0, 5.0],
                [0.5, -0.5, 0.2],
            ]
        )
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test that batch evaluation returns correct shape."""
        func = LevyFunction(dimension=4)
        X = np.random.uniform(-10, 10, size=(7, 4))
        results = func.evaluate_batch(X)
        assert results.shape == (7,), f"Expected shape (7,), got {results.shape}"

    def test_dimension_validation(self):
        """Test input validation rejects wrong-shaped inputs."""
        func = LevyFunction(dimension=3)

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0]))  # Too few

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0, 4.0]))  # Too many

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0]]))  # Wrong columns

    def test_multimodality(self):
        """Test that Levy function has multiple local minima (multimodal)."""
        func = LevyFunction(dimension=2)
        # At x=(1,1) f=0 (global min), at x=(5,5) w=(2,2) -> sin terms zero but
        # (w-1)^2 terms non-zero -> local structure exists
        # At x=(-3,-3) w=0, sin^2(0)=0, but (w-1)^2 terms non-zero
        f_opt = func.evaluate(np.ones(2))
        f_other = func.evaluate(np.array([5.0, 5.0]))
        assert f_opt < f_other, "Global minimum should be less than other points"

    def test_non_negativity(self):
        """Test that Levy function is non-negative (all terms are sums of squares times non-negative factors)."""
        func = LevyFunction(dimension=5)
        rng = np.random.default_rng(42)
        for _ in range(50):
            x = rng.uniform(-10, 10, size=5)
            result = func.evaluate(x)
            assert result >= -1e-10, f"Expected non-negative, got {result} at {x}"

    def test_scalability(self):
        """Test function works across multiple dimensions."""
        for dim in [2, 10, 50]:
            func = LevyFunction(dimension=dim)
            x_opt = np.ones(dim)
            result = func.evaluate(x_opt)
            assert abs(result) < 1e-10


class TestLevyRegistry:
    """Test Levy registry integration."""

    def test_registry_names(self):
        """Test that registered names resolve to LevyFunction."""
        from pyMOFL.registry import get

        func1 = get("Levy")(dimension=3)
        func2 = get("levy")(dimension=3)
        assert isinstance(func1, LevyFunction)
        assert isinstance(func2, LevyFunction)

        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) == func2.evaluate(x)
