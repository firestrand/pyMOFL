"""
Tests for HGBat function.

Following TDD approach with comprehensive test coverage.
Tests validate mathematical correctness, bounds handling, and edge cases.

f(x) = |(|x|^2)^2 - (sum(x_i))^2|^(1/2) + (0.5|x|^2 + sum(x_i))/D + 0.5
Domain: [-100, 100]^D
Global minimum: f(-1, ..., -1) = 0
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.happycat import HGBatFunction


class TestHGBatFunction:
    """Tests for HGBat benchmark function."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = HGBatFunction(dimension=5)
        assert func.dimension == 5
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(5, -100.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(5, 100.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(5, -100.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(5, 100.0))

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [2, 5, 10, 30]:
            func = HGBatFunction(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim
            assert len(func.initialization_bounds.high) == dim

    def test_global_minimum(self):
        """Test function evaluates to 0 at the global minimum (-1, ..., -1)."""
        for dim in [2, 5, 10]:
            func = HGBatFunction(dimension=dim)
            x_opt = np.full(dim, -1.0)
            result = func.evaluate(x_opt)
            # At x=(-1,...,-1): r2=D, sum_x=-D
            # |D^2 - D^2|^0.5 = 0, (0.5*D + (-D))/D = -0.5
            # result = 0 + (-0.5) + 0.5 = 0
            assert abs(result) < 1e-10, f"Expected 0 at optimum for dim={dim}, got {result}"

    def test_get_global_minimum(self):
        """Test static get_global_minimum method returns correct point and value."""
        for dim in [2, 5, 10]:
            min_point, min_value = HGBatFunction.get_global_minimum(dim)
            np.testing.assert_array_equal(min_point, np.full(dim, -1.0))
            assert min_value == 0.0

    def test_get_global_minimum_consistency(self):
        """Test that evaluating at the returned global minimum gives the returned value."""
        for dim in [2, 5, 10]:
            func = HGBatFunction(dimension=dim)
            min_point, min_value = HGBatFunction.get_global_minimum(dim)
            evaluated = func.evaluate(min_point)
            assert abs(evaluated - min_value) < 1e-10

    def test_known_values(self):
        """Test function evaluation at hand-computed known points."""
        func2 = HGBatFunction(dimension=2)

        # f(0,0) for D=2:
        # r2=0, sum_x=0, |0-0|^0.5=0, (0+0)/2=0
        # result = 0 + 0 + 0.5 = 0.5
        result_origin = func2.evaluate(np.zeros(2))
        assert abs(result_origin - 0.5) < 1e-10, f"f(0,0) expected 0.5, got {result_origin}"

        # f(1,1) for D=2:
        # r2=2, sum_x=2, |4-4|^0.5=0, (0.5*2+2)/2=1.5
        # result = 0 + 1.5 + 0.5 = 2.0
        result_ones = func2.evaluate(np.ones(2))
        assert abs(result_ones - 2.0) < 1e-10, f"f(1,1) expected 2.0, got {result_ones}"

        # f(1,0) for D=2:
        # r2=1, sum_x=1, |1-1|^0.5=0, (0.5+1)/2=0.75
        # result = 0 + 0.75 + 0.5 = 1.25
        result_10 = func2.evaluate(np.array([1.0, 0.0]))
        assert abs(result_10 - 1.25) < 1e-10, f"f(1,0) expected 1.25, got {result_10}"

        # f(1,-1) for D=2:
        # r2=2, sum_x=0, |4-0|^0.5=2, (0.5*2+0)/2=0.5
        # result = 2 + 0.5 + 0.5 = 3.0
        result_1m1 = func2.evaluate(np.array([1.0, -1.0]))
        assert abs(result_1m1 - 3.0) < 1e-10, f"f(1,-1) expected 3.0, got {result_1m1}"

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluations."""
        func = HGBatFunction(dimension=3)
        X = np.array(
            [
                [-1.0, -1.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.5, -0.5, 0.2],
            ]
        )
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test that batch evaluation returns correct shape."""
        func = HGBatFunction(dimension=4)
        X = np.random.uniform(-100, 100, size=(7, 4))
        results = func.evaluate_batch(X)
        assert results.shape == (7,), f"Expected shape (7,), got {results.shape}"

    def test_dimension_validation(self):
        """Test input validation rejects wrong-shaped inputs."""
        func = HGBatFunction(dimension=3)

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0]))  # Too few

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0, 4.0]))  # Too many

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0]]))  # Wrong columns

    def test_non_separability(self):
        """Test that HGBat is non-separable due to (sum(x_i))^2 coupling."""
        func = HGBatFunction(dimension=3)
        # f(1,0,0) vs f(0,1,0) should differ because r2 and sum_x interact differently
        # Use vectors with different sum_x but same r2 to show non-separability
        # Let's use vectors with different structures
        x3 = np.array([1.0, 1.0, 0.0])
        x4 = np.array([np.sqrt(2.0), 0.0, 0.0])
        # Same r2=2, but sum_x=2 vs sum_x=sqrt(2)
        f3 = func.evaluate(x3)
        f4 = func.evaluate(x4)
        assert f3 != f4, "HGBat should be non-separable"

    def test_scalability(self):
        """Test function works across multiple dimensions."""
        for dim in [2, 10, 50]:
            func = HGBatFunction(dimension=dim)
            x_opt = np.full(dim, -1.0)
            result = func.evaluate(x_opt)
            assert abs(result) < 1e-10


class TestHGBatRegistry:
    """Test HGBat registry integration."""

    def test_registry_names(self):
        """Test that registered names resolve to HGBatFunction."""
        from pyMOFL.registry import get

        func1 = get("HGBat")(dimension=3)
        func2 = get("hgbat")(dimension=3)
        assert isinstance(func1, HGBatFunction)
        assert isinstance(func2, HGBatFunction)

        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) == func2.evaluate(x)
