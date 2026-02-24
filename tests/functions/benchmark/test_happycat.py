"""
Tests for HappyCat function.

Following TDD approach with comprehensive test coverage.
Tests validate mathematical correctness, bounds handling, and edge cases.

f(x) = ||x|^2 - D|^(1/4) + (0.5|x|^2 + sum(x_i))/D + 0.5
Domain: [-100, 100]^D
Global minimum: f(-1, ..., -1) = 0
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.happycat import HappyCatFunction


class TestHappyCatFunction:
    """Tests for HappyCat benchmark function."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = HappyCatFunction(dimension=5)
        assert func.dimension == 5
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(5, -100.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(5, 100.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(5, -100.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(5, 100.0))

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [2, 5, 10, 30]:
            func = HappyCatFunction(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim
            assert len(func.initialization_bounds.high) == dim

    def test_global_minimum(self):
        """Test function evaluates to 0 at the global minimum (-1, ..., -1)."""
        for dim in [2, 5, 10]:
            func = HappyCatFunction(dimension=dim)
            x_opt = np.full(dim, -1.0)
            result = func.evaluate(x_opt)
            # At x=(-1,...,-1): r2=D, |D-D|^0.25=0, sum_x=-D
            # (0.5*D + (-D))/D = -0.5, result = 0 + (-0.5) + 0.5 = 0
            assert abs(result) < 1e-10, f"Expected 0 at optimum for dim={dim}, got {result}"

    def test_get_global_minimum(self):
        """Test static get_global_minimum method returns correct point and value."""
        for dim in [2, 5, 10]:
            min_point, min_value = HappyCatFunction.get_global_minimum(dim)
            np.testing.assert_array_equal(min_point, np.full(dim, -1.0))
            assert min_value == 0.0

    def test_get_global_minimum_consistency(self):
        """Test that evaluating at the returned global minimum gives the returned value."""
        for dim in [2, 5, 10]:
            func = HappyCatFunction(dimension=dim)
            min_point, min_value = HappyCatFunction.get_global_minimum(dim)
            evaluated = func.evaluate(min_point)
            assert abs(evaluated - min_value) < 1e-10

    def test_known_values(self):
        """Test function evaluation at hand-computed known points."""
        # f(0,...,0) for D=2:
        # r2=0, sum_x=0, |0-2|^0.25 = 2^0.25, (0+0)/2=0
        # result = 2^0.25 + 0 + 0.5 = 2^0.25 + 0.5
        func2 = HappyCatFunction(dimension=2)
        result_origin = func2.evaluate(np.zeros(2))
        expected_origin = 2.0**0.25 + 0.5
        assert abs(result_origin - expected_origin) < 1e-10, (
            f"f(0,0) expected {expected_origin}, got {result_origin}"
        )

        # f(1,1) for D=2:
        # r2=2, sum_x=2, |2-2|^0.25=0, (0.5*2+2)/2=1.5
        # result = 0 + 1.5 + 0.5 = 2.0
        result_ones = func2.evaluate(np.ones(2))
        assert abs(result_ones - 2.0) < 1e-10, f"f(1,1) expected 2.0, got {result_ones}"

        # f(0,...,0) for D=5:
        # |0-5|^0.25 = 5^0.25, (0+0)/5=0, result = 5^0.25 + 0.5
        func5 = HappyCatFunction(dimension=5)
        result_origin_5 = func5.evaluate(np.zeros(5))
        expected_origin_5 = 5.0**0.25 + 0.5
        assert abs(result_origin_5 - expected_origin_5) < 1e-10

        # f(2, 0) for D=2:
        # r2=4, sum_x=2, |4-2|^0.25 = 2^0.25, (0.5*4+2)/2 = 2.0
        # result = 2^0.25 + 2.0 + 0.5
        result_2_0 = func2.evaluate(np.array([2.0, 0.0]))
        expected_2_0 = 2.0**0.25 + 2.0 + 0.5
        assert abs(result_2_0 - expected_2_0) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluations."""
        func = HappyCatFunction(dimension=3)
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
        func = HappyCatFunction(dimension=4)
        X = np.random.uniform(-100, 100, size=(7, 4))
        results = func.evaluate_batch(X)
        assert results.shape == (7,), f"Expected shape (7,), got {results.shape}"

    def test_dimension_validation(self):
        """Test input validation rejects wrong-shaped inputs."""
        func = HappyCatFunction(dimension=3)

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0]))  # Too few

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0, 4.0]))  # Too many

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0]]))  # Wrong columns

    def test_non_negativity_near_optimum(self):
        """Test function is non-negative in the vicinity of the optimum."""
        func = HappyCatFunction(dimension=5)
        rng = np.random.default_rng(42)
        for _ in range(50):
            # Small perturbation around optimum (-1,...,-1)
            x = np.full(5, -1.0) + rng.uniform(-0.1, 0.1, size=5)
            result = func.evaluate(x)
            assert result >= -1e-10, f"Expected non-negative near optimum, got {result} at {x}"

    def test_non_separability(self):
        """Test that HappyCat is non-separable (changing one variable affects contribution of others)."""
        func = HappyCatFunction(dimension=3)
        x_base = np.array([0.5, 0.5, 0.5])
        f_base = func.evaluate(x_base)

        # Changing x[0] while keeping others fixed
        x_mod = x_base.copy()
        x_mod[0] = 1.0
        f_mod = func.evaluate(x_mod)

        # The change in f should depend on the values of all variables due to r2 and sum_x coupling
        assert f_mod != f_base

    def test_scalability(self):
        """Test function works across multiple dimensions."""
        for dim in [2, 10, 50]:
            func = HappyCatFunction(dimension=dim)
            x_opt = np.full(dim, -1.0)
            result = func.evaluate(x_opt)
            assert abs(result) < 1e-10


class TestHappyCatRegistry:
    """Test HappyCat registry integration."""

    def test_registry_names(self):
        """Test that registered names resolve to HappyCatFunction."""
        from pyMOFL.registry import get

        func1 = get("HappyCat")(dimension=3)
        func2 = get("happycat")(dimension=3)
        assert isinstance(func1, HappyCatFunction)
        assert isinstance(func2, HappyCatFunction)

        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) == func2.evaluate(x)
