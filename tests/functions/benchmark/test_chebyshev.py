"""
Tests for Storn's Chebyshev Polynomial Fitting benchmark function (CEC 2019 F1).

Following TDD approach with comprehensive test coverage.
Tests validate mathematical correctness, bounds handling, batch evaluation, and edge cases.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.chebyshev import ChebyshevFunction
from tests.utils.benchmark_validation import BenchmarkValidator


class TestChebyshevFunction:
    """Tests for Chebyshev benchmark function."""

    def test_initialization_default(self):
        """Test default initialization (dimension=9, bounds=[-8192, 8192])."""
        func = ChebyshevFunction()
        assert func.dimension == 9
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(9, -8192.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(9, 8192.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(9, -8192.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(9, 8192.0))

    def test_initialization_invalid_dimension(self):
        """Test that dimension < 3 raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension >= 3"):
            ChebyshevFunction(dimension=2)

    def test_benchmark_contract(self):
        """Test compliance with BenchmarkValidator contract."""
        func = ChebyshevFunction(dimension=9)
        BenchmarkValidator.assert_contract(func, check_global_minimum=True, global_min_atol=1e-8)

    def test_global_minimum_9d(self):
        """Test evaluation at known 9D Chebyshev global optimum gives 0.0."""
        func = ChebyshevFunction(dimension=9)
        x_opt, f_opt = func.get_global_minimum()
        assert f_opt == 0.0
        val = func.evaluate(x_opt)
        assert abs(val) < 1e-10

    def test_global_minimum_17d(self):
        """Test evaluation at known 17D Chebyshev global optimum gives 0.0."""
        func = ChebyshevFunction(dimension=17)
        x_opt, f_opt = func.get_global_minimum()
        assert f_opt == 0.0
        val = func.evaluate(x_opt)
        assert abs(val) < 1e-6

    def test_evaluate_non_optimum(self):
        """Test that non-optimum point produces positive objective value."""
        func = ChebyshevFunction(dimension=9)
        val_ones = func.evaluate(np.ones(9))
        assert val_ones > 0.0

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluations."""
        func = ChebyshevFunction(dimension=9)
        rng = np.random.default_rng(42)
        X = rng.uniform(-100.0, 100.0, size=(8, 9))
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch_results, individual_results, rtol=1e-12, atol=1e-12)

    def test_dimension_validation(self):
        """Test input dimension mismatch raises ValueError."""
        func = ChebyshevFunction(dimension=9)
        with pytest.raises(ValueError):
            func.evaluate(np.zeros(8))
        with pytest.raises(ValueError):
            func.evaluate(np.zeros(10))
        with pytest.raises(ValueError):
            func.evaluate_batch(np.zeros((5, 8)))


class TestChebyshevRegistry:
    """Test Chebyshev registry integration."""

    def test_registry_names(self):
        """Test that registered aliases resolve to ChebyshevFunction."""
        from pyMOFL.registry import get

        func1 = get("Chebyshev")(dimension=9)
        func2 = get("chebyshev")(dimension=9)
        assert isinstance(func1, ChebyshevFunction)
        assert isinstance(func2, ChebyshevFunction)
        x = np.ones(9)
        assert func1.evaluate(x) == func2.evaluate(x)
