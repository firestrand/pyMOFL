"""
Tests for Inverse Hilbert Matrix benchmark function (CEC 2019 F2).

Following TDD approach with comprehensive test coverage.
Tests validate mathematical correctness, bounds handling, batch evaluation, and edge cases.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.hilbert import HilbertFunction
from tests.utils.benchmark_validation import BenchmarkValidator


class TestHilbertFunction:
    """Tests for Hilbert benchmark function."""

    def test_initialization_default(self):
        """Test default initialization (dimension=16, bounds=[-16384, 16384])."""
        func = HilbertFunction()
        assert func.dimension == 16
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(16, -16384.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(16, 16384.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(16, -16384.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(16, 16384.0))

    def test_initialization_invalid_dimension(self):
        """Test that non-square dimension raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension to be a square"):
            HilbertFunction(dimension=10)
        with pytest.raises(ValueError, match="requires dimension to be a square"):
            HilbertFunction(dimension=2)

    def test_benchmark_contract(self):
        """Test compliance with BenchmarkValidator contract."""
        func = HilbertFunction(dimension=16)
        BenchmarkValidator.assert_contract(func, check_global_minimum=True, global_min_atol=1e-8)

    def test_global_minimum_9d(self):
        """Test 3x3 (9D) inverse Hilbert optimum."""
        func = HilbertFunction(dimension=9)
        x_opt, f_opt = func.get_global_minimum()
        assert f_opt == 0.0
        val = func.evaluate(x_opt)
        assert abs(val) < 1e-10

    def test_global_minimum_16d(self):
        """Test 4x4 (16D) inverse Hilbert optimum."""
        func = HilbertFunction(dimension=16)
        x_opt, f_opt = func.get_global_minimum()
        assert f_opt == 0.0
        val = func.evaluate(x_opt)
        assert abs(val) < 1e-10

    def test_evaluate_non_optimum(self):
        """Test that zero matrix yields L1 norm equal to b (sum of identity diagonal)."""
        func = HilbertFunction(dimension=16)
        # H @ 0 = 0, ||0 - I_4||_1 = 4.0
        val_zeros = func.evaluate(np.zeros(16))
        assert abs(val_zeros - 4.0) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluations."""
        func = HilbertFunction(dimension=16)
        rng = np.random.default_rng(42)
        X = rng.uniform(-100.0, 100.0, size=(10, 16))
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch_results, individual_results, rtol=1e-12, atol=1e-12)

    def test_dimension_validation(self):
        """Test input dimension mismatch raises ValueError."""
        func = HilbertFunction(dimension=16)
        with pytest.raises(ValueError):
            func.evaluate(np.zeros(15))
        with pytest.raises(ValueError):
            func.evaluate(np.zeros(17))
        with pytest.raises(ValueError):
            func.evaluate_batch(np.zeros((5, 15)))


class TestHilbertRegistry:
    """Test Hilbert registry integration."""

    def test_registry_names(self):
        """Test that registered aliases resolve to HilbertFunction."""
        from pyMOFL.registry import get

        func1 = get("Hilbert")(dimension=16)
        func2 = get("hilbert")(dimension=16)
        assert isinstance(func1, HilbertFunction)
        assert isinstance(func2, HilbertFunction)
        x = np.ones(16)
        assert func1.evaluate(x) == func2.evaluate(x)
