"""
Tests for FastFractal DoubleDip benchmark function (CEC 2008 F7).

Validates mathematical correctness, C++/Java reference parity, bounds, batch evaluation,
and registration contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyMOFL.functions.benchmark.fractal import FastFractalDoubleDip
from pyMOFL.registry import get
from tests.utils.benchmark_validation import BenchmarkValidator


class TestFastFractalDoubleDip:
    """Tests for FastFractalDoubleDip benchmark function."""

    def test_initialization_default(self):
        """Test default initialization (dimension=1000, bounds=[-1, 1])."""
        func = FastFractalDoubleDip()
        assert func.dimension == 1000
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(1000, -1.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(1000, 1.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(1000, -1.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(1000, 1.0))

    def test_initialization_invalid_dimension(self):
        """Test that dimension < 2 raises ValueError."""
        with pytest.raises(ValueError, match="requires dimension >= 2"):
            FastFractalDoubleDip(dimension=1)

    def test_benchmark_contract(self):
        """Test compliance with BenchmarkValidator contract (global minimum is unknown)."""
        func = FastFractalDoubleDip(dimension=5)
        BenchmarkValidator.assert_contract(func, check_global_minimum=False)

    def test_global_minimum_raises(self):
        """Test that get_global_minimum raises NotImplementedError as per specification."""
        func = FastFractalDoubleDip(dimension=5)
        with pytest.raises(NotImplementedError, match="analytically unknown"):
            func.get_global_minimum()

    def test_reference_parity_2d(self):
        """Test exact parity with official C++/Java CEC 2008 reference for 2D."""
        func = FastFractalDoubleDip(dimension=2)

        # Zero and ones wrap identically
        val_zero = func.evaluate(np.zeros(2))
        val_ones = func.evaluate(np.ones(2))
        assert np.isclose(val_zero, -2.069711033031289e01, atol=1e-12)
        assert np.isclose(val_ones, -2.069711033031289e01, atol=1e-12)

        val_half = func.evaluate(np.full(2, 0.5))
        assert np.isclose(val_half, -1.361670994684805e01, atol=1e-12)

        test1 = np.array([-0.5, 0.0])
        val_test1 = func.evaluate(test1)
        assert np.isclose(val_test1, -6.376363946796473e00, atol=1e-12)

    def test_reference_parity_5d(self):
        """Test exact parity with official C++/Java CEC 2008 reference for 5D."""
        func = FastFractalDoubleDip(dimension=5)

        val_zero = func.evaluate(np.zeros(5))
        assert np.isclose(val_zero, -3.217961534320012e01, atol=1e-12)

        val_half = func.evaluate(np.full(5, 0.5))
        assert np.isclose(val_half, -2.488862900212410e01, atol=1e-12)

        test1 = np.array([-0.5 + i * 0.2 for i in range(5)])
        val_test1 = func.evaluate(test1)
        assert np.isclose(val_test1, -3.182973958338624e01, atol=1e-12)

    def test_reference_parity_10d(self):
        """Test exact parity with official C++/Java CEC 2008 reference for 10D."""
        func = FastFractalDoubleDip(dimension=10)

        val_zero = func.evaluate(np.zeros(10))
        assert np.isclose(val_zero, -5.348740568959130e01, atol=1e-12)

        val_half = func.evaluate(np.full(10, 0.5))
        assert np.isclose(val_half, -6.680486421936910e01, atol=1e-12)

        test1 = np.array([-0.5 + i * 0.1 for i in range(10)])
        val_test1 = func.evaluate(test1)
        assert np.isclose(val_test1, -7.876278725952875e01, atol=1e-12)

    def test_reference_parity_1000d(self):
        """Test exact parity with official C++/Java CEC 2008 reference for 1000D."""
        func = FastFractalDoubleDip(dimension=1000)
        x = np.full(1000, 0.12323)
        val = func.evaluate(x)
        assert np.isclose(val, -5.766784527543412e03, atol=1e-10)

    def test_evaluate_batch_consistency(self):
        """Test batch evaluation matches individual evaluations."""
        func = FastFractalDoubleDip(dimension=4)
        rng = np.random.default_rng(42)
        X = rng.uniform(-1.0, 1.0, size=(6, 4))

        batch_results = func.evaluate_batch(X)
        single_results = np.array([func.evaluate(row) for row in X])

        np.testing.assert_allclose(batch_results, single_results, atol=1e-12)

    def test_registry_aliases(self):
        """Test that all registered aliases resolve to FastFractalDoubleDip."""
        assert get("FastFractalDoubleDip") is FastFractalDoubleDip
        assert get("fast_fractal_double_dip") is FastFractalDoubleDip
        assert get("DoubleDip") is FastFractalDoubleDip
        assert get("FastFractal") is FastFractalDoubleDip
