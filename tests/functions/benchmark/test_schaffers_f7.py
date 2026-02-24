"""
Tests for SchaffersF7Function.

Validates mathematical correctness, bounds handling, batch evaluation,
and structural properties of the Schaffers F7 benchmark function.

s_i = sqrt(x_i^2 + x_{i+1}^2), i = 1..D-1
f(x) = (1/(D-1) * sum(sqrt(s_i) * (1 + sin^2(50 * s_i^0.2))))^2
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.schaffer import SchaffersF7Function


class TestSchaffersF7Function:
    """Tests for SchaffersF7Function."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = SchaffersF7Function(dimension=10)
        assert func.dimension == 10
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(10, -100.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(10, 100.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(10, -100.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(10, 100.0))

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions >= 2."""
        for dim in [2, 5, 10, 30]:
            func = SchaffersF7Function(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim

    def test_dimension_1_raises_error(self):
        """Test that dimension=1 raises ValueError."""
        with pytest.raises(ValueError, match="Schaffers F7 requires dimension >= 2"):
            SchaffersF7Function(dimension=1)

    def test_global_minimum(self):
        """Test that function evaluates to 0 at the origin."""
        for dim in [2, 5, 10]:
            func = SchaffersF7Function(dimension=dim)
            x_opt = np.zeros(dim)
            result = func.evaluate(x_opt)
            assert abs(result) < 1e-12, f"Expected 0.0 at origin for dim={dim}, got {result}"

    def test_get_global_minimum(self):
        """Test get_global_minimum static method returns correct values."""
        for dim in [2, 5, 10, 30]:
            min_point, min_value = SchaffersF7Function.get_global_minimum(dim)
            np.testing.assert_array_equal(min_point, np.zeros(dim))
            assert min_value == 0.0

    def test_get_global_minimum_consistency(self):
        """Test that evaluating at the returned global minimum gives the stated value."""
        for dim in [2, 5, 10]:
            func = SchaffersF7Function(dimension=dim)
            min_point, min_value = SchaffersF7Function.get_global_minimum(dim)
            result = func.evaluate(min_point)
            assert abs(result - min_value) < 1e-12

    def test_known_value_at_origin_dim2(self):
        """Test f(0,0) = 0 for D=2.

        s_1 = sqrt(0+0) = 0, sqrt(0)*(1+sin^2(0)) = 0
        f = (0/1)^2 = 0
        """
        func = SchaffersF7Function(dimension=2)
        x = np.zeros(2)
        result = func.evaluate(x)
        assert abs(result) < 1e-15

    def test_known_value_hand_computed_dim2(self):
        """Test a hand-computed value for D=2 with x=[1, 0]."""
        func = SchaffersF7Function(dimension=2)
        x = np.array([1.0, 0.0])

        # s_1 = sqrt(1 + 0) = 1
        # term = sqrt(1) * (1 + sin^2(50 * 1^0.2))
        #      = 1 * (1 + sin^2(50))
        s = 1.0
        term = np.sqrt(s) * (1.0 + np.sin(50.0 * s**0.2) ** 2)
        expected = (term / 1.0) ** 2  # D-1 = 1
        result = func.evaluate(x)
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_evaluate_batch(self):
        """Test that batch evaluation matches individual evaluation."""
        func = SchaffersF7Function(dimension=5)
        rng = np.random.default_rng(42)
        X = rng.uniform(-100, 100, size=(10, 5))
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test that batch output has correct shape."""
        func = SchaffersF7Function(dimension=5)
        X = np.zeros((7, 5))
        results = func.evaluate_batch(X)
        assert results.shape == (7,), f"Expected shape (7,), got {results.shape}"

    def test_dimension_validation(self):
        """Test that wrong input shape raises an error."""
        func = SchaffersF7Function(dimension=5)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))  # Wrong dimension

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension

    def test_non_negativity(self):
        """Test that function values are always non-negative (it's a square)."""
        func = SchaffersF7Function(dimension=5)
        rng = np.random.default_rng(123)
        for _ in range(50):
            x = rng.uniform(-100, 100, size=5)
            result = func.evaluate(x)
            assert result >= 0.0, f"Function should be non-negative, got {result}"

    def test_multimodality(self):
        """Test that different non-zero inputs can yield different values.

        The sin^2 term creates many local optima, so nearby points can have
        noticeably different function values.
        """
        func = SchaffersF7Function(dimension=5)
        rng = np.random.default_rng(99)
        values = set()
        for _ in range(20):
            x = rng.uniform(-10, 10, size=5)
            result = func.evaluate(x)
            values.add(round(result, 6))
        # Should have multiple distinct values
        assert len(values) > 5, (
            f"Expected many distinct values for multimodal function, got {len(values)}"
        )

    def test_adjacent_pair_coupling(self):
        """Test that the function couples adjacent pairs of variables.

        s_i depends on x_i and x_{i+1}, so changing x_1 affects s_1 only,
        but changing x_2 affects both s_1 and s_2.
        """
        func = SchaffersF7Function(dimension=4)
        base = np.zeros(4)

        # Perturb x_1 only: affects s_1
        x1 = base.copy()
        x1[0] = 1.0
        f1 = func.evaluate(x1)

        # Perturb x_2 only: affects s_1 and s_2
        x2 = base.copy()
        x2[1] = 1.0
        f2 = func.evaluate(x2)

        # Both should be non-zero and potentially different
        assert f1 > 0.0
        assert f2 > 0.0

    def test_registry_names(self):
        """Test that all registry names work."""
        from pyMOFL.registry import get

        func1 = get("schaffers_f7")(dimension=3)
        func2 = get("SchaffersF7")(dimension=3)
        assert isinstance(func1, SchaffersF7Function)
        assert isinstance(func2, SchaffersF7Function)

        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) == func2.evaluate(x)
