"""
Tests for Sum of Different Powers function.

Following TDD approach with comprehensive test coverage.
Tests validate mathematical correctness, bounds handling, and edge cases.

f(x) = sum_{i=1}^{D} |x_i|^(i+1)   (1-based i, exponents 2..D+1)
Domain: [-1, 1]^D
Global minimum: f(0, ..., 0) = 0
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.sum_different_powers import SumDifferentPowersFunction


class TestSumDifferentPowersFunction:
    """Tests for Sum of Different Powers benchmark function."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = SumDifferentPowersFunction(dimension=5)
        assert func.dimension == 5
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(5, -1.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(5, 1.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(5, -1.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(5, 1.0))

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [2, 5, 10, 30]:
            func = SumDifferentPowersFunction(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim
            assert len(func.initialization_bounds.high) == dim

    def test_global_minimum(self):
        """Test function evaluates to 0 at the global minimum (origin)."""
        for dim in [2, 5, 10]:
            func = SumDifferentPowersFunction(dimension=dim)
            x_opt = np.zeros(dim)
            result = func.evaluate(x_opt)
            assert abs(result) < 1e-10, f"Expected 0 at origin for dim={dim}, got {result}"

    def test_get_global_minimum(self):
        """Test static get_global_minimum method returns correct point and value."""
        for dim in [2, 5, 10]:
            min_point, min_value = SumDifferentPowersFunction.get_global_minimum(dim)
            np.testing.assert_array_equal(min_point, np.zeros(dim))
            assert min_value == 0.0

    def test_get_global_minimum_consistency(self):
        """Test that evaluating at the returned global minimum gives the returned value."""
        for dim in [2, 5, 10]:
            func = SumDifferentPowersFunction(dimension=dim)
            min_point, min_value = SumDifferentPowersFunction.get_global_minimum(dim)
            evaluated = func.evaluate(min_point)
            assert abs(evaluated - min_value) < 1e-10

    def test_known_values(self):
        """Test function evaluation at hand-computed known points."""
        func2 = SumDifferentPowersFunction(dimension=2)

        # f(1,1) for D=2: |1|^2 + |1|^3 = 1 + 1 = 2
        result_ones = func2.evaluate(np.ones(2))
        assert abs(result_ones - 2.0) < 1e-10, f"f(1,1) expected 2.0, got {result_ones}"

        # f(0.5, 0.5) for D=2: |0.5|^2 + |0.5|^3 = 0.25 + 0.125 = 0.375
        result_half = func2.evaluate(np.full(2, 0.5))
        assert abs(result_half - 0.375) < 1e-10, f"f(0.5,0.5) expected 0.375, got {result_half}"

        # f(-0.5, -0.5) for D=2: same as f(0.5,0.5) due to absolute value = 0.375
        result_neg_half = func2.evaluate(np.full(2, -0.5))
        assert abs(result_neg_half - 0.375) < 1e-10

        # f(1,1,1) for D=3: |1|^2 + |1|^3 + |1|^4 = 1 + 1 + 1 = 3
        func3 = SumDifferentPowersFunction(dimension=3)
        result_ones_3 = func3.evaluate(np.ones(3))
        assert abs(result_ones_3 - 3.0) < 1e-10, f"f(1,1,1) expected 3.0, got {result_ones_3}"

        # f(1,0) for D=2: |1|^2 + |0|^3 = 1 + 0 = 1
        result_10 = func2.evaluate(np.array([1.0, 0.0]))
        assert abs(result_10 - 1.0) < 1e-10

        # f(0,1) for D=2: |0|^2 + |1|^3 = 0 + 1 = 1
        result_01 = func2.evaluate(np.array([0.0, 1.0]))
        assert abs(result_01 - 1.0) < 1e-10

        # f(0.5, 0.5, 0.5) for D=3:
        # |0.5|^2 + |0.5|^3 + |0.5|^4 = 0.25 + 0.125 + 0.0625 = 0.4375
        result_half_3 = func3.evaluate(np.full(3, 0.5))
        assert abs(result_half_3 - 0.4375) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluations."""
        func = SumDifferentPowersFunction(dimension=3)
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [0.1, -0.2, 0.3],
            ]
        )
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test that batch evaluation returns correct shape."""
        func = SumDifferentPowersFunction(dimension=4)
        X = np.random.uniform(-1, 1, size=(7, 4))
        results = func.evaluate_batch(X)
        assert results.shape == (7,), f"Expected shape (7,), got {results.shape}"

    def test_dimension_validation(self):
        """Test input validation rejects wrong-shaped inputs."""
        func = SumDifferentPowersFunction(dimension=3)

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0]))  # Too few

        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0, 4.0]))  # Too many

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0]]))  # Wrong columns

    def test_separability(self):
        """Test that Sum of Different Powers is separable.

        Each term depends on only one variable, so f(x) = sum of individual contributions.
        """
        func = SumDifferentPowersFunction(dimension=3)

        x = np.array([0.3, 0.5, 0.7])

        # Compute expected as sum of individual contributions
        # |0.3|^2 + |0.5|^3 + |0.7|^4
        expected = 0.3**2 + 0.5**3 + 0.7**4
        result = func.evaluate(x)
        assert abs(result - expected) < 1e-10

        # Verify separability: changing one variable only affects its own term
        x1 = np.array([0.3, 0.5, 0.0])
        x2 = np.array([0.0, 0.5, 0.0])
        f1 = func.evaluate(x1)
        f2 = func.evaluate(x2)
        # Difference should be exactly |0.3|^2 - 0 = 0.09
        assert abs((f1 - f2) - 0.09) < 1e-10

    def test_non_negativity(self):
        """Test that function is always non-negative (sum of absolute values to positive powers)."""
        func = SumDifferentPowersFunction(dimension=5)
        rng = np.random.default_rng(42)
        for _ in range(50):
            x = rng.uniform(-1, 1, size=5)
            result = func.evaluate(x)
            assert result >= -1e-10, f"Expected non-negative, got {result} at {x}"

    def test_symmetry(self):
        """Test that function is symmetric with respect to sign of each variable."""
        func = SumDifferentPowersFunction(dimension=4)
        rng = np.random.default_rng(123)
        for _ in range(20):
            x = rng.uniform(-1, 1, size=4)
            f_pos = func.evaluate(np.abs(x))
            f_neg = func.evaluate(-np.abs(x))
            f_orig = func.evaluate(x)
            # All should be equal because we use |x_i|
            assert abs(f_pos - f_neg) < 1e-10
            assert abs(f_pos - f_orig) < 1e-10

    def test_scalability(self):
        """Test function works across multiple dimensions."""
        for dim in [2, 10, 50]:
            func = SumDifferentPowersFunction(dimension=dim)
            x_opt = np.zeros(dim)
            result = func.evaluate(x_opt)
            assert abs(result) < 1e-10

    def test_increasing_exponents(self):
        """Test that higher-indexed variables have higher exponents.

        For |x_i| < 1, higher exponents produce smaller values.
        For |x_i| = 0.5: 0.5^2=0.25, 0.5^3=0.125, 0.5^4=0.0625, etc.
        """
        func = SumDifferentPowersFunction(dimension=5)

        # Evaluate with only one component nonzero at a time
        contributions = []
        for i in range(5):
            x = np.zeros(5)
            x[i] = 0.5
            contributions.append(func.evaluate(x))

        # Each contribution should be 0.5^(i+2) for i=0..4 (0-based)
        for i, c in enumerate(contributions):
            expected = 0.5 ** (i + 2)
            assert abs(c - expected) < 1e-10

        # Contributions should be strictly decreasing for |x|<1
        for i in range(len(contributions) - 1):
            assert contributions[i] > contributions[i + 1]


class TestSumDifferentPowersRegistry:
    """Test Sum of Different Powers registry integration."""

    def test_registry_names(self):
        """Test that registered names resolve to SumDifferentPowersFunction."""
        from pyMOFL.registry import get

        func1 = get("SumDifferentPowers")(dimension=3)
        func2 = get("sum_different_powers")(dimension=3)
        assert isinstance(func1, SumDifferentPowersFunction)
        assert isinstance(func2, SumDifferentPowersFunction)

        x = np.array([0.5, 0.5, 0.5])
        assert func1.evaluate(x) == func2.evaluate(x)
