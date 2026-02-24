"""
Tests for BentCigarFunction.

Validates mathematical correctness, bounds handling, batch evaluation,
and structural properties of the Bent Cigar benchmark function.

f(x) = x_1^2 + 10^6 * sum(x_i^2 for i=2..D)
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.bent_cigar import BentCigarFunction


class TestBentCigarFunction:
    """Tests for BentCigarFunction."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = BentCigarFunction(dimension=10)
        assert func.dimension == 10
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(10, -100.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(10, 100.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(10, -100.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(10, 100.0))

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [2, 5, 10, 30]:
            func = BentCigarFunction(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim
            assert len(func.initialization_bounds.high) == dim

    def test_global_minimum(self):
        """Test that function evaluates to 0 at the origin."""
        for dim in [2, 5, 10]:
            func = BentCigarFunction(dimension=dim)
            x_opt = np.zeros(dim)
            result = func.evaluate(x_opt)
            assert result == 0.0, f"Expected 0.0 at origin for dim={dim}, got {result}"

    def test_get_global_minimum(self):
        """Test get_global_minimum static method returns correct values."""
        for dim in [2, 5, 10, 30]:
            min_point, min_value = BentCigarFunction.get_global_minimum(dim)
            np.testing.assert_array_equal(min_point, np.zeros(dim))
            assert min_value == 0.0

    def test_get_global_minimum_consistency(self):
        """Test that evaluating at the returned global minimum gives the stated value."""
        for dim in [2, 5, 10]:
            func = BentCigarFunction(dimension=dim)
            min_point, min_value = BentCigarFunction.get_global_minimum(dim)
            result = func.evaluate(min_point)
            assert abs(result - min_value) < 1e-12

    def test_known_values_first_component_only(self):
        """Test f(1,0,...,0) = 1 (only first component contributes)."""
        for dim in [2, 5, 10]:
            func = BentCigarFunction(dimension=dim)
            x = np.zeros(dim)
            x[0] = 1.0
            result = func.evaluate(x)
            assert abs(result - 1.0) < 1e-12, (
                f"f(1,0,...,0) should be 1.0 for dim={dim}, got {result}"
            )

    def test_known_values_second_component_only(self):
        """Test f(0,1,0,...,0) = 10^6 (second component gets 10^6 scaling)."""
        for dim in [2, 5, 10]:
            func = BentCigarFunction(dimension=dim)
            x = np.zeros(dim)
            x[1] = 1.0
            result = func.evaluate(x)
            assert abs(result - 1e6) < 1e-6, (
                f"f(0,1,0,...,0) should be 1e6 for dim={dim}, got {result}"
            )

    def test_known_values_all_ones_dim3(self):
        """Test f(1,1,...,1) for D=3: 1 + 10^6*(1+1) = 1 + 2*10^6."""
        func = BentCigarFunction(dimension=3)
        x = np.ones(3)
        result = func.evaluate(x)
        expected = 1.0 + 2.0 * 1e6
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    def test_known_values_all_ones_general(self):
        """Test f(1,...,1) = 1 + 10^6*(D-1) for general dimensions."""
        for dim in [2, 5, 10]:
            func = BentCigarFunction(dimension=dim)
            x = np.ones(dim)
            result = func.evaluate(x)
            expected = 1.0 + 1e6 * (dim - 1)
            assert abs(result - expected) < 1e-4, (
                f"f(1,...,1) for dim={dim}: expected {expected}, got {result}"
            )

    def test_evaluate_batch(self):
        """Test that batch evaluation matches individual evaluation."""
        func = BentCigarFunction(dimension=5)
        rng = np.random.default_rng(42)
        X = rng.uniform(-100, 100, size=(10, 5))
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test that batch output has correct shape."""
        func = BentCigarFunction(dimension=5)
        X = np.zeros((7, 5))
        results = func.evaluate_batch(X)
        assert results.shape == (7,), f"Expected shape (7,), got {results.shape}"

    def test_dimension_validation(self):
        """Test that wrong input shape raises an error."""
        func = BentCigarFunction(dimension=5)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))  # Wrong dimension

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension

    def test_non_separability(self):
        """Test non-separability: changing x1 vs x2 have very different effects."""
        func = BentCigarFunction(dimension=5)
        base = np.zeros(5)

        # Perturb first component by 1
        x1 = base.copy()
        x1[0] = 1.0
        f1 = func.evaluate(x1)

        # Perturb second component by 1
        x2 = base.copy()
        x2[1] = 1.0
        f2 = func.evaluate(x2)

        # f2 should be 10^6 times larger than f1
        assert f2 / f1 == pytest.approx(1e6), f"Ratio should be 1e6, got {f2 / f1}"

    def test_non_negativity(self):
        """Test that function values are always non-negative."""
        func = BentCigarFunction(dimension=5)
        rng = np.random.default_rng(123)
        for _ in range(50):
            x = rng.uniform(-100, 100, size=5)
            result = func.evaluate(x)
            assert result >= 0.0, f"Function should be non-negative, got {result}"

    def test_symmetry_in_tail_components(self):
        """Test that function is symmetric across tail components (x2..xD)."""
        func = BentCigarFunction(dimension=4)
        # f(0, a, b, c) should equal f(0, c, b, a) for any permutation of tail
        x1 = np.array([0.0, 1.0, 2.0, 3.0])
        x2 = np.array([0.0, 3.0, 1.0, 2.0])
        assert func.evaluate(x1) == pytest.approx(func.evaluate(x2))

    def test_registry_names(self):
        """Test that all registry names work."""
        from pyMOFL.registry import get

        func1 = get("bent_cigar")(dimension=3)
        func2 = get("BentCigar")(dimension=3)
        assert isinstance(func1, BentCigarFunction)
        assert isinstance(func2, BentCigarFunction)

        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) == func2.evaluate(x)
