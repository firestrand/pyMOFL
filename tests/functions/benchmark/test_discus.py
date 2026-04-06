"""
Tests for DiscusFunction.

Validates mathematical correctness, bounds handling, batch evaluation,
and structural properties of the Discus (Tablet) benchmark function.

f(x) = 10^6 * x_1^2 + sum(x_i^2 for i=2..D)
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.bent_cigar import DiscusFunction


class TestDiscusFunction:
    """Tests for DiscusFunction."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = DiscusFunction(dimension=10)
        assert func.dimension == 10
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(10, -100.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(10, 100.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(10, -100.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(10, 100.0))

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [2, 5, 10, 30]:
            func = DiscusFunction(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim
            assert len(func.initialization_bounds.high) == dim

    def test_global_minimum(self):
        """Test that function evaluates to 0 at the origin."""
        for dim in [2, 5, 10]:
            func = DiscusFunction(dimension=dim)
            x_opt = np.zeros(dim)
            result = func.evaluate(x_opt)
            assert result == 0.0, f"Expected 0.0 at origin for dim={dim}, got {result}"

    def test_get_global_minimum(self):
        """Test get_global_minimum method returns correct values."""
        for dim in [2, 5, 10, 30]:
            func = DiscusFunction(dimension=dim)
            min_point, min_value = func.get_global_minimum()
            np.testing.assert_array_equal(min_point, np.zeros(dim))
            assert min_value == 0.0

    def test_get_global_minimum_consistency(self):
        """Test that evaluating at the returned global minimum gives the stated value."""
        for dim in [2, 5, 10]:
            func = DiscusFunction(dimension=dim)
            min_point, min_value = func.get_global_minimum()
            result = func.evaluate(min_point)
            assert abs(result - min_value) < 1e-12

    def test_known_values_first_component_only(self):
        """Test f(1,0,...,0) = 10^6 (first component gets 10^6 scaling)."""
        for dim in [2, 5, 10]:
            func = DiscusFunction(dimension=dim)
            x = np.zeros(dim)
            x[0] = 1.0
            result = func.evaluate(x)
            assert abs(result - 1e6) < 1e-6, (
                f"f(1,0,...,0) should be 1e6 for dim={dim}, got {result}"
            )

    def test_known_values_second_component_only(self):
        """Test f(0,1,0,...,0) = 1 (second component contributes normally)."""
        for dim in [2, 5, 10]:
            func = DiscusFunction(dimension=dim)
            x = np.zeros(dim)
            x[1] = 1.0
            result = func.evaluate(x)
            assert abs(result - 1.0) < 1e-12, (
                f"f(0,1,0,...,0) should be 1.0 for dim={dim}, got {result}"
            )

    def test_known_values_all_ones_dim3(self):
        """Test f(1,1,...,1) for D=3: 10^6 + 1 + 1 = 10^6 + 2."""
        func = DiscusFunction(dimension=3)
        x = np.ones(3)
        result = func.evaluate(x)
        expected = 1e6 + 2.0
        assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

    def test_known_values_all_ones_general(self):
        """Test f(1,...,1) = 10^6 + (D-1) for general dimensions."""
        for dim in [2, 5, 10]:
            func = DiscusFunction(dimension=dim)
            x = np.ones(dim)
            result = func.evaluate(x)
            expected = 1e6 + (dim - 1)
            assert abs(result - expected) < 1e-4, (
                f"f(1,...,1) for dim={dim}: expected {expected}, got {result}"
            )

    def test_evaluate_batch(self):
        """Test that batch evaluation matches individual evaluation."""
        func = DiscusFunction(dimension=5)
        rng = np.random.default_rng(42)
        X = rng.uniform(-100, 100, size=(10, 5))
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test that batch output has correct shape."""
        func = DiscusFunction(dimension=5)
        X = np.zeros((7, 5))
        results = func.evaluate_batch(X)
        assert results.shape == (7,), f"Expected shape (7,), got {results.shape}"

    def test_dimension_validation(self):
        """Test that wrong input shape raises an error."""
        func = DiscusFunction(dimension=5)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))  # Wrong dimension

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension

    def test_inverse_relationship_with_bent_cigar(self):
        """Test that Discus is the inverse of BentCigar in terms of conditioning.

        For BentCigar, x1 has normal conditioning and the rest are scaled by 10^6.
        For Discus, x1 is scaled by 10^6 and the rest have normal conditioning.
        """
        from pyMOFL.functions.benchmark.bent_cigar import BentCigarFunction

        dim = 5
        discus = DiscusFunction(dimension=dim)
        bent_cigar = BentCigarFunction(dimension=dim)

        # e1 = (1, 0, ..., 0)
        e1 = np.zeros(dim)
        e1[0] = 1.0

        # e2 = (0, 1, 0, ..., 0)
        e2 = np.zeros(dim)
        e2[1] = 1.0

        # For Discus: f(e1) = 10^6, f(e2) = 1
        # For BentCigar: f(e1) = 1, f(e2) = 10^6
        assert discus.evaluate(e1) == pytest.approx(1e6)
        assert discus.evaluate(e2) == pytest.approx(1.0)
        assert bent_cigar.evaluate(e1) == pytest.approx(1.0)
        assert bent_cigar.evaluate(e2) == pytest.approx(1e6)

        # The roles are swapped
        assert discus.evaluate(e1) == pytest.approx(bent_cigar.evaluate(e2))
        assert discus.evaluate(e2) == pytest.approx(bent_cigar.evaluate(e1))

    def test_non_negativity(self):
        """Test that function values are always non-negative."""
        func = DiscusFunction(dimension=5)
        rng = np.random.default_rng(123)
        for _ in range(50):
            x = rng.uniform(-100, 100, size=5)
            result = func.evaluate(x)
            assert result >= 0.0, f"Function should be non-negative, got {result}"

    def test_symmetry_in_tail_components(self):
        """Test that function is symmetric across tail components (x2..xD)."""
        func = DiscusFunction(dimension=4)
        x1 = np.array([0.0, 1.0, 2.0, 3.0])
        x2 = np.array([0.0, 3.0, 1.0, 2.0])
        assert func.evaluate(x1) == pytest.approx(func.evaluate(x2))

    def test_registry_names(self):
        """Test that all registry names work."""
        from pyMOFL.registry import get

        func1 = get("discus")(dimension=3)
        func2 = get("Discus")(dimension=3)
        assert isinstance(func1, DiscusFunction)
        assert isinstance(func2, DiscusFunction)

        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) == func2.evaluate(x)
