"""
Tests for Schwefel function variants (2.20, 2.21, 2.22, 2.23, 2.25, 2.26).

References:
    Schwefel, H.P. (1981). "Numerical Optimization of Computer Models".
    Jamil, M., & Yang, X.S. (2013). arXiv:1308.4008
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.schwefel import (
    Schwefel_2_20,
    Schwefel_2_21,
    Schwefel_2_22,
    Schwefel_2_23,
    Schwefel_2_25,
    Schwefel_2_26,
    SchwefelFunction,
)


# ---------------------------------------------------------------------------
# Schwefel 2.20: f(x) = Σ |x_i|   (Manhattan / L1 norm)
# ---------------------------------------------------------------------------
class TestSchwefel_2_20:
    """Tests for Schwefel 2.20 — absolute value sum."""

    def test_initialization(self):
        func = Schwefel_2_20(dimension=3)
        assert func.dimension == 3
        np.testing.assert_allclose(func.operational_bounds.low, [-100, -100, -100])
        np.testing.assert_allclose(func.operational_bounds.high, [100, 100, 100])

    def test_global_minimum(self):
        for d in [2, 5, 10]:
            func = Schwefel_2_20(dimension=d)
            assert np.isclose(func.evaluate(np.zeros(d)), 0.0, atol=1e-15)

    def test_get_global_minimum(self):
        func = Schwefel_2_20(dimension=3)
        point, value = func.get_global_minimum()
        np.testing.assert_array_equal(point, [0, 0, 0])
        assert value == 0.0

    def test_known_values(self):
        func = Schwefel_2_20(dimension=3)
        assert np.isclose(func.evaluate(np.array([1.0, -2.0, 3.0])), 6.0)
        assert np.isclose(func.evaluate(np.array([-5.0, 5.0, 0.0])), 10.0)

    def test_symmetry(self):
        func = Schwefel_2_20(dimension=2)
        x = np.array([3.0, -7.0])
        val = func.evaluate(x)
        assert np.isclose(func.evaluate(-x), val)
        assert np.isclose(func.evaluate(np.abs(x)), val)

    def test_non_negative(self):
        func = Schwefel_2_20(dimension=5)
        rng = np.random.default_rng(42)
        for _ in range(50):
            x = rng.uniform(-100, 100, 5)
            assert func.evaluate(x) >= 0.0

    def test_batch_evaluation(self):
        func = Schwefel_2_20(dimension=3)
        X = np.array([[0, 0, 0], [1, -2, 3], [-5, 5, 0]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-12)

    def test_batch_shape(self):
        func = Schwefel_2_20(dimension=4)
        assert func.evaluate_batch(np.zeros((3, 4))).shape == (3,)

    def test_dimension_validation(self):
        func = Schwefel_2_20(dimension=3)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Schwefel 2.21: f(x) = max_i |x_i|   (Chebyshev / L∞ norm)
# ---------------------------------------------------------------------------
class TestSchwefel_2_21:
    """Tests for Schwefel 2.21 — maximum absolute value."""

    def test_initialization(self):
        func = Schwefel_2_21(dimension=3)
        assert func.dimension == 3
        np.testing.assert_allclose(func.operational_bounds.low, [-100, -100, -100])

    def test_global_minimum(self):
        for d in [2, 5]:
            func = Schwefel_2_21(dimension=d)
            assert np.isclose(func.evaluate(np.zeros(d)), 0.0, atol=1e-15)

    def test_get_global_minimum(self):
        func = Schwefel_2_21(dimension=4)
        point, value = func.get_global_minimum()
        np.testing.assert_array_equal(point, np.zeros(4))
        assert value == 0.0

    def test_known_values(self):
        func = Schwefel_2_21(dimension=3)
        assert np.isclose(func.evaluate(np.array([1.0, -5.0, 3.0])), 5.0)
        assert np.isclose(func.evaluate(np.array([-2.0, 3.0, -7.0])), 7.0)

    def test_single_large_dimension_dominates(self):
        """Only the worst dimension matters."""
        func = Schwefel_2_21(dimension=5)
        x = np.array([0.01, 0.01, 0.01, 0.01, 99.0])
        assert np.isclose(func.evaluate(x), 99.0)

    def test_non_negative(self):
        func = Schwefel_2_21(dimension=3)
        rng = np.random.default_rng(42)
        for _ in range(50):
            assert func.evaluate(rng.uniform(-100, 100, 3)) >= 0.0

    def test_batch_evaluation(self):
        func = Schwefel_2_21(dimension=3)
        X = np.array([[0, 0, 0], [1, -5, 3], [-2, 3, -7]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-12)

    def test_batch_shape(self):
        func = Schwefel_2_21(dimension=4)
        assert func.evaluate_batch(np.zeros((5, 4))).shape == (5,)


# ---------------------------------------------------------------------------
# Schwefel 2.22: f(x) = Σ|x_i| + Π|x_i|   (Sum + Product)
# ---------------------------------------------------------------------------
class TestSchwefel_2_22:
    """Tests for Schwefel 2.22 — sum of absolute values plus product."""

    def test_initialization(self):
        func = Schwefel_2_22(dimension=3)
        assert func.dimension == 3
        np.testing.assert_allclose(func.operational_bounds.low, [-100, -100, -100])

    def test_global_minimum(self):
        for d in [2, 5]:
            func = Schwefel_2_22(dimension=d)
            assert np.isclose(func.evaluate(np.zeros(d)), 0.0, atol=1e-15)

    def test_get_global_minimum(self):
        func = Schwefel_2_22(dimension=3)
        point, value = func.get_global_minimum()
        np.testing.assert_array_equal(point, np.zeros(3))
        assert value == 0.0

    def test_known_values(self):
        func = Schwefel_2_22(dimension=3)
        # f(1,2,3) = (1+2+3) + (1*2*3) = 6 + 6 = 12
        assert np.isclose(func.evaluate(np.array([1.0, 2.0, 3.0])), 12.0)
        # f(-1,-1) = 2 + 1 = 3
        func2 = Schwefel_2_22(dimension=2)
        assert np.isclose(func2.evaluate(np.array([-1.0, -1.0])), 3.0)

    def test_product_term_dominates_far_from_origin(self):
        """The product term grows exponentially faster than the sum term."""
        func = Schwefel_2_22(dimension=5)
        x = np.full(5, 10.0)
        sum_part = 50.0
        prod_part = 100000.0
        assert np.isclose(func.evaluate(x), sum_part + prod_part)

    def test_zero_in_any_dim_kills_product(self):
        """If any x_i = 0, the product term is 0."""
        func = Schwefel_2_22(dimension=3)
        x = np.array([5.0, 0.0, 3.0])
        assert np.isclose(func.evaluate(x), 8.0)  # sum only

    def test_non_negative(self):
        func = Schwefel_2_22(dimension=3)
        rng = np.random.default_rng(42)
        for _ in range(50):
            assert func.evaluate(rng.uniform(-100, 100, 3)) >= 0.0

    def test_batch_evaluation(self):
        func = Schwefel_2_22(dimension=3)
        X = np.array([[0, 0, 0], [1, 2, 3], [5, 0, 3]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-12)

    def test_batch_shape(self):
        func = Schwefel_2_22(dimension=4)
        assert func.evaluate_batch(np.zeros((3, 4))).shape == (3,)


# ---------------------------------------------------------------------------
# Schwefel 2.23: f(x) = Σ x_i^10   (Power function)
# ---------------------------------------------------------------------------
class TestSchwefel_2_23:
    """Tests for Schwefel 2.23 — 10th power sum."""

    def test_initialization(self):
        func = Schwefel_2_23(dimension=3)
        assert func.dimension == 3
        np.testing.assert_allclose(func.operational_bounds.low, [-10, -10, -10])
        np.testing.assert_allclose(func.operational_bounds.high, [10, 10, 10])

    def test_global_minimum(self):
        for d in [2, 5]:
            func = Schwefel_2_23(dimension=d)
            assert np.isclose(func.evaluate(np.zeros(d)), 0.0, atol=1e-15)

    def test_get_global_minimum(self):
        func = Schwefel_2_23(dimension=3)
        point, value = func.get_global_minimum()
        np.testing.assert_array_equal(point, np.zeros(3))
        assert value == 0.0

    def test_known_values(self):
        func = Schwefel_2_23(dimension=2)
        # f(1,1) = 1 + 1 = 2
        assert np.isclose(func.evaluate(np.array([1.0, 1.0])), 2.0)
        # f(2,0) = 2^10 = 1024
        assert np.isclose(func.evaluate(np.array([2.0, 0.0])), 1024.0)
        # f(0.5, 0.5) = 2 * 0.5^10 ≈ 0.001953125
        assert np.isclose(func.evaluate(np.array([0.5, 0.5])), 2 * 0.5**10)

    def test_flat_near_origin(self):
        """Gradient near zero is essentially flat — key property of this function."""
        func = Schwefel_2_23(dimension=2)
        # f(0.1, 0.1) = 2 * 0.1^10 = 2e-10
        val = func.evaluate(np.array([0.1, 0.1]))
        assert val < 1e-9

    def test_even_power_symmetry(self):
        """x^10 is even, so f(x) = f(-x)."""
        func = Schwefel_2_23(dimension=3)
        x = np.array([1.5, -2.0, 0.7])
        assert np.isclose(func.evaluate(x), func.evaluate(-x))

    def test_non_negative(self):
        func = Schwefel_2_23(dimension=3)
        rng = np.random.default_rng(42)
        for _ in range(50):
            assert func.evaluate(rng.uniform(-10, 10, 3)) >= 0.0

    def test_batch_evaluation(self):
        func = Schwefel_2_23(dimension=2)
        X = np.array([[0, 0], [1, 1], [2, 0], [0.5, 0.5]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-12)

    def test_batch_shape(self):
        func = Schwefel_2_23(dimension=3)
        assert func.evaluate_batch(np.zeros((4, 3))).shape == (4,)


# ---------------------------------------------------------------------------
# Schwefel 2.25: f(x) = Σ_{i=1}^{D-1} (x_i² - x_{i+1})² + (x_i - 1)²
#                (Rosenbrock variant without 100x scaling)
# ---------------------------------------------------------------------------
class TestSchwefel_2_25:
    """Tests for Schwefel 2.25 — Rosenbrock-like variable coupling."""

    def test_initialization(self):
        func = Schwefel_2_25(dimension=3)
        assert func.dimension == 3
        np.testing.assert_allclose(func.operational_bounds.low, [0, 0, 0])
        np.testing.assert_allclose(func.operational_bounds.high, [10, 10, 10])

    def test_global_minimum(self):
        for d in [2, 3, 5]:
            func = Schwefel_2_25(dimension=d)
            x_opt = np.ones(d)
            assert np.isclose(func.evaluate(x_opt), 0.0, atol=1e-15)

    def test_get_global_minimum(self):
        func = Schwefel_2_25(dimension=4)
        point, value = func.get_global_minimum()
        np.testing.assert_array_equal(point, np.ones(4))
        assert value == 0.0

    def test_known_value(self):
        func = Schwefel_2_25(dimension=3)
        # f(2, 4, 16) = (4-4)^2 + (2-1)^2 + (16-16)^2 + (4-1)^2 = 0+1+0+9 = 10
        assert np.isclose(func.evaluate(np.array([2.0, 4.0, 16.0])), 10.0)

    def test_non_separable(self):
        """Changing x_i affects the term involving x_{i+1}."""
        func = Schwefel_2_25(dimension=2)
        # f(a, b) = (a^2 - b)^2 + (a-1)^2
        # f(1, 1) = 0, f(2, 1) = (4-1)^2 + (2-1)^2 = 9+1 = 10
        assert np.isclose(func.evaluate(np.array([2.0, 1.0])), 10.0)

    def test_non_negative(self):
        func = Schwefel_2_25(dimension=3)
        rng = np.random.default_rng(42)
        for _ in range(50):
            assert func.evaluate(rng.uniform(0, 10, 3)) >= 0.0

    def test_batch_evaluation(self):
        func = Schwefel_2_25(dimension=3)
        X = np.array([[1, 1, 1], [2, 4, 16], [2, 1, 1]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-12)

    def test_batch_shape(self):
        func = Schwefel_2_25(dimension=4)
        assert func.evaluate_batch(np.ones((3, 4))).shape == (3,)

    def test_dimension_validation(self):
        func = Schwefel_2_25(dimension=3)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Schwefel 2.26: f(x) = -Σ x_i sin(sqrt(|x_i|))
#                (pure form, no 418.9829*D offset)
# ---------------------------------------------------------------------------
class TestSchwefel_2_26:
    """Tests for Schwefel 2.26 — the standard Schwefel (pure form)."""

    def test_initialization(self):
        func = Schwefel_2_26(dimension=3)
        assert func.dimension == 3
        np.testing.assert_allclose(func.operational_bounds.low, [-500, -500, -500])
        np.testing.assert_allclose(func.operational_bounds.high, [500, 500, 500])

    def test_value_at_origin(self):
        func = Schwefel_2_26(dimension=2)
        assert np.isclose(func.evaluate(np.zeros(2)), 0.0, atol=1e-15)

    def test_global_minimum_region(self):
        """f* ≈ -418.983 per dimension at x_i ≈ 420.9687."""
        func = Schwefel_2_26(dimension=1)
        val = func.evaluate(np.array([420.9687]))
        assert np.isclose(val, -418.9829, atol=0.01)

    def test_global_minimum_d2(self):
        func = Schwefel_2_26(dimension=2)
        x_opt = np.full(2, 420.9687)
        val = func.evaluate(x_opt)
        assert np.isclose(val, -837.9658, atol=0.01)

    def test_get_global_minimum(self):
        func = Schwefel_2_26(dimension=3)
        point, value = func.get_global_minimum()
        np.testing.assert_allclose(point, np.full(3, 420.9687), atol=1e-4)
        assert value < -1250  # 3 * -418.98 ≈ -1256.95

    def test_multimodal(self):
        """Many local minima — second best is far from global."""
        func = Schwefel_2_26(dimension=2)
        rng = np.random.default_rng(42)
        vals = [func.evaluate(rng.uniform(-500, 500, 2)) for _ in range(200)]
        assert max(vals) - min(vals) > 100  # wide range of values

    def test_separable(self):
        """Each dimension contributes independently."""
        func = Schwefel_2_26(dimension=2)
        func1d = Schwefel_2_26(dimension=1)
        x = np.array([100.0, 200.0])
        expected = func1d.evaluate(np.array([100.0])) + func1d.evaluate(np.array([200.0]))
        assert np.isclose(func.evaluate(x), expected, atol=1e-10)

    def test_batch_evaluation(self):
        func = Schwefel_2_26(dimension=2)
        X = np.array([[0, 0], [420.9687, 420.9687], [100, -200]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-10)

    def test_batch_shape(self):
        func = Schwefel_2_26(dimension=3)
        assert func.evaluate_batch(np.zeros((4, 3))).shape == (4,)


# ---------------------------------------------------------------------------
# SchwefelFunction (offset form): f(x) = 418.9829*D - Σ x_i sin(sqrt(|x_i|))
# Verify the existing implementation is correct (was labeled STUB)
# ---------------------------------------------------------------------------
class TestSchwefelFunction:
    """Tests for SchwefelFunction — offset form with f* ≈ 0."""

    def test_global_minimum_near_zero(self):
        func = SchwefelFunction(dimension=2)
        x_opt = np.full(2, 420.9687)
        val = func.evaluate(x_opt)
        assert abs(val) < 0.01  # Should be very close to 0

    def test_get_global_minimum(self):
        func = SchwefelFunction(dimension=3)
        point, value = func.get_global_minimum()
        np.testing.assert_allclose(point, np.full(3, 420.9687), atol=1e-4)
        assert value == 0.0  # Nominal value

    def test_value_at_origin(self):
        """f(0,...,0) = 418.9829*D."""
        func = SchwefelFunction(dimension=3)
        expected = 418.9829 * 3
        assert np.isclose(func.evaluate(np.zeros(3)), expected, atol=0.01)

    def test_batch_matches_individual(self):
        func = SchwefelFunction(dimension=2)
        X = np.array([[0, 0], [420.9687, 420.9687], [100, -200]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-10)

    def test_schwefel_function_and_2_26_differ_by_offset(self):
        """SchwefelFunction = 418.9829*D + Schwefel_2_26."""
        d = 3
        x = np.array([100.0, -200.0, 300.0])
        sf = SchwefelFunction(dimension=d)
        s26 = Schwefel_2_26(dimension=d)
        offset = 418.9829 * d
        assert np.isclose(sf.evaluate(x), s26.evaluate(x) + offset, atol=1e-8)
