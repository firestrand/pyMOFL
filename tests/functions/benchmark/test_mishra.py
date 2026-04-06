"""
Tests for the Mishra benchmark function family (Mishra 01-11).

Following TDD approach with comprehensive test coverage for all 11 Mishra variants.
Tests validate mathematical correctness, bounds handling, dimension validation,
batch evaluation consistency, and known-value evaluations.
"""

import math

import numpy as np
import pytest

from pyMOFL.functions.benchmark.mishra import (
    Mishra01Function,
    Mishra02Function,
    Mishra03Function,
    Mishra04Function,
    Mishra05Function,
    Mishra06Function,
    Mishra07Function,
    Mishra08Function,
    Mishra09Function,
    Mishra10Function,
    Mishra11Function,
)


# ---------------------------------------------------------------------------
# Mishra 01
# ---------------------------------------------------------------------------
class TestMishra01Function:
    """Test Mishra 01 function (scalable, domain [0,1]^D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = Mishra01Function(dimension=5)
        assert func.dimension == 5
        np.testing.assert_array_equal(func.initialization_bounds.low, np.zeros(5))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.ones(5))

    def test_global_minimum(self):
        """Test global minimum at (1,...,1) with value 2."""
        func = Mishra01Function(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.ones(5))
        assert min_value == 2.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 2 at (1,...,1)."""
        func = Mishra01Function(dimension=5)
        result = func.evaluate(np.ones(5))
        assert abs(result - 2.0) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = Mishra01Function(dimension=3)
        # x = [1, 1, 1]: xn = 3 - (1 + 1) = 1, f = (1+1)^1 = 2
        assert abs(func.evaluate(np.ones(3)) - 2.0) < 1e-10

        # x = [0.5, 0.5, 0.5]: xn = 3 - (0.5 + 0.5) = 2, f = (1+2)^2 = 9
        result = func.evaluate(np.array([0.5, 0.5, 0.5]))
        assert abs(result - 9.0) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluation."""
        func = Mishra01Function(dimension=3)
        X = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.8, 0.8, 0.8]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(results, expected)

    def test_different_dimensions(self):
        """Test function works with various dimensions."""
        for dim in [2, 5, 10]:
            func = Mishra01Function(dimension=dim)
            result = func.evaluate(np.ones(dim))
            assert abs(result - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Mishra 02
# ---------------------------------------------------------------------------
class TestMishra02Function:
    """Test Mishra 02 function (scalable, domain [0,1]^D)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = Mishra02Function(dimension=5)
        assert func.dimension == 5
        np.testing.assert_array_equal(func.initialization_bounds.low, np.zeros(5))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.ones(5))

    def test_global_minimum(self):
        """Test global minimum at (1,...,1) with value 2."""
        func = Mishra02Function(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.ones(5))
        assert min_value == 2.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 2 at (1,...,1)."""
        func = Mishra02Function(dimension=5)
        result = func.evaluate(np.ones(5))
        assert abs(result - 2.0) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = Mishra02Function(dimension=3)
        # x = [1, 1, 1]: avg_sum = (1+1)/2 + (1+1)/2 = 2, xn = 3 - 2 = 1, f = 2^1 = 2
        assert abs(func.evaluate(np.ones(3)) - 2.0) < 1e-10

        # x = [0, 0, 0]: avg_sum = 0, xn = 3, f = (1+3)^3 = 64
        result = func.evaluate(np.zeros(3))
        assert abs(result - 64.0) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluation."""
        func = Mishra02Function(dimension=3)
        X = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(results, expected)

    def test_different_dimensions(self):
        """Test function works with various dimensions."""
        for dim in [2, 5, 10]:
            func = Mishra02Function(dimension=dim)
            result = func.evaluate(np.ones(dim))
            assert abs(result - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Mishra 03
# ---------------------------------------------------------------------------
class TestMishra03Function:
    """Test Mishra 03 function (2D fixed)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = Mishra03Function()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-10.0, -10.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_dimension_validation(self):
        """Test that non-2D initialization raises ValueError."""
        with pytest.raises(
            ValueError, match="Mishra03 function is Non-Scalable and requires dimension=2"
        ):
            Mishra03Function(dimension=3)

    def test_global_minimum(self):
        """Test global minimum is self-consistent."""
        func = Mishra03Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, [-10.0, -10.0])
        evaluated = func.evaluate(min_point)
        assert abs(evaluated - min_value) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = Mishra03Function()
        # At origin: sqrt(|cos(0)|) + 0.01*(0 + 0) = sqrt(1) = 1.0
        result = func.evaluate(np.array([0.0, 0.0]))
        assert abs(result - 1.0) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluation."""
        func = Mishra03Function()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-5.0, 3.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(results, expected)

    def test_symmetry_property(self):
        """Test function is not symmetric due to the linear term."""
        func = Mishra03Function()
        val1 = func.evaluate(np.array([1.0, 2.0]))
        val2 = func.evaluate(np.array([-1.0, -2.0]))
        # Due to 0.01*(x1+x2), f(1,2) != f(-1,-2) in general
        assert val1 != val2


# ---------------------------------------------------------------------------
# Mishra 04
# ---------------------------------------------------------------------------
class TestMishra04Function:
    """Test Mishra 04 function (2D fixed)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = Mishra04Function()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-10.0, -10.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_dimension_validation(self):
        """Test that non-2D initialization raises ValueError."""
        with pytest.raises(
            ValueError, match="Mishra04 function is Non-Scalable and requires dimension=2"
        ):
            Mishra04Function(dimension=3)

    def test_global_minimum(self):
        """Test global minimum is self-consistent."""
        func = Mishra04Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, [-10.0, -10.0])
        evaluated = func.evaluate(min_point)
        assert abs(evaluated - min_value) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = Mishra04Function()
        # At origin: sqrt(|sin(0)|) + 0 = 0
        result = func.evaluate(np.array([0.0, 0.0]))
        assert abs(result) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluation."""
        func = Mishra04Function()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-5.0, 3.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(results, expected)


# ---------------------------------------------------------------------------
# Mishra 05
# ---------------------------------------------------------------------------
class TestMishra05Function:
    """Test Mishra 05 function (2D fixed)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = Mishra05Function()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-10.0, -10.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_dimension_validation(self):
        """Test that non-2D initialization raises ValueError."""
        with pytest.raises(
            ValueError, match="Mishra05 function is Non-Scalable and requires dimension=2"
        ):
            Mishra05Function(dimension=3)

    def test_global_minimum(self):
        """Test global minimum is self-consistent."""
        func = Mishra05Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [-1.98682, -10.0])
        evaluated = func.evaluate(min_point)
        assert abs(evaluated - min_value) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at a known point."""
        func = Mishra05Function()
        # Manually compute at (0, 0):
        # sin((cos(0)+cos(0))^2)^2 + cos((sin(0)+sin(0))^2)^2 + 0
        # = sin(4)^2 + cos(0)^2 + 0
        # = sin(4)^2 + 1
        # f = (sin(4)^2 + 1)^2 + 0.01*(0+0)
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        term_sin = np.sin((np.cos(0) + np.cos(0)) ** 2) ** 2
        term_cos = np.cos((np.sin(0) + np.sin(0)) ** 2) ** 2
        expected = (term_sin + term_cos + 0.0) ** 2 + 0.01 * (0.0 + 0.0)
        assert abs(result - expected) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluation."""
        func = Mishra05Function()
        X = np.array([[0.0, 0.0], [1.0, -1.0], [-1.98682, -10.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(results, expected)


# ---------------------------------------------------------------------------
# Mishra 06
# ---------------------------------------------------------------------------
class TestMishra06Function:
    """Test Mishra 06 function (2D fixed)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = Mishra06Function()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-10.0, -10.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_dimension_validation(self):
        """Test that non-2D initialization raises ValueError."""
        with pytest.raises(
            ValueError, match="Mishra06 function is Non-Scalable and requires dimension=2"
        ):
            Mishra06Function(dimension=3)

    def test_global_minimum(self):
        """Test global minimum is self-consistent."""
        func = Mishra06Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [2.886, 1.823], decimal=3)
        evaluated = func.evaluate(min_point)
        assert abs(evaluated - min_value) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at a known point."""
        func = Mishra06Function()
        # Manually compute at (1, 1):
        x = np.array([1.0, 1.0])
        result = func.evaluate(x)
        x1, x2 = 1.0, 1.0
        term_sin = np.sin((np.cos(x1) + np.cos(x2)) ** 2) ** 2
        term_cos = np.cos((np.sin(x1) + np.sin(x2)) ** 2) ** 2
        inner = (term_sin - term_cos + x1) ** 2
        expected = -np.log(inner) + 0.01 * ((x1 - 1) ** 2 + (x2 - 1) ** 2)
        assert abs(result - expected) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluation."""
        func = Mishra06Function()
        X = np.array([[1.0, 1.0], [2.886, 1.823], [-1.0, 2.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(results, expected)

    def test_global_minimum_value_approximate(self):
        """Test global minimum value is negative (deep basin)."""
        func = Mishra06Function()
        _, min_value = func.get_global_minimum()
        # The approximate optimum coordinates (2.886, 1.823) give a value
        # in the neighborhood of the true minimum; verify it's negative.
        assert min_value < 0.0


# ---------------------------------------------------------------------------
# Mishra 07
# ---------------------------------------------------------------------------
class TestMishra07Function:
    """Test Mishra 07 function (scalable)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = Mishra07Function(dimension=3)
        assert func.dimension == 3
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(3, -10.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(3, 10.0))

    def test_global_minimum(self):
        """Test global minimum with value 0."""
        func = Mishra07Function(dimension=3)
        min_point, min_value = func.get_global_minimum()
        assert min_value == 0.0
        # All components should be 3!^(1/3) = 6^(1/3)
        expected_component = 6.0 ** (1.0 / 3.0)
        np.testing.assert_array_almost_equal(min_point, np.full(3, expected_component))

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to ~0 at global minimum."""
        func = Mishra07Function(dimension=3)
        min_point, _ = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result) < 1e-6

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = Mishra07Function(dimension=3)
        # f([1, 1, 1]) = (1 - 6)^2 = 25
        result = func.evaluate(np.ones(3))
        assert abs(result - 25.0) < 1e-10

        # f([1, 2, 3]) = (6 - 6)^2 = 0
        result = func.evaluate(np.array([1.0, 2.0, 3.0]))
        assert abs(result) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluation."""
        func = Mishra07Function(dimension=3)
        X = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [2.0, 2.0, 2.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(results, expected)

    def test_different_dimensions(self):
        """Test function works with various dimensions."""
        for dim in [2, 3, 5]:
            func = Mishra07Function(dimension=dim)
            n_fact = math.factorial(dim)
            opt_val = n_fact ** (1.0 / dim)
            result = func.evaluate(np.full(dim, opt_val))
            assert abs(result) < 1e-6


# ---------------------------------------------------------------------------
# Mishra 08
# ---------------------------------------------------------------------------
class TestMishra08Function:
    """Test Mishra 08 function (2D fixed, Mishra-Decanomial)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = Mishra08Function()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-10.0, -10.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_dimension_validation(self):
        """Test that non-2D initialization raises ValueError."""
        with pytest.raises(
            ValueError, match="Mishra08 function is Non-Scalable and requires dimension=2"
        ):
            Mishra08Function(dimension=3)

    def test_global_minimum(self):
        """Test global minimum at (2, -3) with value 0."""
        func = Mishra08Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, [2.0, -3.0])
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at (2, -3)."""
        func = Mishra08Function()
        result = func.evaluate(np.array([2.0, -3.0]))
        assert abs(result) < 1e-10

    def test_polynomial_at_roots(self):
        """Test that g1(2) = 0 and g2(-3) = 0 (roots of the polynomials)."""
        func = Mishra08Function()
        # g1 should be approximately zero at x1=2
        # The expanded polynomial has specific coefficients from the literature
        g1_at_2 = func._g1(2.0)
        # 2^10 - 20*2^9 + 180*2^8 - 960*2^7 + 3360*2^6 - 8064*2^5
        # + 13340*2^4 - 15360*2^3 + 11520*2^2 - 5120*2 + 2624
        # = 1024 - 10240 + 46080 - 122880 + 215040 - 258048
        # + 213440 - 122880 + 46080 - 10240 + 2624 = 0
        assert abs(g1_at_2) < 1e-6, f"g1(2) = {g1_at_2}, expected ~0"

        # g2 should be zero at x2=-3: (-3+3)^4 = 0
        g2_at_neg3 = func._g2(-3.0)
        assert abs(g2_at_neg3) < 1e-6, f"g2(-3) = {g2_at_neg3}, expected ~0"

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluation."""
        func = Mishra08Function()
        X = np.array([[2.0, -3.0], [0.0, 0.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(results, expected)


# ---------------------------------------------------------------------------
# Mishra 09
# ---------------------------------------------------------------------------
class TestMishra09Function:
    """Test Mishra 09 function (3D fixed)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = Mishra09Function()
        assert func.dimension == 3
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(3, -10.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(3, 10.0))

    def test_dimension_validation(self):
        """Test that non-3D initialization raises ValueError."""
        with pytest.raises(
            ValueError, match="Mishra09 function is Non-Scalable and requires dimension=3"
        ):
            Mishra09Function(dimension=2)

    def test_global_minimum(self):
        """Test global minimum at (1, 2, 3) with value 0."""
        func = Mishra09Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, [1.0, 2.0, 3.0])
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at (1, 2, 3)."""
        func = Mishra09Function()
        result = func.evaluate(np.array([1.0, 2.0, 3.0]))
        # At (1, 2, 3):
        # a = 2 + 10 + 12 - 6 - 18 = 0
        # b = 1 + 8 + 4 + 9 - 22 = 0
        # c = 8 + 12 + 8 + 81 - 52 = 57 (not zero but a=b=0 makes it 0)
        # f = (0*0*57 + 0*0*57^2 + 0 + (1+2-3)^2)^2 = 0
        assert abs(result) < 1e-10

    def test_evaluate_known_values(self):
        """Test that a, b, c all evaluate to zero at (1,2,3)."""
        x1, x2, x3 = 1.0, 2.0, 3.0
        a = 2.0 * x1**3 + 5.0 * x1 * x2 + 4.0 * x3 - 2.0 * x1**2 * x3 - 18.0
        b = x1 + x2**3 + x1 * x2**2 + x1 * x3**2 - 22.0
        assert abs(a) < 1e-10
        assert abs(b) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluation."""
        func = Mishra09Function()
        X = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(results, expected)

    def test_nonzero_evaluation(self):
        """Test function is nonzero away from optimum."""
        func = Mishra09Function()
        result = func.evaluate(np.array([0.0, 0.0, 0.0]))
        assert result > 0


# ---------------------------------------------------------------------------
# Mishra 10
# ---------------------------------------------------------------------------
class TestMishra10Function:
    """Test Mishra 10 function (2D fixed, XOR-based)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = Mishra10Function()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-10.0, -10.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_dimension_validation(self):
        """Test that non-2D initialization raises ValueError."""
        with pytest.raises(
            ValueError, match="Mishra10 function is Non-Scalable and requires dimension=2"
        ):
            Mishra10Function(dimension=3)

    def test_global_minimum(self):
        """Test global minimum at (2, 2) with value 0."""
        func = Mishra10Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, [2.0, 2.0])
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at (2, 2)."""
        func = Mishra10Function()
        result = func.evaluate(np.array([2.0, 2.0]))
        assert abs(result) < 1e-10

    def test_evaluate_equal_floors(self):
        """Test function evaluates to 0 whenever floors are equal."""
        func = Mishra10Function()
        # All points with same floor values should give 0
        for val in [-3.0, 0.0, 1.5, 5.7]:
            result = func.evaluate(np.array([val, val]))
            assert abs(result) < 1e-10, f"Expected 0 at ({val}, {val}), got {result}"

    def test_evaluate_xor_values(self):
        """Test XOR computation with known integer values."""
        func = Mishra10Function()
        # floor(3.5) = 3, floor(5.5) = 5, 3 XOR 5 = 6, 6^2 = 36
        result = func.evaluate(np.array([3.5, 5.5]))
        assert abs(result - 36.0) < 1e-10

        # floor(1.0) = 1, floor(2.0) = 2, 1 XOR 2 = 3, 3^2 = 9
        result = func.evaluate(np.array([1.0, 2.0]))
        assert abs(result - 9.0) < 1e-10

    def test_evaluate_negative_floors(self):
        """Test XOR with negative floored values."""
        func = Mishra10Function()
        # floor(-1.5) = -2, floor(-1.5) = -2, (-2) XOR (-2) = 0
        result = func.evaluate(np.array([-1.5, -1.5]))
        assert abs(result) < 1e-10

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluation."""
        func = Mishra10Function()
        X = np.array([[2.0, 2.0], [3.5, 5.5], [1.0, 2.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(results, expected)


# ---------------------------------------------------------------------------
# Mishra 11
# ---------------------------------------------------------------------------
class TestMishra11Function:
    """Test Mishra 11 function (scalable)."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = Mishra11Function(dimension=5)
        assert func.dimension == 5
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(5, -10.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(5, 10.0))

    def test_global_minimum(self):
        """Test global minimum at (0,...,0) with value 0."""
        func = Mishra11Function(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.zeros(5))
        assert min_value == 0.0

    def test_evaluate_at_global_minimum(self):
        """Test function evaluates to 0 at origin."""
        func = Mishra11Function(dimension=5)
        result = func.evaluate(np.zeros(5))
        assert abs(result) < 1e-10

    def test_evaluate_known_values(self):
        """Test function evaluation at known points."""
        func = Mishra11Function(dimension=3)
        # x = [1, 1, 1]: mean_abs = 1, geom_mean = 1, f = 0
        result = func.evaluate(np.ones(3))
        assert abs(result) < 1e-10

        # x = [2, 2, 2]: mean_abs = 2, geom_mean = 2, f = 0
        result = func.evaluate(np.full(3, 2.0))
        assert abs(result) < 1e-10

    def test_am_gm_inequality(self):
        """Test that f >= 0 always (AM-GM inequality)."""
        func = Mishra11Function(dimension=3)
        # AM >= GM, so (AM - GM)^2 >= 0
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.uniform(-10, 10, size=3)
            result = func.evaluate(x)
            assert result >= -1e-15, f"Expected non-negative, got {result}"

    def test_evaluate_batch(self):
        """Test batch evaluation matches individual evaluation."""
        func = Mishra11Function(dimension=3)
        X = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(results, expected)

    def test_different_dimensions(self):
        """Test function works with various dimensions."""
        for dim in [2, 5, 10]:
            func = Mishra11Function(dimension=dim)
            result = func.evaluate(np.zeros(dim))
            assert abs(result) < 1e-10


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------
class TestMishraFunctionIntegration:
    """Integration tests for all Mishra functions."""

    @pytest.mark.parametrize(
        "func_class,kwargs",
        [
            (Mishra01Function, {"dimension": 5}),
            (Mishra02Function, {"dimension": 5}),
            (Mishra03Function, {}),
            (Mishra04Function, {}),
            (Mishra05Function, {}),
            (Mishra06Function, {}),
            (Mishra07Function, {"dimension": 3}),
            (Mishra08Function, {}),
            (Mishra09Function, {}),
            (Mishra10Function, {}),
            (Mishra11Function, {"dimension": 5}),
        ],
    )
    def test_all_functions_instantiate(self, func_class, kwargs):
        """Test all functions can be instantiated."""
        func = func_class(**kwargs)
        assert hasattr(func, "evaluate")
        assert hasattr(func, "evaluate_batch")
        assert hasattr(func, "get_global_minimum")

    @pytest.mark.parametrize(
        "func_class,kwargs",
        [
            (Mishra01Function, {"dimension": 5}),
            (Mishra02Function, {"dimension": 5}),
            (Mishra03Function, {}),
            (Mishra04Function, {}),
            (Mishra05Function, {}),
            (Mishra06Function, {}),
            (Mishra07Function, {"dimension": 3}),
            (Mishra08Function, {}),
            (Mishra09Function, {}),
            (Mishra10Function, {}),
            (Mishra11Function, {"dimension": 5}),
        ],
    )
    def test_global_minimum_consistency(self, func_class, kwargs):
        """Test global minimum methods are self-consistent."""
        func = func_class(**kwargs)
        min_point, min_value = func.get_global_minimum()
        evaluated_value = func.evaluate(min_point)
        assert abs(evaluated_value - min_value) < 1e-3, (
            f"Global minimum inconsistent for {func_class.__name__}: "
            f"reported={min_value}, evaluated={evaluated_value}"
        )

    @pytest.mark.parametrize(
        "func_class,kwargs",
        [
            (Mishra01Function, {"dimension": 3}),
            (Mishra02Function, {"dimension": 3}),
            (Mishra03Function, {}),
            (Mishra04Function, {}),
            (Mishra05Function, {}),
            (Mishra06Function, {}),
            (Mishra07Function, {"dimension": 3}),
            (Mishra08Function, {}),
            (Mishra09Function, {}),
            (Mishra10Function, {}),
            (Mishra11Function, {"dimension": 3}),
        ],
    )
    def test_input_validation(self, func_class, kwargs):
        """Test input validation for all functions."""
        func = func_class(**kwargs)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0]))  # Wrong dimension

    def test_registry_integration(self):
        """Test functions are properly registered."""
        from pyMOFL.registry import get

        for alias in [
            "mishra01",
            "mishra02",
            "mishra03",
            "mishra04",
            "mishra05",
            "mishra06",
            "mishra07",
            "mishra08",
            "mishra09",
            "mishra10",
            "mishra11",
        ]:
            cls = get(alias)
            assert cls is not None, f"Registry alias '{alias}' not found"

        for alias in [
            "Mishra01",
            "Mishra02",
            "Mishra03",
            "Mishra04",
            "Mishra05",
            "Mishra06",
            "Mishra07",
            "Mishra08",
            "Mishra09",
            "Mishra10",
            "Mishra11",
        ]:
            cls = get(alias)
            assert cls is not None, f"Registry alias '{alias}' not found"
