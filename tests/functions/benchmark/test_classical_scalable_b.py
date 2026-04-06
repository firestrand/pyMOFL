"""
Tests for classical scalable benchmark functions (batch B).

Following TDD approach with comprehensive test coverage for:
- Cosine Mixture function
- Csendes (Infinity) function
- Deb01 function
- Deb03 function
- Deflected Corrugated Spring function
- Exponential function
- Keane function
- Quintic function
- Rana function
- Deceptive function
- Odd Square function
- Xin-She Yang 01 function

Tests validate mathematical correctness, bounds handling, and edge cases.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.cosine_mixture import CosineMixtureFunction
from pyMOFL.functions.benchmark.csendes import CsendesFunction
from pyMOFL.functions.benchmark.deb01 import Deb01Function
from pyMOFL.functions.benchmark.deb03 import Deb03Function
from pyMOFL.functions.benchmark.deceptive import DeceptiveFunction
from pyMOFL.functions.benchmark.deflected_corrugated_spring import DeflectedCorrugatedSpringFunction
from pyMOFL.functions.benchmark.exponential_function import ExponentialFunction
from pyMOFL.functions.benchmark.keane import KeaneFunction
from pyMOFL.functions.benchmark.odd_square import OddSquareFunction
from pyMOFL.functions.benchmark.quintic import QuinticFunction
from pyMOFL.functions.benchmark.rana import RanaFunction
from pyMOFL.functions.benchmark.xin_she_yang01 import XinSheYang01Function


class TestCosineMixtureFunction:
    """Test Cosine Mixture function: f(x) = -0.1*sum(cos(5*pi*x_i)) - sum(x_i^2)."""

    def test_initialization(self):
        func = CosineMixtureFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-1.0, -1.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [1.0, 1.0])

    def test_initialization_higher_dim(self):
        func = CosineMixtureFunction(dimension=10)
        assert func.dimension == 10

    def test_global_minimum(self):
        func = CosineMixtureFunction(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.zeros(5))
        assert min_value == pytest.approx(-0.5)

    def test_global_minimum_consistency(self):
        func = CosineMixtureFunction(dimension=3)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)

    def test_evaluate_at_origin(self):
        func = CosineMixtureFunction(dimension=2)
        x = np.array([0.0, 0.0])
        # f(0,0) = -0.1*sum(cos(0)) - 0 = -0.1*2 = -0.2
        result = func.evaluate(x)
        assert result == pytest.approx(-0.2, abs=1e-10)

    def test_known_value(self):
        func = CosineMixtureFunction(dimension=1)
        x = np.array([0.5])
        # f(0.5) = -0.1*cos(5*pi*0.5) - 0.25 = -0.1*cos(2.5*pi) - 0.25
        # cos(2.5*pi) = 0
        expected = -0.1 * np.cos(2.5 * np.pi) - 0.25
        result = func.evaluate(x)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_evaluate_batch(self):
        func = CosineMixtureFunction(dimension=2)
        X = np.array([[0.0, 0.0], [0.5, 0.5], [-0.3, 0.7]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = CosineMixtureFunction(dimension=dim)
        x = np.zeros(dim)
        result = func.evaluate(x)
        assert result == pytest.approx(-0.1 * dim, abs=1e-10)


class TestCsendesFunction:
    """Test Csendes function: f(x) = sum(x_i^6 * (2 + sin(1/x_i)))."""

    def test_initialization(self):
        func = CsendesFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-1.0, -1.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [1.0, 1.0])

    def test_global_minimum(self):
        func = CsendesFunction(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.zeros(5))
        assert min_value == 0.0

    def test_global_minimum_consistency(self):
        func = CsendesFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)

    def test_evaluate_at_origin(self):
        func = CsendesFunction(dimension=3)
        x = np.zeros(3)
        result = func.evaluate(x)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_handles_zero_components(self):
        """Test that zero components are handled gracefully."""
        func = CsendesFunction(dimension=2)
        x = np.array([0.0, 0.5])
        result = func.evaluate(x)
        # Only the second component contributes
        expected = 0.5**6 * (2.0 + np.sin(1.0 / 0.5))
        assert result == pytest.approx(expected, abs=1e-10)

    def test_known_value(self):
        func = CsendesFunction(dimension=1)
        x = np.array([0.5])
        expected = 0.5**6 * (2.0 + np.sin(2.0))
        result = func.evaluate(x)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_nonnegative(self):
        """Csendes is nonnegative since x^6 >= 0 and (2+sin(1/x)) >= 1."""
        func = CsendesFunction(dimension=3)
        rng = np.random.RandomState(42)
        for _ in range(20):
            x = rng.uniform(-1, 1, size=3)
            assert func.evaluate(x) >= -1e-15

    def test_evaluate_batch(self):
        func = CsendesFunction(dimension=2)
        X = np.array([[0.0, 0.0], [0.5, 0.5], [0.0, 0.3]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = CsendesFunction(dimension=dim)
        x = np.zeros(dim)
        result = func.evaluate(x)
        assert result == 0.0


class TestDeb01Function:
    """Test Deb01 function: f(x) = -(1/D) * sum(sin(5*pi*x_i)^6)."""

    def test_initialization(self):
        func = Deb01Function(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-1.0, -1.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [1.0, 1.0])

    def test_global_minimum(self):
        func = Deb01Function(dimension=5)
        _min_point, min_value = func.get_global_minimum()
        assert min_value == -1.0

    def test_global_minimum_consistency(self):
        func = Deb01Function(dimension=3)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)

    def test_evaluate_at_optimum(self):
        func = Deb01Function(dimension=2)
        # x_i = 0.1 => sin(5*pi*0.1) = sin(pi/2) = 1 => sin^6 = 1
        x = np.full(2, 0.1)
        result = func.evaluate(x)
        assert result == pytest.approx(-1.0, abs=1e-10)

    def test_evaluate_at_origin(self):
        func = Deb01Function(dimension=2)
        x = np.array([0.0, 0.0])
        # sin(0)^6 = 0
        result = func.evaluate(x)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        func = Deb01Function(dimension=1)
        x = np.array([0.3])
        # sin(5*pi*0.3) = sin(1.5*pi) = -1, (-1)^6 = 1
        expected = -1.0
        result = func.evaluate(x)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_evaluate_batch(self):
        func = Deb01Function(dimension=2)
        X = np.array([[0.0, 0.0], [0.1, 0.1], [0.5, 0.5]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = Deb01Function(dimension=dim)
        x = np.full(dim, 0.1)
        result = func.evaluate(x)
        assert result == pytest.approx(-1.0, abs=1e-10)


class TestDeb03Function:
    """Test Deb03 function: f(x) = -(1/D) * sum(sin(5*pi*(x_i^0.75 - 0.05))^6)."""

    def test_initialization(self):
        func = Deb03Function(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [0.0, 0.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [1.0, 1.0])

    def test_global_minimum(self):
        func = Deb03Function(dimension=5)
        _min_point, min_value = func.get_global_minimum()
        assert min_value == -1.0

    def test_global_minimum_consistency(self):
        func = Deb03Function(dimension=3)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)

    def test_known_value_at_optimum(self):
        func = Deb03Function(dimension=2)
        # At optimum: sin(5*pi*(x^0.75 - 0.05))^6 = 1
        min_point, _ = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(-1.0, abs=1e-10)

    def test_evaluate_batch(self):
        func = Deb03Function(dimension=2)
        X = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = Deb03Function(dimension=dim)
        min_point, _ = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(-1.0, abs=1e-10)


class TestDeflectedCorrugatedSpringFunction:
    """Test Deflected Corrugated Spring: f(x) = 0.1*sum((x_i-alpha)^2) - cos(K*sqrt(sum((x_i-alpha)^2)))."""

    def test_initialization(self):
        func = DeflectedCorrugatedSpringFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [0.0, 0.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_initialization_custom_alpha(self):
        func = DeflectedCorrugatedSpringFunction(dimension=2, alpha=3.0)
        np.testing.assert_array_equal(func.initialization_bounds.high, [6.0, 6.0])

    def test_global_minimum(self):
        func = DeflectedCorrugatedSpringFunction(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.full(5, 5.0))
        assert min_value == -1.0

    def test_global_minimum_consistency(self):
        func = DeflectedCorrugatedSpringFunction(dimension=3)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)

    def test_evaluate_at_optimum(self):
        func = DeflectedCorrugatedSpringFunction(dimension=2)
        x = np.array([5.0, 5.0])
        # sum = 0, f = 0 - cos(0) = -1
        result = func.evaluate(x)
        assert result == pytest.approx(-1.0, abs=1e-10)

    def test_known_value(self):
        func = DeflectedCorrugatedSpringFunction(dimension=1, alpha=5.0, K=5.0)
        x = np.array([0.0])
        # (0-5)^2 = 25, f = 0.1*25 - cos(5*5) = 2.5 - cos(25)
        expected = 2.5 - np.cos(25.0)
        result = func.evaluate(x)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_evaluate_batch(self):
        func = DeflectedCorrugatedSpringFunction(dimension=2)
        X = np.array([[5.0, 5.0], [0.0, 0.0], [3.0, 7.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = DeflectedCorrugatedSpringFunction(dimension=dim)
        x = np.full(dim, 5.0)
        result = func.evaluate(x)
        assert result == pytest.approx(-1.0, abs=1e-10)


class TestExponentialFunction:
    """Test Exponential function: f(x) = -exp(-0.5 * sum(x_i^2))."""

    def test_initialization(self):
        func = ExponentialFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-1.0, -1.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [1.0, 1.0])

    def test_global_minimum(self):
        func = ExponentialFunction(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.zeros(5))
        assert min_value == -1.0

    def test_global_minimum_consistency(self):
        func = ExponentialFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)

    def test_evaluate_at_origin(self):
        func = ExponentialFunction(dimension=3)
        x = np.zeros(3)
        result = func.evaluate(x)
        assert result == pytest.approx(-1.0, abs=1e-10)

    def test_known_value(self):
        func = ExponentialFunction(dimension=1)
        x = np.array([1.0])
        # f(1) = -exp(-0.5) ~ -0.6065
        expected = -np.exp(-0.5)
        result = func.evaluate(x)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_known_value_2d(self):
        func = ExponentialFunction(dimension=2)
        x = np.array([1.0, 1.0])
        # f = -exp(-0.5*2) = -exp(-1) ~ -0.3679
        expected = -np.exp(-1.0)
        result = func.evaluate(x)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_negative_output(self):
        """Exponential function always returns negative values."""
        func = ExponentialFunction(dimension=2)
        rng = np.random.RandomState(42)
        for _ in range(20):
            x = rng.uniform(-1, 1, size=2)
            assert func.evaluate(x) < 0.0

    def test_evaluate_batch(self):
        func = ExponentialFunction(dimension=2)
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = ExponentialFunction(dimension=dim)
        x = np.zeros(dim)
        result = func.evaluate(x)
        assert result == pytest.approx(-1.0, abs=1e-10)


class TestKeaneFunction:
    """Test Keane function."""

    def test_initialization(self):
        func = KeaneFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [0.0, 0.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_global_minimum(self):
        func = KeaneFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        assert len(min_point) == 2
        assert isinstance(min_value, float)

    def test_global_minimum_consistency(self):
        func = KeaneFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)

    def test_evaluate_at_zero_returns_zero(self):
        """Denominator is zero at origin, should return 0."""
        func = KeaneFunction(dimension=2)
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        assert result == 0.0

    def test_evaluate_returns_finite(self):
        func = KeaneFunction(dimension=2)
        x = np.array([1.0, 2.0])
        result = func.evaluate(x)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_known_value(self):
        func = KeaneFunction(dimension=2)
        x = np.array([1.0, 1.0])
        cos1 = np.cos(1.0)
        numer = np.abs(2.0 * cos1**4 - 2.0 * cos1**4)
        denom = np.sqrt(1.0 * 1.0 + 2.0 * 1.0)
        expected = -numer / denom  # numer = 0 here
        result = func.evaluate(x)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_evaluate_batch(self):
        func = KeaneFunction(dimension=2)
        X = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 5.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    def test_evaluate_batch_zero_denominator(self):
        """Batch should handle zero denominators correctly."""
        func = KeaneFunction(dimension=2)
        X = np.array([[0.0, 0.0], [1.0, 2.0]])
        results = func.evaluate_batch(X)
        assert results[0] == 0.0
        assert not np.isnan(results[1])


class TestQuinticFunction:
    """Test Quintic function: f(x) = sum(|x_i^5 - 3*x_i^4 + 4*x_i^3 + 2*x_i^2 - 10*x_i - 4|)."""

    def test_initialization(self):
        func = QuinticFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-10.0, -10.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_global_minimum(self):
        func = QuinticFunction(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.full(5, -1.0))
        assert min_value == 0.0

    def test_global_minimum_consistency(self):
        func = QuinticFunction(dimension=3)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)

    def test_evaluate_at_minus_one(self):
        func = QuinticFunction(dimension=2)
        x = np.array([-1.0, -1.0])
        # (-1)^5 - 3*(-1)^4 + 4*(-1)^3 + 2*(-1)^2 - 10*(-1) - 4
        # = -1 - 3 - 4 + 2 + 10 - 4 = 0
        result = func.evaluate(x)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_at_two(self):
        func = QuinticFunction(dimension=2)
        x = np.array([2.0, 2.0])
        # 32 - 48 + 32 + 8 - 20 - 4 = 0
        result = func.evaluate(x)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_mixed_optima(self):
        """The function should be zero at any combination of -1 and 2."""
        func = QuinticFunction(dimension=3)
        x = np.array([-1.0, 2.0, -1.0])
        result = func.evaluate(x)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_evaluate_batch(self):
        func = QuinticFunction(dimension=2)
        X = np.array([[-1.0, -1.0], [2.0, 2.0], [0.0, 0.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = QuinticFunction(dimension=dim)
        x = np.full(dim, -1.0)
        result = func.evaluate(x)
        assert result == pytest.approx(0.0, abs=1e-10)


class TestRanaFunction:
    """Test Rana function."""

    def test_initialization(self):
        func = RanaFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-500.0, -500.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [500.0, 500.0])

    def test_global_minimum(self):
        func = RanaFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        assert len(min_point) == 2
        assert isinstance(min_value, float)

    def test_global_minimum_consistency(self):
        func = RanaFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)

    def test_evaluate_returns_finite(self):
        func = RanaFunction(dimension=2)
        x = np.array([1.0, 2.0])
        result = func.evaluate(x)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_known_value_2d(self):
        func = RanaFunction(dimension=2)
        x = np.array([0.0, 0.0])
        # x_i=0, x_{i+1}=0:
        # t1 = sqrt(|0 - 0 + 1|) = 1, t2 = sqrt(|0 + 0 + 1|) = 1
        # 0 * sin(1) * cos(1) + (0+1) * cos(1) * sin(1)
        # = cos(1) * sin(1)
        expected = np.cos(1.0) * np.sin(1.0)
        result = func.evaluate(x)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_evaluate_batch(self):
        func = RanaFunction(dimension=3)
        X = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 5, 10])
    def test_multi_dimension(self, dim):
        func = RanaFunction(dimension=dim)
        x = np.zeros(dim)
        result = func.evaluate(x)
        assert isinstance(result, float)
        assert not np.isnan(result)


class TestDeceptiveFunction:
    """Test Deceptive function: f(x) = -(1/D * sum(g_i))^2."""

    def test_initialization(self):
        func = DeceptiveFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [0.0, 0.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [1.0, 1.0])

    def test_global_minimum(self):
        func = DeceptiveFunction(dimension=5)
        min_point, min_value = func.get_global_minimum()
        # alpha_i = i/6 for D=5
        expected_alpha = np.arange(1, 6, dtype=float) / 6.0
        np.testing.assert_allclose(min_point, expected_alpha, atol=1e-10)
        assert min_value == -1.0

    def test_global_minimum_consistency(self):
        func = DeceptiveFunction(dimension=3)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)

    def test_evaluate_at_optimum(self):
        func = DeceptiveFunction(dimension=2)
        # alpha = [1/3, 2/3]
        x = np.array([1.0 / 3.0, 2.0 / 3.0])
        result = func.evaluate(x)
        # At alpha_i, g_i = 1 (from the second/third region boundary)
        # Actually, at x_i = alpha_i exactly:
        # 5*alpha_i/alpha_i - 4 = 5 - 4 = 1
        # avg_g = 1, f = -1
        assert result == pytest.approx(-1.0, abs=1e-10)

    def test_evaluate_at_zero(self):
        func = DeceptiveFunction(dimension=2)
        x = np.array([0.0, 0.0])
        # alpha = [1/3, 2/3]
        # g_1(0): 0 <= 0 <= 4*(1/3)/5 = 4/15 => g_1 = -0/(1/3) + 4/5 = 4/5
        # g_2(0): 0 <= 0 <= 4*(2/3)/5 = 8/15 => g_2 = -0/(2/3) + 4/5 = 4/5
        # avg_g = 4/5, f = -(4/5)^2 = -16/25 = -0.64
        result = func.evaluate(x)
        assert result == pytest.approx(-0.64, abs=1e-10)

    def test_evaluate_at_one(self):
        func = DeceptiveFunction(dimension=2)
        x = np.array([1.0, 1.0])
        # alpha = [1/3, 2/3]
        # g_1(1): (1+4*(1/3))/5 = 7/15 < 1 <= 1 => region 4
        #   g_1 = (1-1)/(1 - 1/3) + 4/5 = 0 + 4/5 = 4/5
        # g_2(1): (1+4*(2/3))/5 = 11/15 < 1 <= 1 => region 4
        #   g_2 = (1-1)/(1 - 2/3) + 4/5 = 0 + 4/5 = 4/5
        # avg_g = 4/5, f = -16/25 = -0.64
        result = func.evaluate(x)
        assert result == pytest.approx(-0.64, abs=1e-10)

    def test_evaluate_batch(self):
        func = DeceptiveFunction(dimension=2)
        X = np.array([[0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 5, 10])
    def test_multi_dimension(self, dim):
        func = DeceptiveFunction(dimension=dim)
        min_point, _ = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(-1.0, abs=1e-10)


class TestOddSquareFunction:
    """Test Odd Square function."""

    def test_initialization(self):
        func = OddSquareFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_allclose(func.initialization_bounds.low, [-5.0 * np.pi, -5.0 * np.pi])
        np.testing.assert_allclose(func.initialization_bounds.high, [5.0 * np.pi, 5.0 * np.pi])

    def test_global_minimum(self):
        func = OddSquareFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.ones(2))
        assert isinstance(min_value, float)

    def test_global_minimum_consistency(self):
        func = OddSquareFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)

    def test_evaluate_at_b(self):
        func = OddSquareFunction(dimension=2)
        x = np.array([1.0, 1.0])
        # d=0, h=2, f = -exp(0)*cos(0)*(1 + 0.02*2/0.01) = -1*1*(1+4) = -5
        result = func.evaluate(x)
        assert result == pytest.approx(-5.0, abs=1e-10)

    def test_evaluate_returns_finite(self):
        func = OddSquareFunction(dimension=2)
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_evaluate_batch(self):
        func = OddSquareFunction(dimension=2)
        X = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 3.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 5, 10])
    def test_multi_dimension(self, dim):
        func = OddSquareFunction(dimension=dim)
        x = np.ones(dim)  # At b
        result = func.evaluate(x)
        assert isinstance(result, float)
        assert not np.isnan(result)


class TestXinSheYang01Function:
    """Test Xin-She Yang 01 function: f(x) = sum(epsilon_i * |x_i|^i)."""

    def test_initialization(self):
        func = XinSheYang01Function(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-5.0, -5.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [5.0, 5.0])

    def test_global_minimum(self):
        func = XinSheYang01Function(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.zeros(5))
        assert min_value == 0.0

    def test_global_minimum_consistency(self):
        func = XinSheYang01Function(dimension=3)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert result == pytest.approx(min_value, abs=1e-10)

    def test_evaluate_at_origin(self):
        func = XinSheYang01Function(dimension=3)
        x = np.zeros(3)
        result = func.evaluate(x)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_nonnegative(self):
        """Function is nonnegative since epsilon_i >= 0 and |x_i|^i >= 0."""
        func = XinSheYang01Function(dimension=3)
        rng = np.random.RandomState(42)
        for _ in range(20):
            x = rng.uniform(-5, 5, size=3)
            assert func.evaluate(x) >= -1e-15

    def test_reproducible_with_same_seed(self):
        """Same seed should give same random coefficients."""
        func1 = XinSheYang01Function(dimension=3, seed=42)
        func2 = XinSheYang01Function(dimension=3, seed=42)
        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) == func2.evaluate(x)

    def test_different_seed_gives_different_result(self):
        """Different seeds should give different results."""
        func1 = XinSheYang01Function(dimension=3, seed=42)
        func2 = XinSheYang01Function(dimension=3, seed=99)
        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) != func2.evaluate(x)

    def test_known_value(self):
        func = XinSheYang01Function(dimension=2, seed=42)
        x = np.array([1.0, 1.0])
        # epsilon * |1|^1 + epsilon * |1|^2
        expected = func._epsilon[0] * 1.0 + func._epsilon[1] * 1.0
        result = func.evaluate(x)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_evaluate_batch(self):
        func = XinSheYang01Function(dimension=2)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-2.0, 3.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = XinSheYang01Function(dimension=dim)
        x = np.zeros(dim)
        result = func.evaluate(x)
        assert result == 0.0


class TestClassicalScalableBIntegration:
    """Integration tests for all classical scalable functions (batch B)."""

    ALL_CLASSES = [
        CosineMixtureFunction,
        CsendesFunction,
        Deb01Function,
        Deb03Function,
        DeflectedCorrugatedSpringFunction,
        ExponentialFunction,
        KeaneFunction,
        QuinticFunction,
        RanaFunction,
        DeceptiveFunction,
        OddSquareFunction,
        XinSheYang01Function,
    ]

    @pytest.mark.parametrize("func_class", ALL_CLASSES)
    def test_all_functions_instantiate(self, func_class):
        func = func_class(dimension=2)
        assert hasattr(func, "evaluate")
        assert hasattr(func, "evaluate_batch")
        assert hasattr(func, "get_global_minimum")

    @pytest.mark.parametrize("func_class", ALL_CLASSES)
    def test_batch_evaluation_consistency(self, func_class):
        func = func_class(dimension=2)
        rng = np.random.RandomState(42)
        low = func.initialization_bounds.low
        high = func.initialization_bounds.high
        X = rng.uniform(low, high, size=(5, 2))
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("func_class", ALL_CLASSES)
    def test_input_validation(self, func_class):
        func = func_class(dimension=2)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0]))  # Wrong dimension
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension

    @pytest.mark.parametrize("func_class", ALL_CLASSES)
    def test_mathematical_consistency(self, func_class):
        func = func_class(dimension=2)
        low = func.initialization_bounds.low
        high = func.initialization_bounds.high
        test_point = (low + high) / 2.0
        result = func.evaluate(test_point)
        assert isinstance(result, (int, float))
        assert not np.isnan(result)
        assert not np.isinf(result)

    @pytest.mark.parametrize("func_class", ALL_CLASSES)
    def test_scalable_dimensions(self, func_class):
        """Scalable functions should accept any dimension."""
        for dim in [3, 5, 10]:
            func = func_class(dimension=dim)
            assert func.dimension == dim
            x = np.zeros(dim)
            result = func.evaluate(x)
            assert isinstance(result, (int, float))
