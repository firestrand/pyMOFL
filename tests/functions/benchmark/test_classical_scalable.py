"""
Tests for classical scalable benchmark functions.

Following TDD approach with comprehensive test coverage for:
- Styblinski-Tang function
- Salomon function
- Michalewicz function
- Langermann function
- Brown function
- Chung-Reynolds function
- Qing function
- Quartic (De Jong 4) function

Tests validate mathematical correctness, bounds handling, and edge cases.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.brown import BrownFunction
from pyMOFL.functions.benchmark.chung_reynolds import ChungReynoldsFunction
from pyMOFL.functions.benchmark.langermann import LangermannFunction
from pyMOFL.functions.benchmark.michalewicz import MichalewiczFunction
from pyMOFL.functions.benchmark.qing import QingFunction
from pyMOFL.functions.benchmark.quartic import QuarticFunction
from pyMOFL.functions.benchmark.salomon import SalomonFunction
from pyMOFL.functions.benchmark.styblinski_tang import StyblinskiTangFunction


class TestStyblinskiTangFunction:
    """Test Styblinski-Tang function: f(x) = 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i)."""

    def test_initialization(self):
        func = StyblinskiTangFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-5.0, -5.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [5.0, 5.0])

    def test_initialization_higher_dim(self):
        func = StyblinskiTangFunction(dimension=10)
        assert func.dimension == 10

    def test_evaluate_at_global_minimum(self):
        func = StyblinskiTangFunction(dimension=2)
        x_star = np.full(2, -2.903534)
        result = func.evaluate(x_star)
        expected = -39.16617 * 2
        assert abs(result - expected) < 0.01

    def test_evaluate_at_origin(self):
        func = StyblinskiTangFunction(dimension=2)
        x = np.array([0.0, 0.0])
        # f(0) = 0.5 * sum(0 - 0 + 0) = 0
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_known_value(self):
        func = StyblinskiTangFunction(dimension=1)
        x = np.array([1.0])
        # f(1) = 0.5 * (1 - 16 + 5) = 0.5 * (-10) = -5
        result = func.evaluate(x)
        assert abs(result - (-5.0)) < 1e-10

    def test_global_minimum(self):
        func = StyblinskiTangFunction(dimension=3)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_allclose(min_point, np.full(3, -2.903534), atol=1e-4)
        assert abs(min_value - (-39.16617 * 3)) < 0.1

    def test_global_minimum_consistency(self):
        func = StyblinskiTangFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-4

    def test_evaluate_batch(self):
        func = StyblinskiTangFunction(dimension=2)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 2.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = StyblinskiTangFunction(dimension=dim)
        x = np.zeros(dim)
        result = func.evaluate(x)
        assert result == 0.0


class TestSalomonFunction:
    """Test Salomon function: f(x) = 1 - cos(2*pi*||x||) + 0.1*||x||."""

    def test_initialization(self):
        func = SalomonFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-100.0, -100.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [100.0, 100.0])

    def test_evaluate_at_global_minimum(self):
        func = SalomonFunction(dimension=2)
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        # f(0) = 1 - cos(0) + 0 = 1 - 1 = 0
        assert abs(result) < 1e-10

    def test_evaluate_known_value(self):
        func = SalomonFunction(dimension=1)
        x = np.array([1.0])
        # ||x|| = 1, f = 1 - cos(2*pi) + 0.1 = 1 - 1 + 0.1 = 0.1
        result = func.evaluate(x)
        assert abs(result - 0.1) < 1e-10

    def test_global_minimum(self):
        func = SalomonFunction(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.zeros(5))
        assert min_value == 0.0

    def test_global_minimum_consistency(self):
        func = SalomonFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10

    def test_evaluate_batch(self):
        func = SalomonFunction(dimension=2)
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = SalomonFunction(dimension=dim)
        x = np.zeros(dim)
        result = func.evaluate(x)
        assert result == 0.0

    def test_symmetry(self):
        """Salomon is radially symmetric around origin."""
        func = SalomonFunction(dimension=2)
        x1 = np.array([1.0, 0.0])
        x2 = np.array([0.0, 1.0])
        assert abs(func.evaluate(x1) - func.evaluate(x2)) < 1e-10


class TestMichalewiczFunction:
    """Test Michalewicz function: f(x) = -sum(sin(x_i)*sin(i*x_i^2/pi)^(2m)), m=10."""

    def test_initialization(self):
        func = MichalewiczFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [0.0, 0.0])
        np.testing.assert_array_almost_equal(func.initialization_bounds.high, [np.pi, np.pi])

    def test_evaluate_at_origin(self):
        func = MichalewiczFunction(dimension=2)
        x = np.array([0.0, 0.0])
        # sin(0) = 0, so all terms = 0
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_known_2d(self):
        """Known approximate optimum for 2D: f* ~ -1.8013."""
        func = MichalewiczFunction(dimension=2)
        # The 2D optimum is approximately at (2.20, 1.57)
        x = np.array([2.20, 1.57])
        result = func.evaluate(x)
        assert result < -1.7  # Should be near -1.8013

    def test_global_minimum_2d(self):
        func = MichalewiczFunction(dimension=2)
        _min_point, min_value = func.get_global_minimum()
        assert min_value < -1.7  # Known ~ -1.8013

    def test_global_minimum_consistency(self):
        func = MichalewiczFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-6

    def test_evaluate_batch(self):
        func = MichalewiczFunction(dimension=2)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.5]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = MichalewiczFunction(dimension=dim)
        x = np.zeros(dim)
        result = func.evaluate(x)
        assert result == 0.0

    def test_m_parameter(self):
        """Test that m parameter controls steepness (default m=10)."""
        func = MichalewiczFunction(dimension=2, m=10)
        x = np.array([1.0, 1.0])
        result = func.evaluate(x)
        assert isinstance(result, float)
        assert not np.isnan(result)


class TestLangermannFunction:
    """Test Langermann function."""

    def test_initialization(self):
        func = LangermannFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [0.0, 0.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_evaluate_returns_float(self):
        func = LangermannFunction(dimension=2)
        x = np.array([1.0, 1.0])
        result = func.evaluate(x)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_evaluate_at_known_point(self):
        """Test at origin — should produce a non-trivial value."""
        func = LangermannFunction(dimension=2)
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        assert isinstance(result, float)

    def test_global_minimum(self):
        func = LangermannFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        assert len(min_point) == 2
        assert isinstance(min_value, float)

    def test_global_minimum_consistency(self):
        func = LangermannFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-6

    def test_evaluate_batch(self):
        func = LangermannFunction(dimension=2)
        X = np.array([[1.0, 1.0], [5.0, 5.0], [8.0, 2.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10])
    def test_multi_dimension(self, dim):
        func = LangermannFunction(dimension=dim)
        x = np.ones(dim)
        result = func.evaluate(x)
        assert isinstance(result, float)

    def test_high_dim_zero_padding(self):
        """For dim > 2, A matrix columns beyond 2nd are zero-padded.

        This means the first two coordinates use the standard A coefficients,
        while additional dimensions contribute only x_i^2 distance terms.
        """
        func_2d = LangermannFunction(dimension=2)
        func_5d = LangermannFunction(dimension=5)

        # Verify A matrix shape and zero-padding
        assert func_5d._A.shape == (5, 5)
        np.testing.assert_array_equal(func_5d._A[:, 2:], np.zeros((5, 3)))
        np.testing.assert_array_equal(func_5d._A[:, :2], func_2d._A)

        # Evaluating with zeros in extra dims should match the 2D value
        x_2d = np.array([1.0, 2.0])
        x_5d = np.array([1.0, 2.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(func_5d.evaluate(x_5d), func_2d.evaluate(x_2d), atol=1e-10)


class TestBrownFunction:
    """Test Brown function: f(x) = sum((x_i^2)^(x_{i+1}^2+1) + (x_{i+1}^2)^(x_i^2+1))."""

    def test_initialization(self):
        func = BrownFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-1.0, -1.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [4.0, 4.0])

    def test_evaluate_at_global_minimum(self):
        func = BrownFunction(dimension=2)
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_known_value(self):
        func = BrownFunction(dimension=2)
        x = np.array([1.0, 1.0])
        # (1^2)^(1^2+1) + (1^2)^(1^2+1) = 1^2 + 1^2 = 2
        result = func.evaluate(x)
        assert abs(result - 2.0) < 1e-10

    def test_global_minimum(self):
        func = BrownFunction(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.zeros(5))
        assert min_value == 0.0

    def test_global_minimum_consistency(self):
        func = BrownFunction(dimension=3)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10

    def test_evaluate_batch(self):
        func = BrownFunction(dimension=2)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = BrownFunction(dimension=dim)
        x = np.zeros(dim)
        result = func.evaluate(x)
        assert result == 0.0


class TestChungReynoldsFunction:
    """Test Chung-Reynolds function: f(x) = (sum(x_i^2))^2."""

    def test_initialization(self):
        func = ChungReynoldsFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-100.0, -100.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [100.0, 100.0])

    def test_evaluate_at_global_minimum(self):
        func = ChungReynoldsFunction(dimension=2)
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_known_value(self):
        func = ChungReynoldsFunction(dimension=2)
        x = np.array([1.0, 1.0])
        # (1+1)^2 = 4
        result = func.evaluate(x)
        assert abs(result - 4.0) < 1e-10

    def test_evaluate_known_value_2(self):
        func = ChungReynoldsFunction(dimension=3)
        x = np.array([1.0, 2.0, 3.0])
        # (1+4+9)^2 = 196
        result = func.evaluate(x)
        assert abs(result - 196.0) < 1e-10

    def test_global_minimum(self):
        func = ChungReynoldsFunction(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.zeros(5))
        assert min_value == 0.0

    def test_global_minimum_consistency(self):
        func = ChungReynoldsFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10

    def test_evaluate_batch(self):
        func = ChungReynoldsFunction(dimension=2)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = ChungReynoldsFunction(dimension=dim)
        x = np.zeros(dim)
        result = func.evaluate(x)
        assert result == 0.0


class TestQingFunction:
    """Test Qing function: f(x) = sum((x_i^2 - i)^2)."""

    def test_initialization(self):
        func = QingFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-500.0, -500.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [500.0, 500.0])

    def test_evaluate_at_global_minimum(self):
        func = QingFunction(dimension=2)
        # x_i = sqrt(i), so x = [1, sqrt(2)]
        x = np.array([1.0, np.sqrt(2.0)])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_at_global_minimum_3d(self):
        func = QingFunction(dimension=3)
        x = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_at_origin(self):
        func = QingFunction(dimension=2)
        x = np.array([0.0, 0.0])
        # (0-1)^2 + (0-2)^2 = 1 + 4 = 5
        result = func.evaluate(x)
        assert abs(result - 5.0) < 1e-10

    def test_evaluate_known_value(self):
        func = QingFunction(dimension=1)
        x = np.array([2.0])
        # (4 - 1)^2 = 9
        result = func.evaluate(x)
        assert abs(result - 9.0) < 1e-10

    def test_global_minimum(self):
        func = QingFunction(dimension=3)
        min_point, min_value = func.get_global_minimum()
        expected_point = np.array([1.0, np.sqrt(2.0), np.sqrt(3.0)])
        np.testing.assert_allclose(min_point, expected_point, atol=1e-10)
        assert min_value == 0.0

    def test_global_minimum_consistency(self):
        func = QingFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10

    def test_evaluate_batch(self):
        func = QingFunction(dimension=2)
        X = np.array([[0.0, 0.0], [1.0, np.sqrt(2.0)], [2.0, 2.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = QingFunction(dimension=dim)
        x = np.sqrt(np.arange(1, dim + 1, dtype=float))
        result = func.evaluate(x)
        assert abs(result) < 1e-8


class TestQuarticFunction:
    """Test Quartic (De Jong 4) function: f(x) = sum(i * x_i^4), no noise."""

    def test_initialization(self):
        func = QuarticFunction(dimension=2)
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-1.28, -1.28])
        np.testing.assert_array_equal(func.initialization_bounds.high, [1.28, 1.28])

    def test_evaluate_at_global_minimum(self):
        func = QuarticFunction(dimension=2)
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_known_value(self):
        func = QuarticFunction(dimension=2)
        x = np.array([1.0, 1.0])
        # 1*1^4 + 2*1^4 = 1 + 2 = 3
        result = func.evaluate(x)
        assert abs(result - 3.0) < 1e-10

    def test_evaluate_known_value_2(self):
        func = QuarticFunction(dimension=3)
        x = np.array([1.0, 1.0, 1.0])
        # 1*1 + 2*1 + 3*1 = 6
        result = func.evaluate(x)
        assert abs(result - 6.0) < 1e-10

    def test_global_minimum(self):
        func = QuarticFunction(dimension=5)
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, np.zeros(5))
        assert min_value == 0.0

    def test_global_minimum_consistency(self):
        func = QuarticFunction(dimension=2)
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10

    def test_evaluate_batch(self):
        func = QuarticFunction(dimension=2)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, -0.5]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 10, 30])
    def test_multi_dimension(self, dim):
        func = QuarticFunction(dimension=dim)
        x = np.zeros(dim)
        result = func.evaluate(x)
        assert result == 0.0


class TestClassicalScalableIntegration:
    """Integration tests for all classical scalable functions."""

    ALL_CLASSES = [
        StyblinskiTangFunction,
        SalomonFunction,
        MichalewiczFunction,
        LangermannFunction,
        BrownFunction,
        ChungReynoldsFunction,
        QingFunction,
        QuarticFunction,
    ]

    @pytest.mark.parametrize(
        "func_class",
        ALL_CLASSES,
    )
    def test_all_functions_instantiate(self, func_class):
        func = func_class(dimension=2)
        assert hasattr(func, "evaluate")
        assert hasattr(func, "evaluate_batch")
        assert hasattr(func, "get_global_minimum")

    @pytest.mark.parametrize(
        "func_class",
        ALL_CLASSES,
    )
    def test_batch_evaluation_consistency(self, func_class):
        func = func_class(dimension=2)
        rng = np.random.RandomState(42)
        low = func.initialization_bounds.low
        high = func.initialization_bounds.high
        X = rng.uniform(low, high, size=(5, 2))
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize(
        "func_class",
        ALL_CLASSES,
    )
    def test_input_validation(self, func_class):
        func = func_class(dimension=2)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0]))  # Wrong dimension
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension

    @pytest.mark.parametrize(
        "func_class",
        ALL_CLASSES,
    )
    def test_mathematical_consistency(self, func_class):
        func = func_class(dimension=2)
        low = func.initialization_bounds.low
        high = func.initialization_bounds.high
        test_point = (low + high) / 2.0
        result = func.evaluate(test_point)
        assert isinstance(result, (int, float))
        assert not np.isnan(result)
        assert not np.isinf(result)
