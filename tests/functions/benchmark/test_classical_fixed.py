"""
Tests for classical fixed-dimension benchmark functions.

Following TDD approach with comprehensive test coverage for:
- Beale function (2D)
- Booth function (2D)
- Bohachevsky functions 1, 2, 3 (2D)
- Bukin N.6 function (2D)
- Six-Hump Camel function (2D)
- Three-Hump Camel function (2D)
- Cross-in-Tray function (2D)
- Drop-Wave function (2D)
- Eggholder function (2D)
- Holder Table function (2D)
- Hartmann 3 function (3D)
- Hartmann 6 function (6D)
- Colville function (4D)

Tests validate mathematical correctness, bounds handling, dimension enforcement, and edge cases.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.beale import BealeFunction
from pyMOFL.functions.benchmark.bohachevsky import (
    Bohachevsky1Function,
    Bohachevsky2Function,
    Bohachevsky3Function,
)
from pyMOFL.functions.benchmark.booth import BoothFunction
from pyMOFL.functions.benchmark.bukin import Bukin6Function
from pyMOFL.functions.benchmark.camel import SixHumpCamelFunction, ThreeHumpCamelFunction
from pyMOFL.functions.benchmark.colville import ColvilleFunction
from pyMOFL.functions.benchmark.cross_in_tray import CrossInTrayFunction
from pyMOFL.functions.benchmark.drop_wave import DropWaveFunction
from pyMOFL.functions.benchmark.eggholder import EggholderFunction
from pyMOFL.functions.benchmark.hartmann import Hartmann3Function, Hartmann6Function
from pyMOFL.functions.benchmark.holder_table import HolderTableFunction


class TestBealeFunction:
    """Test Beale function: f(x) = sum_i (c_i - x1*(1-x2^i))^2."""

    def test_initialization(self):
        func = BealeFunction()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-4.5, -4.5])
        np.testing.assert_array_equal(func.initialization_bounds.high, [4.5, 4.5])

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=2"):
            BealeFunction(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = BealeFunction()
        x = np.array([3.0, 0.5])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_at_origin(self):
        func = BealeFunction()
        x = np.array([0.0, 0.0])
        # c = [1.5, 2.25, 2.625]
        # f = (1.5-0)^2 + (2.25-0)^2 + (2.625-0)^2 = 2.25 + 5.0625 + 6.890625 = 14.203125
        result = func.evaluate(x)
        expected = 1.5**2 + 2.25**2 + 2.625**2
        assert abs(result - expected) < 1e-10

    def test_global_minimum(self):
        func = BealeFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [3.0, 0.5])
        assert min_value == 0.0

    def test_global_minimum_consistency(self):
        func = BealeFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10

    def test_evaluate_batch(self):
        func = BealeFunction()
        X = np.array([[3.0, 0.5], [0.0, 0.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestBoothFunction:
    """Test Booth function: f(x) = (x1+2*x2-7)^2 + (2*x1+x2-5)^2."""

    def test_initialization(self):
        func = BoothFunction()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-10.0, -10.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=2"):
            BoothFunction(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = BoothFunction()
        x = np.array([1.0, 3.0])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_at_origin(self):
        func = BoothFunction()
        x = np.array([0.0, 0.0])
        # (0+0-7)^2 + (0+0-5)^2 = 49 + 25 = 74
        result = func.evaluate(x)
        assert abs(result - 74.0) < 1e-10

    def test_global_minimum(self):
        func = BoothFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [1.0, 3.0])
        assert min_value == 0.0

    def test_global_minimum_consistency(self):
        func = BoothFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10

    def test_evaluate_batch(self):
        func = BoothFunction()
        X = np.array([[1.0, 3.0], [0.0, 0.0], [2.0, 2.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestBohachevsky1Function:
    """Test Bohachevsky1: f(x) = x1^2 + 2*x2^2 - 0.3*cos(3*pi*x1) - 0.4*cos(4*pi*x2) + 0.7."""

    def test_initialization(self):
        func = Bohachevsky1Function()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-100.0, -100.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [100.0, 100.0])

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=2"):
            Bohachevsky1Function(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = Bohachevsky1Function()
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        # 0 + 0 - 0.3*cos(0) - 0.4*cos(0) + 0.7 = -0.3 - 0.4 + 0.7 = 0
        assert abs(result) < 1e-10

    def test_evaluate_known_value(self):
        func = Bohachevsky1Function()
        x = np.array([1.0, 1.0])
        expected = 1.0 + 2.0 - 0.3 * np.cos(3 * np.pi) - 0.4 * np.cos(4 * np.pi) + 0.7
        result = func.evaluate(x)
        assert abs(result - expected) < 1e-10

    def test_global_minimum(self):
        func = Bohachevsky1Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, [0.0, 0.0])
        assert min_value == 0.0

    def test_evaluate_batch(self):
        func = Bohachevsky1Function()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestBohachevsky2Function:
    """Test Bohachevsky2: f(x) = x1^2 + 2*x2^2 - 0.3*cos(3*pi*x1)*cos(4*pi*x2) + 0.3."""

    def test_initialization(self):
        func = Bohachevsky2Function()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=2"):
            Bohachevsky2Function(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = Bohachevsky2Function()
        x = np.array([0.0, 0.0])
        # 0 + 0 - 0.3*cos(0)*cos(0) + 0.3 = -0.3 + 0.3 = 0
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_known_value(self):
        func = Bohachevsky2Function()
        x = np.array([1.0, 1.0])
        expected = 1.0 + 2.0 - 0.3 * np.cos(3 * np.pi) * np.cos(4 * np.pi) + 0.3
        result = func.evaluate(x)
        assert abs(result - expected) < 1e-10

    def test_global_minimum(self):
        func = Bohachevsky2Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, [0.0, 0.0])
        assert min_value == 0.0

    def test_evaluate_batch(self):
        func = Bohachevsky2Function()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestBohachevsky3Function:
    """Test Bohachevsky3: f(x) = x1^2 + 2*x2^2 - 0.3*cos(3*pi*x1 + 4*pi*x2) + 0.3."""

    def test_initialization(self):
        func = Bohachevsky3Function()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=2"):
            Bohachevsky3Function(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = Bohachevsky3Function()
        x = np.array([0.0, 0.0])
        # 0 + 0 - 0.3*cos(0) + 0.3 = -0.3 + 0.3 = 0
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_known_value(self):
        func = Bohachevsky3Function()
        x = np.array([1.0, 1.0])
        expected = 1.0 + 2.0 - 0.3 * np.cos(3 * np.pi + 4 * np.pi) + 0.3
        result = func.evaluate(x)
        assert abs(result - expected) < 1e-10

    def test_global_minimum(self):
        func = Bohachevsky3Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, [0.0, 0.0])
        assert min_value == 0.0

    def test_evaluate_batch(self):
        func = Bohachevsky3Function()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestBukin6Function:
    """Test Bukin N.6: f(x) = 100*sqrt(|x2-0.01*x1^2|) + 0.01*|x1+10|."""

    def test_initialization(self):
        func = Bukin6Function()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-15.0, -3.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [-5.0, 3.0])

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=2"):
            Bukin6Function(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = Bukin6Function()
        x = np.array([-10.0, 1.0])
        result = func.evaluate(x)
        # 100*sqrt(|1 - 0.01*100|) + 0.01*|0| = 100*sqrt(0) + 0 = 0
        assert abs(result) < 1e-10

    def test_evaluate_known_value(self):
        func = Bukin6Function()
        x = np.array([-10.0, 0.0])
        # 100*sqrt(|0 - 0.01*100|) + 0.01*|0| = 100*sqrt(1) + 0 = 100
        result = func.evaluate(x)
        assert abs(result - 100.0) < 1e-10

    def test_global_minimum(self):
        func = Bukin6Function()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_almost_equal(min_point, [-10.0, 1.0])
        assert min_value == 0.0

    def test_global_minimum_consistency(self):
        func = Bukin6Function()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10

    def test_evaluate_batch(self):
        func = Bukin6Function()
        X = np.array([[-10.0, 1.0], [-10.0, 0.0], [-8.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestSixHumpCamelFunction:
    """Test Six-Hump Camel: f(x) = (4-2.1*x1^2+x1^4/3)*x1^2 + x1*x2 + (-4+4*x2^2)*x2^2."""

    def test_initialization(self):
        func = SixHumpCamelFunction()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-5.0, -5.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [5.0, 5.0])

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=2"):
            SixHumpCamelFunction(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = SixHumpCamelFunction()
        # Two symmetric global minima
        x1 = np.array([0.0898, -0.7126])
        x2 = np.array([-0.0898, 0.7126])
        r1 = func.evaluate(x1)
        r2 = func.evaluate(x2)
        assert abs(r1 - (-1.0316)) < 0.001
        assert abs(r2 - (-1.0316)) < 0.001

    def test_evaluate_at_origin(self):
        func = SixHumpCamelFunction()
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_global_minimum(self):
        func = SixHumpCamelFunction()
        _min_point, min_value = func.get_global_minimum()
        assert abs(min_value - (-1.0316)) < 0.001

    def test_global_minimum_consistency(self):
        func = SixHumpCamelFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-4

    def test_evaluate_batch(self):
        func = SixHumpCamelFunction()
        X = np.array([[0.0, 0.0], [0.0898, -0.7126], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestThreeHumpCamelFunction:
    """Test Three-Hump Camel: f(x) = 2*x1^2 - 1.05*x1^4 + x1^6/6 + x1*x2 + x2^2."""

    def test_initialization(self):
        func = ThreeHumpCamelFunction()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-5.0, -5.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [5.0, 5.0])

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=2"):
            ThreeHumpCamelFunction(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = ThreeHumpCamelFunction()
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_known_value(self):
        func = ThreeHumpCamelFunction()
        x = np.array([1.0, 1.0])
        # 2 - 1.05 + 1/6 + 1 + 1 = 2.11667
        expected = 2.0 - 1.05 + 1.0 / 6.0 + 1.0 + 1.0
        result = func.evaluate(x)
        assert abs(result - expected) < 1e-10

    def test_global_minimum(self):
        func = ThreeHumpCamelFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, [0.0, 0.0])
        assert min_value == 0.0

    def test_evaluate_batch(self):
        func = ThreeHumpCamelFunction()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestCrossInTrayFunction:
    """Test Cross-in-Tray: f(x) = -0.0001*(|sin(x1)*sin(x2)*exp(|100-sqrt(x1^2+x2^2)/pi|)|+1)^0.1."""

    def test_initialization(self):
        func = CrossInTrayFunction()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-10.0, -10.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=2"):
            CrossInTrayFunction(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = CrossInTrayFunction()
        # Four symmetric global minima
        x = np.array([1.3491, 1.3491])
        result = func.evaluate(x)
        assert abs(result - (-2.06261)) < 0.001

    def test_evaluate_at_origin(self):
        func = CrossInTrayFunction()
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        # sin(0)*sin(0) = 0, so inner |...| = 0, (0+1)^0.1 = 1, f = -0.0001
        assert abs(result - (-0.0001)) < 1e-8

    def test_global_minimum(self):
        func = CrossInTrayFunction()
        _min_point, min_value = func.get_global_minimum()
        assert abs(min_value - (-2.06261)) < 0.001

    def test_global_minimum_consistency(self):
        func = CrossInTrayFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-4

    def test_evaluate_batch(self):
        func = CrossInTrayFunction()
        X = np.array([[0.0, 0.0], [1.3491, 1.3491], [5.0, 5.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestDropWaveFunction:
    """Test Drop-Wave: f(x) = -(1+cos(12*sqrt(x1^2+x2^2))) / (0.5*(x1^2+x2^2)+2)."""

    def test_initialization(self):
        func = DropWaveFunction()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-5.12, -5.12])
        np.testing.assert_array_equal(func.initialization_bounds.high, [5.12, 5.12])

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=2"):
            DropWaveFunction(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = DropWaveFunction()
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        # -(1+cos(0)) / (0+2) = -2/2 = -1
        assert abs(result - (-1.0)) < 1e-10

    def test_global_minimum(self):
        func = DropWaveFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, [0.0, 0.0])
        assert min_value == -1.0

    def test_global_minimum_consistency(self):
        func = DropWaveFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10

    def test_evaluate_batch(self):
        func = DropWaveFunction()
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.5]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestEggholderFunction:
    """Test Eggholder: f(x) = -(x2+47)*sin(sqrt(|x2+x1/2+47|)) - x1*sin(sqrt(|x1-(x2+47)|))."""

    def test_initialization(self):
        func = EggholderFunction()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-512.0, -512.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [512.0, 512.0])

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=2"):
            EggholderFunction(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = EggholderFunction()
        x = np.array([512.0, 404.2319])
        result = func.evaluate(x)
        assert abs(result - (-959.6407)) < 0.1

    def test_evaluate_at_origin(self):
        func = EggholderFunction()
        x = np.array([0.0, 0.0])
        result = func.evaluate(x)
        expected = -(0 + 47) * np.sin(np.sqrt(abs(0 + 0 / 2 + 47))) - 0 * np.sin(
            np.sqrt(abs(0 - (0 + 47)))
        )
        assert abs(result - expected) < 1e-10

    def test_global_minimum(self):
        func = EggholderFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_allclose(min_point, [512.0, 404.2319], atol=0.01)
        assert abs(min_value - (-959.6407)) < 0.1

    def test_global_minimum_consistency(self):
        func = EggholderFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-4

    def test_evaluate_batch(self):
        func = EggholderFunction()
        X = np.array([[0.0, 0.0], [512.0, 404.2319], [100.0, 100.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestHolderTableFunction:
    """Test Holder Table: f(x) = -|sin(x1)*cos(x2)*exp(|1-sqrt(x1^2+x2^2)/pi|)|."""

    def test_initialization(self):
        func = HolderTableFunction()
        assert func.dimension == 2
        np.testing.assert_array_equal(func.initialization_bounds.low, [-10.0, -10.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0, 10.0])

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=2"):
            HolderTableFunction(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = HolderTableFunction()
        x = np.array([8.05502, 9.66459])
        result = func.evaluate(x)
        assert abs(result - (-19.2085)) < 0.01

    def test_evaluate_at_origin(self):
        func = HolderTableFunction()
        x = np.array([0.0, 0.0])
        # sin(0)*cos(0)*exp(...) = 0
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_global_minimum(self):
        func = HolderTableFunction()
        _min_point, min_value = func.get_global_minimum()
        assert abs(min_value - (-19.2085)) < 0.01

    def test_global_minimum_consistency(self):
        func = HolderTableFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-4

    def test_evaluate_batch(self):
        func = HolderTableFunction()
        X = np.array([[0.0, 0.0], [8.05502, 9.66459], [5.0, 5.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestHartmann3Function:
    """Test Hartmann 3D function."""

    def test_initialization(self):
        func = Hartmann3Function()
        assert func.dimension == 3
        np.testing.assert_array_equal(func.initialization_bounds.low, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(func.initialization_bounds.high, [1.0, 1.0, 1.0])

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=3"):
            Hartmann3Function(dimension=2)

    def test_evaluate_at_global_minimum(self):
        func = Hartmann3Function()
        x = np.array([0.114614, 0.555649, 0.852547])
        result = func.evaluate(x)
        assert abs(result - (-3.8628)) < 0.01

    def test_global_minimum(self):
        func = Hartmann3Function()
        _min_point, min_value = func.get_global_minimum()
        assert abs(min_value - (-3.8628)) < 0.01

    def test_global_minimum_consistency(self):
        func = Hartmann3Function()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-6

    def test_evaluate_batch(self):
        func = Hartmann3Function()
        X = np.array([[0.1, 0.5, 0.8], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestHartmann6Function:
    """Test Hartmann 6D function."""

    def test_initialization(self):
        func = Hartmann6Function()
        assert func.dimension == 6
        np.testing.assert_array_equal(func.initialization_bounds.low, np.zeros(6))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.ones(6))

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=6"):
            Hartmann6Function(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = Hartmann6Function()
        x = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        result = func.evaluate(x)
        assert abs(result - (-3.3224)) < 0.01

    def test_global_minimum(self):
        func = Hartmann6Function()
        _min_point, min_value = func.get_global_minimum()
        assert abs(min_value - (-3.3224)) < 0.01

    def test_global_minimum_consistency(self):
        func = Hartmann6Function()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-6

    def test_evaluate_batch(self):
        func = Hartmann6Function()
        X = np.array([[0.5] * 6, [0.2, 0.15, 0.47, 0.27, 0.31, 0.65]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestColvilleFunction:
    """Test Colville function (4D)."""

    def test_initialization(self):
        func = ColvilleFunction()
        assert func.dimension == 4
        np.testing.assert_array_equal(func.initialization_bounds.low, [-10.0] * 4)
        np.testing.assert_array_equal(func.initialization_bounds.high, [10.0] * 4)

    def test_dimension_validation(self):
        with pytest.raises(ValueError, match="dimension=4"):
            ColvilleFunction(dimension=3)

    def test_evaluate_at_global_minimum(self):
        func = ColvilleFunction()
        x = np.array([1.0, 1.0, 1.0, 1.0])
        result = func.evaluate(x)
        assert abs(result) < 1e-10

    def test_evaluate_at_origin(self):
        func = ColvilleFunction()
        x = np.array([0.0, 0.0, 0.0, 0.0])
        # At origin: 100*(0-0)^2 + (1-0)^2 + 90*(0-0)^2 + (1-0)^2 + 10.1*((0-1)^2+(0-1)^2) + 19.8*(0-1)*(0-1)
        # = 0 + 1 + 0 + 1 + 10.1*(1+1) + 19.8*1 = 1 + 1 + 20.2 + 19.8 = 42.0
        result = func.evaluate(x)
        expected = 42.0
        assert abs(result - expected) < 1e-10

    def test_global_minimum(self):
        func = ColvilleFunction()
        min_point, min_value = func.get_global_minimum()
        np.testing.assert_array_equal(min_point, [1.0, 1.0, 1.0, 1.0])
        assert min_value == 0.0

    def test_global_minimum_consistency(self):
        func = ColvilleFunction()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-10

    def test_evaluate_batch(self):
        func = ColvilleFunction()
        X = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)


class TestClassicalFixedIntegration:
    """Integration tests for all classical fixed-dimension functions."""

    ALL_CLASSES_WITH_DIM = [
        (BealeFunction, 2),
        (BoothFunction, 2),
        (Bohachevsky1Function, 2),
        (Bohachevsky2Function, 2),
        (Bohachevsky3Function, 2),
        (Bukin6Function, 2),
        (SixHumpCamelFunction, 2),
        (ThreeHumpCamelFunction, 2),
        (CrossInTrayFunction, 2),
        (DropWaveFunction, 2),
        (EggholderFunction, 2),
        (HolderTableFunction, 2),
        (Hartmann3Function, 3),
        (Hartmann6Function, 6),
        (ColvilleFunction, 4),
    ]

    @pytest.mark.parametrize("func_class,dim", ALL_CLASSES_WITH_DIM)
    def test_all_functions_instantiate(self, func_class, dim):
        func = func_class()
        assert func.dimension == dim
        assert hasattr(func, "evaluate")
        assert hasattr(func, "evaluate_batch")
        assert hasattr(func, "get_global_minimum")

    @pytest.mark.parametrize("func_class,dim", ALL_CLASSES_WITH_DIM)
    def test_batch_evaluation_consistency(self, func_class, dim):
        func = func_class()
        rng = np.random.RandomState(42)
        low = func.initialization_bounds.low
        high = func.initialization_bounds.high
        X = rng.uniform(low, high, size=(5, dim))
        results = func.evaluate_batch(X)
        expected = [func.evaluate(x) for x in X]
        np.testing.assert_allclose(results, expected, atol=1e-10)

    @pytest.mark.parametrize("func_class,dim", ALL_CLASSES_WITH_DIM)
    def test_global_minimum_consistency(self, func_class, dim):
        func = func_class()
        min_point, min_value = func.get_global_minimum()
        result = func.evaluate(min_point)
        assert abs(result - min_value) < 1e-3, (
            f"Global minimum inconsistent for {func_class.__name__}: "
            f"evaluate({min_point}) = {result}, expected {min_value}"
        )

    @pytest.mark.parametrize("func_class,dim", ALL_CLASSES_WITH_DIM)
    def test_input_validation(self, func_class, dim):
        func = func_class()
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0]))  # Wrong dimension
