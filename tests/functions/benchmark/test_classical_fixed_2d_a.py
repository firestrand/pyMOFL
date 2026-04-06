"""
Tests for classical fixed 2D optimization functions (Batch A).

Tests validate mathematical correctness, bounds handling, and edge cases
for: Adjiman, BartelsConn, Bird, Brent, Chichinadze, EggCrate,
FreudensteinRoth, Giunta, Hansen, Hosaki, JennrichSampson, Parsopoulos.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.adjiman import AdjimanFunction
from pyMOFL.functions.benchmark.bartels_conn import BartelsConnFunction
from pyMOFL.functions.benchmark.bird import BirdFunction
from pyMOFL.functions.benchmark.brent import BrentFunction
from pyMOFL.functions.benchmark.chichinadze import ChichinadzeFunction
from pyMOFL.functions.benchmark.egg_crate import EggCrateFunction
from pyMOFL.functions.benchmark.freudenstein_roth import FreudensteinRothFunction
from pyMOFL.functions.benchmark.giunta import GiuntaFunction
from pyMOFL.functions.benchmark.hansen import HansenFunction
from pyMOFL.functions.benchmark.hosaki import HosakiFunction
from pyMOFL.functions.benchmark.jennrich_sampson import JennrichSampsonFunction
from pyMOFL.functions.benchmark.parsopoulos import ParsopoulosFunction


class TestAdjimanFunction:
    def test_initialization(self):
        func = AdjimanFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            AdjimanFunction(dimension=3)

    def test_global_minimum(self):
        func = AdjimanFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-4

    def test_evaluate_at_origin(self):
        func = AdjimanFunction()
        # f(0,0) = cos(0)*sin(0) - 0/(0+1) = 0
        result = func.evaluate(np.array([0.0, 0.0]))
        assert abs(result) < 1e-10

    def test_known_value(self):
        func = AdjimanFunction()
        _point, value = func.get_global_minimum()
        assert value < -2.0  # Known minimum is approximately -2.02181

    def test_evaluate_batch(self):
        func = AdjimanFunction()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)
        assert abs(results[0]) < 1e-10


class TestBartelsConnFunction:
    def test_initialization(self):
        func = BartelsConnFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            BartelsConnFunction(dimension=3)

    def test_global_minimum(self):
        func = BartelsConnFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-10

    def test_at_origin(self):
        func = BartelsConnFunction()
        # f(0,0) = |0+0+0| + |sin(0)| + |cos(0)| = 0 + 0 + 1 = 1
        result = func.evaluate(np.array([0.0, 0.0]))
        assert abs(result - 1.0) < 1e-10

    def test_evaluate_batch(self):
        func = BartelsConnFunction()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)
        assert abs(results[0] - 1.0) < 1e-10


class TestBirdFunction:
    def test_initialization(self):
        func = BirdFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            BirdFunction(dimension=3)

    def test_global_minimum(self):
        func = BirdFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-2

    def test_minimum_value(self):
        func = BirdFunction()
        _, value = func.get_global_minimum()
        assert abs(value - (-106.764537)) < 0.01

    def test_evaluate_batch(self):
        func = BirdFunction()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestBrentFunction:
    def test_initialization(self):
        func = BrentFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            BrentFunction(dimension=3)

    def test_global_minimum(self):
        func = BrentFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-6

    def test_at_minimum(self):
        func = BrentFunction()
        # f(-10,-10) = 0 + 0 + e^(-200) ≈ 0
        result = func.evaluate(np.array([-10.0, -10.0]))
        assert abs(result) < 1e-6

    def test_evaluate_batch(self):
        func = BrentFunction()
        X = np.array([[-10.0, -10.0], [0.0, 0.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)
        assert abs(results[0]) < 1e-6


class TestChichinadzeFunction:
    def test_initialization(self):
        func = ChichinadzeFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            ChichinadzeFunction(dimension=3)

    def test_global_minimum(self):
        func = ChichinadzeFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 0.01

    def test_minimum_value(self):
        func = ChichinadzeFunction()
        _, value = func.get_global_minimum()
        assert value < -42.0

    def test_evaluate_batch(self):
        func = ChichinadzeFunction()
        X = np.array([[0.0, 0.0], [6.19, 0.5]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestEggCrateFunction:
    def test_initialization(self):
        func = EggCrateFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            EggCrateFunction(dimension=3)

    def test_global_minimum(self):
        func = EggCrateFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-10

    def test_at_origin(self):
        func = EggCrateFunction()
        result = func.evaluate(np.array([0.0, 0.0]))
        assert abs(result) < 1e-10

    def test_evaluate_batch(self):
        func = EggCrateFunction()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)
        assert abs(results[0]) < 1e-10


class TestFreudensteinRothFunction:
    def test_initialization(self):
        func = FreudensteinRothFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            FreudensteinRothFunction(dimension=3)

    def test_global_minimum(self):
        func = FreudensteinRothFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-6

    def test_at_minimum(self):
        func = FreudensteinRothFunction()
        result = func.evaluate(np.array([5.0, 4.0]))
        assert abs(result) < 1e-6

    def test_evaluate_batch(self):
        func = FreudensteinRothFunction()
        X = np.array([[5.0, 4.0], [0.0, 0.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)
        assert abs(results[0]) < 1e-6


class TestGiuntaFunction:
    def test_initialization(self):
        func = GiuntaFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            GiuntaFunction(dimension=3)

    def test_global_minimum(self):
        func = GiuntaFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-3

    def test_minimum_value(self):
        func = GiuntaFunction()
        _, value = func.get_global_minimum()
        assert abs(value - 0.06447) < 0.001

    def test_evaluate_batch(self):
        func = GiuntaFunction()
        X = np.array([[0.0, 0.0], [0.4673, 0.4673]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestHansenFunction:
    def test_initialization(self):
        func = HansenFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            HansenFunction(dimension=3)

    def test_global_minimum(self):
        func = HansenFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 0.01

    def test_minimum_value(self):
        func = HansenFunction()
        _, value = func.get_global_minimum()
        assert abs(value - (-176.5418)) < 0.01

    def test_evaluate_batch(self):
        func = HansenFunction()
        X = np.array([[0.0, 0.0], [-7.59, -7.71]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestHosakiFunction:
    def test_initialization(self):
        func = HosakiFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            HosakiFunction(dimension=3)

    def test_global_minimum(self):
        func = HosakiFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-3

    def test_minimum_value(self):
        func = HosakiFunction()
        _, value = func.get_global_minimum()
        assert abs(value - (-2.3458)) < 0.001

    def test_at_origin(self):
        func = HosakiFunction()
        # f(0,0) = (1)*0*e^0 = 0
        result = func.evaluate(np.array([0.0, 0.0]))
        assert abs(result) < 1e-10

    def test_evaluate_batch(self):
        func = HosakiFunction()
        X = np.array([[0.0, 0.0], [4.0, 2.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestJennrichSampsonFunction:
    def test_initialization(self):
        func = JennrichSampsonFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            JennrichSampsonFunction(dimension=3)

    def test_global_minimum(self):
        func = JennrichSampsonFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1.0  # Relaxed tolerance for this function

    def test_minimum_value(self):
        func = JennrichSampsonFunction()
        _, value = func.get_global_minimum()
        assert abs(value - 124.3622) < 0.1

    def test_evaluate_batch(self):
        func = JennrichSampsonFunction()
        X = np.array([[0.1, 0.1], [0.258, 0.258]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestParsopoulosFunction:
    def test_initialization(self):
        func = ParsopoulosFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            ParsopoulosFunction(dimension=3)

    def test_global_minimum(self):
        func = ParsopoulosFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-10

    def test_at_minimum(self):
        func = ParsopoulosFunction()
        result = func.evaluate(np.array([np.pi / 2, 0.0]))
        assert abs(result) < 1e-10

    def test_known_value(self):
        func = ParsopoulosFunction()
        # f(0, 0) = cos²(0) + sin²(0) = 1 + 0 = 1
        result = func.evaluate(np.array([0.0, 0.0]))
        assert abs(result - 1.0) < 1e-10

    def test_evaluate_batch(self):
        func = ParsopoulosFunction()
        X = np.array([[np.pi / 2, 0.0], [0.0, 0.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)
        assert abs(results[0]) < 1e-10
        assert abs(results[1] - 1.0) < 1e-10
