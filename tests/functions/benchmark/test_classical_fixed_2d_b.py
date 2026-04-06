"""
Tests for classical fixed 2D optimization functions (Batch B).

Tests for: Damavandi, Leon, CrossLegTable, CrownedCross, DeckkersAarts,
ElAttar, Exp2, TestTubeHolder, Ursem01, VenterSobieski, Zettl.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.cross_leg_table import CrossLegTableFunction
from pyMOFL.functions.benchmark.crowned_cross import CrownedCrossFunction
from pyMOFL.functions.benchmark.damavandi import DamavandiFunction
from pyMOFL.functions.benchmark.deckkers_aarts import DeckkersAartsFunction
from pyMOFL.functions.benchmark.el_attar import ElAttarVidyasagarDuttaFunction
from pyMOFL.functions.benchmark.exp2 import Exp2Function
from pyMOFL.functions.benchmark.leon import LeonFunction
from pyMOFL.functions.benchmark.test_tube_holder import TestTubeHolderFunction
from pyMOFL.functions.benchmark.ursem01 import Ursem01Function
from pyMOFL.functions.benchmark.venter_sobieski import VenterSobieskiFunction
from pyMOFL.functions.benchmark.zettl import ZettlFunction


class TestDamavandiFunction:
    def test_initialization(self):
        func = DamavandiFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            DamavandiFunction(dimension=3)

    def test_global_minimum(self):
        func = DamavandiFunction()
        _point, value = func.get_global_minimum()
        assert abs(value) < 1e-6

    def test_near_minimum(self):
        func = DamavandiFunction()
        # At exact (2,2) the sinc terms are 0, so formula gives 0
        result = func.evaluate(np.array([2.0, 2.0]))
        assert abs(result) < 1e-6

    def test_evaluate_batch(self):
        func = DamavandiFunction()
        X = np.array([[2.0, 2.0], [5.0, 5.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestLeonFunction:
    def test_initialization(self):
        func = LeonFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            LeonFunction(dimension=3)

    def test_global_minimum(self):
        func = LeonFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-10

    def test_at_minimum(self):
        func = LeonFunction()
        result = func.evaluate(np.array([1.0, 1.0]))
        assert abs(result) < 1e-10

    def test_evaluate_batch(self):
        func = LeonFunction()
        X = np.array([[1.0, 1.0], [0.0, 0.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)
        assert abs(results[0]) < 1e-10


class TestCrossLegTableFunction:
    def test_initialization(self):
        func = CrossLegTableFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            CrossLegTableFunction(dimension=3)

    def test_global_minimum(self):
        func = CrossLegTableFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-6

    def test_minimum_value(self):
        func = CrossLegTableFunction()
        _, value = func.get_global_minimum()
        assert abs(value - (-1.0)) < 1e-6

    def test_evaluate_batch(self):
        func = CrossLegTableFunction()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestCrownedCrossFunction:
    def test_initialization(self):
        func = CrownedCrossFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            CrownedCrossFunction(dimension=3)

    def test_global_minimum(self):
        func = CrownedCrossFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-6

    def test_minimum_value(self):
        func = CrownedCrossFunction()
        _, value = func.get_global_minimum()
        assert abs(value - 0.0001) < 1e-6

    def test_evaluate_batch(self):
        func = CrownedCrossFunction()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestDeckkersAartsFunction:
    def test_initialization(self):
        func = DeckkersAartsFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            DeckkersAartsFunction(dimension=3)

    def test_global_minimum(self):
        func = DeckkersAartsFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1.0

    def test_minimum_value(self):
        func = DeckkersAartsFunction()
        _, value = func.get_global_minimum()
        assert abs(value - (-24777.0)) < 10.0  # Approximate from literature

    def test_evaluate_batch(self):
        func = DeckkersAartsFunction()
        X = np.array([[0.0, 15.0], [0.0, -15.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestElAttarFunction:
    def test_initialization(self):
        func = ElAttarVidyasagarDuttaFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            ElAttarVidyasagarDuttaFunction(dimension=3)

    def test_global_minimum(self):
        func = ElAttarVidyasagarDuttaFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 0.1

    def test_minimum_value(self):
        func = ElAttarVidyasagarDuttaFunction()
        _, value = func.get_global_minimum()
        assert abs(value - 1.7128) < 0.1

    def test_evaluate_batch(self):
        func = ElAttarVidyasagarDuttaFunction()
        X = np.array([[3.409, -2.171], [0.0, 0.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestExp2Function:
    def test_initialization(self):
        func = Exp2Function()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            Exp2Function(dimension=3)

    def test_global_minimum(self):
        func = Exp2Function()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-10

    def test_at_minimum(self):
        func = Exp2Function()
        result = func.evaluate(np.array([1.0, 10.0]))
        assert abs(result) < 1e-10

    def test_evaluate_batch(self):
        func = Exp2Function()
        X = np.array([[1.0, 10.0], [5.0, 5.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)
        assert abs(results[0]) < 1e-10


class TestTestTubeHolderFunction:
    def test_initialization(self):
        func = TestTubeHolderFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            TestTubeHolderFunction(dimension=3)

    def test_global_minimum(self):
        func = TestTubeHolderFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 0.01

    def test_minimum_value(self):
        func = TestTubeHolderFunction()
        _, value = func.get_global_minimum()
        assert abs(value - (-10.8723)) < 0.01

    def test_evaluate_batch(self):
        func = TestTubeHolderFunction()
        X = np.array([[0.0, 0.0], [-np.pi / 2, 0.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestUrsem01Function:
    def test_initialization(self):
        func = Ursem01Function()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            Ursem01Function(dimension=3)

    def test_global_minimum(self):
        func = Ursem01Function()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 0.01

    def test_minimum_value(self):
        func = Ursem01Function()
        _, value = func.get_global_minimum()
        assert abs(value - (-4.8168)) < 0.01

    def test_evaluate_batch(self):
        func = Ursem01Function()
        X = np.array([[0.0, 0.0], [1.697, 0.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)


class TestVenterSobieskiFunction:
    def test_initialization(self):
        func = VenterSobieskiFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            VenterSobieskiFunction(dimension=3)

    def test_global_minimum(self):
        func = VenterSobieskiFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-6

    def test_at_origin(self):
        func = VenterSobieskiFunction()
        result = func.evaluate(np.array([0.0, 0.0]))
        assert abs(result - (-400.0)) < 1e-6

    def test_evaluate_batch(self):
        func = VenterSobieskiFunction()
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)
        assert abs(results[0] - (-400.0)) < 1e-6


class TestZettlFunction:
    def test_initialization(self):
        func = ZettlFunction()
        assert func.dimension == 2

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            ZettlFunction(dimension=3)

    def test_global_minimum(self):
        func = ZettlFunction()
        point, value = func.get_global_minimum()
        result = func.evaluate(point)
        assert abs(result - value) < 1e-4

    def test_minimum_value(self):
        func = ZettlFunction()
        _, value = func.get_global_minimum()
        assert abs(value - (-0.003791)) < 0.001

    def test_evaluate_batch(self):
        func = ZettlFunction()
        X = np.array([[-0.0299, 0.0], [0.0, 0.0]])
        results = func.evaluate_batch(X)
        assert results.shape == (2,)
