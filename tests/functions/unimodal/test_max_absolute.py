"""
Tests for the MaxAbsolute function (refactored for new bounds logic).
"""

import pytest
import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.decorators import Biased
from pyMOFL.core.function import OptimizationFunction
import warnings
from pyMOFL.decorators.max_absolute import MaxAbsolute

class IdentityFunction(OptimizationFunction):
    def __init__(self, dimension, initialization_bounds=None, operational_bounds=None):
        self.dimension = dimension
        self.initialization_bounds = initialization_bounds
        self.operational_bounds = operational_bounds
    def evaluate(self, x):
        return x
    def evaluate_batch(self, X):
        return X

class TestMaxAbsolute:
    """Tests for the unified MaxAbsolute base/decorator."""
    def test_base_basic(self):
        func = MaxAbsolute(dimension=2)
        assert func.dimension == 2
        assert func(np.array([1, -3])) == 3.0
        assert func(np.array([0, 0])) == 0.0
    def test_base_batch(self):
        func = MaxAbsolute(dimension=2)
        X = np.array([[0, 0], [1, -1], [2, 3]])
        expected = np.array([0, 1, 3])
        np.testing.assert_allclose(func.evaluate_batch(X), expected)
    def test_base_quantization(self):
        op_bounds = Bounds(low=np.array([0]), high=np.array([10]), qtype=QuantizationTypeEnum.INTEGER)
        func = MaxAbsolute(dimension=1, operational_bounds=op_bounds)
        assert func(np.array([2.7])) == 3.0
        assert func(np.array([10.9])) == 10.0
        assert func(np.array([-1.2])) == 0.0
    def test_decorator_basic(self):
        base = IdentityFunction(2)
        func = MaxAbsolute(base_function=base)
        assert func(np.array([1, -3])) == 3.0
        assert func(np.array([0, 0])) == 0.0
    def test_decorator_batch(self):
        base = IdentityFunction(2)
        func = MaxAbsolute(base_function=base)
        X = np.array([[0, 0], [1, -1], [2, 3]])
        expected = np.array([0, 1, 3])
        np.testing.assert_allclose(func.evaluate_batch(X), expected)
    def test_decorator_quantization(self):
        op_bounds = Bounds(low=np.array([0]), high=np.array([10]), qtype=QuantizationTypeEnum.INTEGER)
        base = IdentityFunction(1, operational_bounds=op_bounds)
        func = MaxAbsolute(base_function=base)
        assert func(np.array([2.7])) == 3.0
        assert func(np.array([10.9])) == 10.0
        assert func(np.array([-1.2])) == 0.0
    def test_bounds_and_properties(self):
        func = MaxAbsolute(dimension=2)
        assert func.bounds.shape == (2, 2)
        assert np.allclose(func.bounds, np.array([[-100, 100], [-100, 100]]))
    def test_dimension_validation(self):
        func = MaxAbsolute(dimension=2)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1, 2, 3]))
        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1, 2, 3], [4, 5, 6]]))
    def test_global_minimum(self):
        for dim in [2, 5, 10]:
            point = np.zeros(dim)
            value = 0.0
            func = MaxAbsolute(dimension=dim)
            assert np.isclose(func.evaluate(point), value)
            bias_value = 3.0
            biased_func = Biased(func, bias=bias_value)
            assert np.isclose(biased_func.evaluate(point), value + bias_value) 