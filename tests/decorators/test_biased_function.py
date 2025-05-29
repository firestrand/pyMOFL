import numpy as np
import pytest
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.decorators.biased import BiasedFunction

class DummyFunction(OptimizationFunction):
    def __init__(self, bounds):
        self.initialization_bounds = bounds
        self.operational_bounds = bounds
        self.dimension = len(bounds.low)
        self.constraint_penalty = 1e8
    def evaluate(self, x):
        return np.sum(x)
    def evaluate_batch(self, X):
        return np.sum(X, axis=1)
    def violations(self, x):
        return 0.0

def test_biased_function_adds_bias():
    bounds = Bounds(low=np.array([0.0]), high=np.array([10.0]))
    base = DummyFunction(bounds)
    biased = BiasedFunction(base, bias=5.0)
    assert biased(np.array([2.0])) == 7.0
    assert biased(np.array([0.0])) == 5.0

def test_biased_function_batch():
    bounds = Bounds(low=np.array([0.0, 0.0]), high=np.array([10.0, 10.0]))
    base = DummyFunction(bounds)
    biased = BiasedFunction(base, bias=2.0)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(biased.evaluate_batch(X), np.array([5.0, 9.0]))

def test_biased_function_delegates_bounds():
    bounds = Bounds(low=np.array([-1.0]), high=np.array([1.0]))
    base = DummyFunction(bounds)
    biased = BiasedFunction(base, bias=1.0)
    assert np.allclose(biased.initialization_bounds.low, [-1.0])
    assert np.allclose(biased.operational_bounds.high, [1.0])

def test_biased_function_delegates_violations():
    class ViolatingDummy(DummyFunction):
        def violations(self, x):
            return 1.0
    bounds = Bounds(low=np.array([0.0]), high=np.array([1.0]))
    base = ViolatingDummy(bounds)
    biased = BiasedFunction(base, bias=1.0)
    assert biased.violations(np.array([0.5])) == 1.0 