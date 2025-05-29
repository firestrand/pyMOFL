import numpy as np
import pytest
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.decorators.shifted import ShiftedFunction

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

def test_shifted_function_applies_shift():
    bounds = Bounds(low=np.array([0.0, 0.0]), high=np.array([10.0, 10.0]))
    base = DummyFunction(bounds)
    shift = np.array([1.0, 2.0])
    shifted = ShiftedFunction(base, shift=shift)
    # Should subtract shift before evaluating
    assert shifted(np.array([2.0, 4.0])) == 3.0  # (2-1)+(4-2)=1+2=3
    assert shifted(np.array([1.0, 2.0])) == 0.0

def test_shifted_function_batch():
    bounds = Bounds(low=np.array([0.0, 0.0]), high=np.array([10.0, 10.0]))
    base = DummyFunction(bounds)
    shift = np.array([1.0, 1.0])
    shifted = ShiftedFunction(base, shift=shift)
    X = np.array([[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_allclose(shifted.evaluate_batch(X), np.array([3.0, 7.0]))

def test_shifted_function_delegates_bounds():
    bounds = Bounds(low=np.array([-1.0]), high=np.array([1.0]))
    base = DummyFunction(bounds)
    shift = np.array([0.5])
    shifted = ShiftedFunction(base, shift=shift)
    assert np.allclose(shifted.initialization_bounds.low, [-1.0])
    assert np.allclose(shifted.operational_bounds.high, [1.0])

def test_shifted_function_delegates_violations():
    class ViolatingDummy(DummyFunction):
        def violations(self, x):
            return np.sum(x)
    bounds = Bounds(low=np.array([0.0]), high=np.array([1.0]))
    base = ViolatingDummy(bounds)
    shift = np.array([0.5])
    shifted = ShiftedFunction(base, shift=shift)
    # Should pass x-shift to base violations
    assert shifted.violations(np.array([1.5])) == 1.0  # (1.5-0.5)=1.0

def test_shifted_function_requires_shift():
    bounds = Bounds(low=np.array([0.0]), high=np.array([1.0]))
    base = DummyFunction(bounds)
    with pytest.raises(ValueError):
        ShiftedFunction(base, shift=None) 