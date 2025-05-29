import numpy as np
import pytest
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.decorators.rotated import RotatedFunction

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

def test_rotated_function_applies_rotation():
    bounds = Bounds(low=np.array([0.0, 0.0]), high=np.array([10.0, 10.0]))
    base = DummyFunction(bounds)
    rotation_matrix = np.array([[0, 1], [1, 0]])  # swaps x and y
    rotated = RotatedFunction(base, rotation_matrix=rotation_matrix)
    # Should swap coordinates before evaluating
    assert rotated(np.array([2.0, 3.0])) == 5.0  # sum([3,2])
    assert rotated(np.array([1.0, 2.0])) == 3.0  # sum([2,1])

def test_rotated_function_batch():
    bounds = Bounds(low=np.array([0.0, 0.0]), high=np.array([10.0, 10.0]))
    base = DummyFunction(bounds)
    rotation_matrix = np.eye(2)
    rotated = RotatedFunction(base, rotation_matrix=rotation_matrix)
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(rotated.evaluate_batch(X), np.array([3.0, 7.0]))

def test_rotated_function_delegates_bounds():
    bounds = Bounds(low=np.array([-1.0]), high=np.array([1.0]))
    base = DummyFunction(bounds)
    rotation_matrix = np.eye(1)
    rotated = RotatedFunction(base, rotation_matrix=rotation_matrix)
    assert np.allclose(rotated.initialization_bounds.low, [-1.0])
    assert np.allclose(rotated.operational_bounds.high, [1.0])

def test_rotated_function_delegates_violations():
    class ViolatingDummy(DummyFunction):
        def violations(self, x):
            return np.sum(x)
    bounds = Bounds(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]))
    base = ViolatingDummy(bounds)
    rotation_matrix = np.eye(2)
    rotated = RotatedFunction(base, rotation_matrix=rotation_matrix)
    # Should pass rotated x to base violations
    assert rotated.violations(np.array([1.0, 2.0])) == 3.0

def test_rotated_function_requires_rotation_matrix():
    bounds = Bounds(low=np.array([0.0]), high=np.array([1.0]))
    base = DummyFunction(bounds)
    with pytest.raises(ValueError):
        RotatedFunction(base, rotation_matrix=None) 