import numpy as np
import pytest
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum

class DummyFunction(OptimizationFunction):
    def __init__(self, bounds, constraint_violation=0.0):
        self.initialization_bounds = bounds
        self.operational_bounds = bounds
        self._constraint_violation = constraint_violation

    def evaluate(self, z):
        return np.sum(z)

    def violations(self, x):
        return self._constraint_violation

def test_call_within_bounds():
    bounds = Bounds(low=np.array([0.0]), high=np.array([1.0]))
    f = DummyFunction(bounds)
    assert f(np.array([0.5])) == 0.5

def test_call_out_of_bounds():
    bounds = Bounds(low=np.array([0.0]), high=np.array([1.0]))
    f = DummyFunction(bounds)
    # Should be clipped to 1.0
    assert f(np.array([2.0])) == 1.0
    # Should be clipped to 0.0
    assert f(np.array([-1.0])) == 0.0

def test_call_with_constraint_violation():
    bounds = Bounds(low=np.array([0.0]), high=np.array([1.0]))
    f = DummyFunction(bounds, constraint_violation=1.0)
    # Any input should return np.nan due to violation
    result = f(np.array([0.5]))
    assert np.isnan(result)

def test_enforce_returns_nan_on_violation():
    bounds = Bounds(low=np.array([0.0]), high=np.array([1.0]))
    f = DummyFunction(bounds, constraint_violation=2.0)
    enforced = f._enforce(np.array([0.5]))
    assert np.all(np.isnan(enforced))

def test_enforce_projects_to_bounds():
    bounds = Bounds(low=np.array([0.0]), high=np.array([1.0]))
    f = DummyFunction(bounds)
    enforced = f._enforce(np.array([2.0]))
    assert np.allclose(enforced, [1.0]) 