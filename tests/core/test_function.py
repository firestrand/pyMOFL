import numpy as np

from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction


class DummyFunction(OptimizationFunction):
    def __init__(self, bounds, dimension, constraint_violation=0.0):
        self.initialization_bounds = bounds
        self.operational_bounds = bounds
        self.dimension = dimension
        self._constraint_violation = constraint_violation

    def evaluate(self, z: np.ndarray) -> float:  # type: ignore[override]
        return float(np.sum(z))

    def violations(self, x):
        return self._constraint_violation


def test_call_within_bounds():
    bounds = Bounds(low=np.array([0.0]), high=np.array([1.0]))
    f = DummyFunction(bounds, 1)
    assert f(np.array([0.5])) == 0.5
