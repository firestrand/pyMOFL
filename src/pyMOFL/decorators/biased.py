"""
Biased: Output-transforming decorator for adding a constant bias to an OptimizationFunction.

Inherits from OutputTransformingFunction. Subclasses should only implement _apply and _apply_batch, never override evaluate or evaluate_batch.

Usage:
    # As a base function (rare, but possible)
    f = Biased(dimension=1, bias=5.0)
    value = f(x)  # always returns 5.0

    # As a decorator
    base = SphereFunction(...)
    f = Biased(base_function=base, bias=5.0)
    value = f(x)
"""
from pyMOFL.core.composable_function import OutputTransformingFunction
import numpy as np

class Biased(OutputTransformingFunction):
    def __init__(self, base_function=None, dimension=None, bias=0.0, initialization_bounds=None, operational_bounds=None):
        self.bias = bias
        super().__init__(base_function=base_function, dimension=dimension, initialization_bounds=initialization_bounds, operational_bounds=operational_bounds)

    def _apply(self, y):
        y = np.asarray(y)
        if y.shape == () or y.size == 1:
            return float(y) + self.bias
        return y + self.bias

    def _apply_batch(self, Y):
        return np.asarray(Y) + self.bias