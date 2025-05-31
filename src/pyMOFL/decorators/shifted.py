"""
Shifted: Input-transforming decorator for applying a shift transformation to an OptimizationFunction.

Inherits from InputTransformingFunction. Subclasses should only implement _apply and _apply_batch, never override evaluate or evaluate_batch.

Usage:
    # As a base function (rare, but possible)
    f = Shifted(dimension=3, shift=np.array([1, 2, 3]))
    value = f(x)

    # As a decorator
    base = SphereFunction(...)
    f = Shifted(base_function=base, shift=np.array([1, 2, 3]))
    value = f(x)
"""

import numpy as np
from pyMOFL.core.composable_function import InputTransformingFunction

class Shifted(InputTransformingFunction):
    def __init__(self, base_function=None, dimension=None, shift=None, initialization_bounds=None, operational_bounds=None):
        if shift is None:
            raise ValueError("Shifted requires a shift vector. None was provided.")
        self.shift = np.asarray(shift)
        if base_function is not None:
            dim = base_function.dimension
        else:
            dim = dimension
        if self.shift.shape[0] != dim:
            raise ValueError(f"Expected shift dimension {dim}, got {self.shift.shape[0]}")
        super().__init__(base_function=base_function, dimension=dimension, initialization_bounds=initialization_bounds, operational_bounds=operational_bounds)

    def _apply(self, x):
        return x - self.shift

    def _apply_batch(self, X):
        return X - self.shift