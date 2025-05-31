"""
Scaled: Input-transforming decorator for applying a scaling transformation to an OptimizationFunction.

Inherits from InputTransformingFunction. Subclasses should only implement _apply and _apply_batch, never override evaluate or evaluate_batch.

Usage:
    # As a base function (rare, but possible)
    f = Scaled(dimension=3, lambda_coef=2.0)
    value = f(x)

    # As a decorator
    base = SphereFunction(...)
    f = Scaled(base_function=base, lambda_coef=2.0)
    value = f(x)
"""

import numpy as np
from pyMOFL.core.composable_function import InputTransformingFunction

class Scaled(InputTransformingFunction):
    def __init__(self, base_function=None, dimension=None, lambda_coef=1.0, initialization_bounds=None, operational_bounds=None):
        if base_function is not None:
            dim = base_function.dimension
        else:
            dim = dimension
        if isinstance(lambda_coef, (int, float)):
            self.lambda_coef = np.full(dim, lambda_coef)
        else:
            self.lambda_coef = np.asarray(lambda_coef)
            if self.lambda_coef.shape[0] != dim:
                raise ValueError(f"Expected lambda dimension {dim}, got {self.lambda_coef.shape[0]}")
        super().__init__(base_function=base_function, dimension=dimension, initialization_bounds=initialization_bounds, operational_bounds=operational_bounds)

    def _apply(self, x):
        return x / self.lambda_coef

    def _apply_batch(self, X):
        return X / self.lambda_coef