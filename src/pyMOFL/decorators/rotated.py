"""
Rotated: Input-transforming decorator for applying a rotation transformation to an OptimizationFunction.

Inherits from InputTransformingFunction. Subclasses should only implement _apply and _apply_batch, never override evaluate or evaluate_batch.

Usage:
    # As a base function (rare, but possible)
    f = Rotated(dimension=3, rotation_matrix=np.eye(3))
    value = f(x)

    # As a decorator
    base = SphereFunction(...)
    f = Rotated(base_function=base, rotation_matrix=np.eye(3))
    value = f(x)
"""

import numpy as np
from pyMOFL.core.composable_function import InputTransformingFunction

class Rotated(InputTransformingFunction):
    def __init__(self, base_function=None, dimension=None, rotation_matrix=None, initialization_bounds=None, operational_bounds=None):
        if rotation_matrix is None:
            raise ValueError("Rotated requires a rotation matrix. None was provided.")
        self.rotation_matrix = np.asarray(rotation_matrix)
        if base_function is not None:
            dim = base_function.dimension
        else:
            dim = dimension
        if self.rotation_matrix.shape != (dim, dim):
            raise ValueError(f"Expected rotation matrix shape ({dim}, {dim}), got {self.rotation_matrix.shape}")
        super().__init__(base_function=base_function, dimension=dimension, initialization_bounds=initialization_bounds, operational_bounds=operational_bounds)

    def _apply(self, x):
        return np.dot(self.rotation_matrix, x)

    def _apply_batch(self, X):
        return np.dot(X, self.rotation_matrix.T)