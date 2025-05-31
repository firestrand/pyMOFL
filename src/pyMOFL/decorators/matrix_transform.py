"""
MatrixTransform: Input-transforming decorator for applying a matrix transformation to an OptimizationFunction.

Inherits from InputTransformingFunction. Subclasses should only implement _apply and _apply_batch, never override evaluate or evaluate_batch.

Usage:
    # As a decorator
    base = SphereFunction(...)
    f = MatrixTransform(base_function=base, matrix_data=np.eye(3))
    value = f(x)
"""

import numpy as np
from pyMOFL.core.composable_function import InputTransformingFunction

class MatrixTransform(InputTransformingFunction):
    def __init__(self, base_function=None, dimension=None, matrix_data=None, initialization_bounds=None, operational_bounds=None):
        if matrix_data is None:
            raise ValueError("MatrixTransform requires matrix_data.")
        self.A_matrix = np.array(matrix_data)
        if base_function is not None:
            dim = base_function.dimension
        else:
            dim = dimension
        if self.A_matrix.shape != (dim, dim):
            raise ValueError(f"Matrix shape {self.A_matrix.shape} incompatible with dimension {dim}")
        # Set A matrix in base function if it supports it
        if base_function is not None and hasattr(base_function, 'set_A_matrix'):
            base_function.set_A_matrix(self.A_matrix)
        super().__init__(base_function=base_function, dimension=dimension, initialization_bounds=initialization_bounds, operational_bounds=operational_bounds)

    def set_B_vector(self, B_vector: np.ndarray):
        if hasattr(self.base_function, 'set_B_vector'):
            self.base_function.set_B_vector(B_vector)

    def _apply(self, x):
        return np.dot(self.A_matrix, x)

    def _apply_batch(self, X):
        return np.dot(X, self.A_matrix.T)