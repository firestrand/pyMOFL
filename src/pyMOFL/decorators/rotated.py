"""
Rotated function decorator implementation.

This module provides a decorator that applies a rotation transformation to a base optimization function.
The rotation follows the CEC convention where rotation is applied as M*x (matrix times vector).
"""

import numpy as np
from pyMOFL.core.function import OptimizationFunction


class RotatedFunction(OptimizationFunction):
    """
    Decorator that applies a rotation transformation to a base optimization function.
    The rotation is applied as M*x (matrix times vector), following CEC convention.
    Requires an explicit rotation matrix.
    
    Attributes:
        base (OptimizationFunction): The base optimization function to be rotated.
        rotation_matrix (np.ndarray): The rotation matrix to be applied to the input.
        dimension (int): The dimensionality of the function (inherited from the base function).
    """
    
    def __init__(self, base_func: OptimizationFunction, rotation_matrix: np.ndarray):
        """
        Initialize the rotated function decorator.
        
        Args:
            base_func (OptimizationFunction): The base optimization function to be rotated.
            rotation_matrix (np.ndarray): The rotation matrix to be applied to the input.
        """
        if rotation_matrix is None:
            raise ValueError("RotatedFunction requires a rotation matrix. None was provided.")

        self.base = base_func
        self.dimension = base_func.dimension
        
        # Ensure the rotation matrix is a numpy array with the correct shape
        self.rotation_matrix = np.asarray(rotation_matrix)
        if self.rotation_matrix.shape != (self.dimension, self.dimension):
            raise ValueError(f"Expected rotation matrix shape ({self.dimension}, {self.dimension}), "
                            f"got {self.rotation_matrix.shape}")
        
        self.constraint_penalty = base_func.constraint_penalty
    
    def evaluate(self, x):
        rotated_x = np.dot(self.rotation_matrix, x)
        return self.base(rotated_x)
    
    def evaluate_batch(self, X):
        X = np.asarray(X)
        if X.shape[1] != self.dimension:
            raise ValueError(f"Each input must have dimension {self.dimension}, got {X.shape[1]}")
        return np.array([self.base(np.dot(self.rotation_matrix, x)) for x in X])

    def violations(self, x):
        rotated_x = np.dot(self.rotation_matrix, x)
        return self.base.violations(rotated_x)

    @property
    def initialization_bounds(self):
        return self.base.initialization_bounds

    @property
    def operational_bounds(self):
        return self.base.operational_bounds 