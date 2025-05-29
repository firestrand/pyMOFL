"""
Matrix transform function decorator implementation.

This module provides a decorator that applies a matrix transformation to the input 
before evaluating the base optimization function. This is used for functions like
Schwefel's Problem 2.6 which require matrix multiplication transformations.
"""

import numpy as np
from pyMOFL.core.function import OptimizationFunction


class MatrixTransformFunction(OptimizationFunction):
    """
    A decorator that applies a matrix transformation to a base optimization function.
    
    The matrix transformation loads an A matrix from file and sets it in the base function.
    For Schwefel's Problem 2.6, this enables computation of B = A * adjusted_shift_vector
    by a subsequent boundary adjusted shift decorator.
    
    Attributes:
        base (OptimizationFunction): The base optimization function to be decorated.
        A_matrix (np.ndarray): The transformation matrix loaded from file.
        dimension (int): The dimensionality inherited from the base function.
        bounds (np.ndarray): The bounds inherited from the base function.
    """
    
    def __init__(self, base_function: OptimizationFunction, matrix_data: np.ndarray):
        """
        Initialize the matrix transform decorator.
        
        Args:
            base_function: The base optimization function to apply matrix transform to.
            matrix_data: The matrix data (A matrix) to use for transformation.
        """
        self.base = base_function
        self.dimension = base_function.dimension
        self.constraint_penalty = base_function.constraint_penalty
        
        # Store A matrix and set it in base function
        self.A_matrix = np.array(matrix_data)
        
        # Ensure matrix is compatible with dimension
        if self.A_matrix.shape[0] != self.dimension or self.A_matrix.shape[1] != self.dimension:
            raise ValueError(f"Matrix shape {self.A_matrix.shape} incompatible with dimension {self.dimension}")
        
        # Set A matrix in base function if it supports it
        if hasattr(self.base, 'set_A_matrix'):
            self.base.set_A_matrix(self.A_matrix)
    
    def set_B_vector(self, B_vector: np.ndarray):
        """
        Set the B vector in the base function (for use by boundary adjusted shift decorator).
        
        Args:
            B_vector: The B vector to set in the base function.
        """
        if hasattr(self.base, 'set_B_vector'):
            self.base.set_B_vector(B_vector)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the function with matrix transformation applied.
        
        The actual matrix transformation (A*x) is handled by the base function.
        This decorator just passes through to the base function.
        
        Args:
            x: Input vector to evaluate.
            
        Returns:
            Function value from base function.
        """
        return self.base.evaluate(x)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the function for a batch of inputs with matrix transformation.
        
        Args:
            X: Batch of input vectors to evaluate.
            
        Returns:
            Array of function values from base function.
        """
        return self.base.evaluate_batch(X)

    def violations(self, x):
        return self.base.violations(x)

    @property
    def initialization_bounds(self):
        return self.base.initialization_bounds

    @property
    def operational_bounds(self):
        return self.base.operational_bounds 