"""
Scaled function decorator implementation.

This module provides a decorator that applies a scaling transformation to a base optimization function.
"""

import numpy as np
from ..base import OptimizationFunction


class ScaledFunction(OptimizationFunction):
    """
    A decorator that applies a scaling transformation to a base optimization function.
    
    The scaling transformation scales the input of the function by a lambda coefficient.
    This is useful for functions that need to be compressed or stretched in the input space.
    
    Attributes:
        base (OptimizationFunction): The base optimization function to be scaled.
        lambda_coef (float or np.ndarray): The scaling coefficient(s) to be applied to the input.
        dimension (int): The dimensionality of the function (inherited from the base function).
    """
    
    def __init__(self, base_func: OptimizationFunction, lambda_coef: float or np.ndarray = 1.0):
        """
        Initialize the scaled function decorator.
        
        Args:
            base_func (OptimizationFunction): The base optimization function to be scaled.
            lambda_coef (float or np.ndarray, optional): The scaling coefficient(s) to be applied to the input.
                If a float, the same scaling is applied to all dimensions.
                If an array, each dimension is scaled by the corresponding coefficient.
                Default is 1.0 (no scaling).
        """
        self.base = base_func
        self.dimension = base_func.dimension
        
        # Handle scalar lambda coefficient
        if isinstance(lambda_coef, (int, float)):
            self.lambda_coef = np.full(self.dimension, lambda_coef)
        else:
            # Ensure the lambda is a numpy array with the correct dimension
            self.lambda_coef = np.asarray(lambda_coef)
            if self.lambda_coef.shape[0] != self.dimension:
                raise ValueError(f"Expected lambda dimension {self.dimension}, got {self.lambda_coef.shape[0]}")
        
        # Adjust the bounds to account for the scaling
        # If lambda is positive, bounds are scaled normally
        # If lambda is negative, bounds are scaled and flipped
        base_bounds = base_func.bounds
        self._bounds = np.zeros_like(base_bounds)
        
        for i in range(self.dimension):
            lambda_i = self.lambda_coef[i]
            if lambda_i > 0:
                self._bounds[i, 0] = base_bounds[i, 0] * lambda_i
                self._bounds[i, 1] = base_bounds[i, 1] * lambda_i
            else:
                # If lambda is negative, the bounds need to be flipped
                self._bounds[i, 0] = base_bounds[i, 1] * lambda_i
                self._bounds[i, 1] = base_bounds[i, 0] * lambda_i
    
    @property
    def bounds(self) -> np.ndarray:
        """
        Get the search space bounds for the function.
        
        Returns:
            np.ndarray: A 2D array of shape (dimension, 2) with lower and upper bounds.
        """
        return self._bounds
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the scaled function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Apply the scaling transformation and evaluate the base function
        return self.base.evaluate(x / self.lambda_coef)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the scaled function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Apply the scaling transformation and evaluate the base function in batches
        scaled_X = X / self.lambda_coef
        return self.base.evaluate_batch(scaled_X) 