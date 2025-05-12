"""
Biased function decorator implementation.

This module provides a decorator that applies a bias transformation to a base optimization function.
"""

import numpy as np
from ..base import OptimizationFunction


class BiasedFunction(OptimizationFunction):
    """
    A decorator that applies a bias transformation to a base optimization function.
    
    The bias transformation adds a constant to the function value, shifting the minimum value
    but not changing the location of the optimum.
    
    Attributes:
        base (OptimizationFunction): The base optimization function to be biased.
        bias (float): The bias value to be added to the function value.
        dimension (int): The dimensionality of the function (inherited from the base function).
    """
    
    def __init__(self, base_func: OptimizationFunction, bias: float = 0.0):
        """
        Initialize the biased function decorator.
        
        Args:
            base_func (OptimizationFunction): The base optimization function to be biased.
            bias (float, optional): The bias value to be added to the function value. Defaults to 0.0.
        """
        self.base = base_func
        self.dimension = base_func.dimension
        self.bias = bias
        
        # Use the same bounds as the base function
        self._bounds = base_func.bounds.copy()
    
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
        Evaluate the biased function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x with the bias added.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Evaluate the base function and add the bias
        return self.base.evaluate(x) + self.bias
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the biased function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point with the bias added.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Evaluate the base function and add the bias
        return self.base.evaluate_batch(X) + self.bias 