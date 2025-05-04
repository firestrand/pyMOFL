"""
Shifted function decorator implementation.

This module provides a decorator that applies a shift transformation to a base optimization function.
"""

import numpy as np
from ..base import OptimizationFunction


class ShiftedFunction(OptimizationFunction):
    """
    A decorator that applies a shift transformation to a base optimization function.
    
    The shift transformation moves the optimum of the function to a new location.
    
    Attributes:
        base (OptimizationFunction): The base optimization function to be shifted.
        shift (np.ndarray): The shift vector to be applied to the input.
        dimension (int): The dimensionality of the function (inherited from the base function).
    """
    
    def __init__(self, base_func: OptimizationFunction, shift: np.ndarray = None):
        """
        Initialize the shifted function decorator.
        
        Args:
            base_func (OptimizationFunction): The base optimization function to be shifted.
            shift (np.ndarray, optional): The shift vector to be applied to the input.
                                         If None, a random shift vector is generated within the bounds.
        """
        self.base = base_func
        self.dimension = base_func.dimension
        
        # If no shift is provided, generate a random shift within the bounds
        if shift is None:
            bounds = base_func.bounds
            # Generate a random shift within the bounds
            self.shift = np.array([np.random.uniform(low, high) for low, high in bounds])
        else:
            # Ensure the shift is a numpy array with the correct dimension
            self.shift = np.asarray(shift)
            if self.shift.shape[0] != self.dimension:
                raise ValueError(f"Expected shift dimension {self.dimension}, got {self.shift.shape[0]}")
        
        # Adjust the bounds to account for the shift
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
        Evaluate the shifted function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Apply the shift transformation and evaluate the base function
        return self.base.evaluate(x - self.shift)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the shifted function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Apply the shift transformation and evaluate the base function
        return self.base.evaluate_batch(X - self.shift) 