"""
Shifted function decorator implementation.

This module provides a decorator that applies a shift transformation to a base optimization function.
"""

import numpy as np
from pyMOFL.core.function import OptimizationFunction


class ShiftedFunction(OptimizationFunction):
    """
    Decorator that applies a shift transformation to a base optimization function.
    The shift transformation moves the optimum of the function to a new location.
    Requires an explicit shift vector.
    
    Attributes:
        base (OptimizationFunction): The base optimization function to be shifted.
        shift (np.ndarray): The shift vector to be applied to the input.
        dimension (int): The dimensionality of the function (inherited from the base function).
    """
    
    def __init__(self, base_func: OptimizationFunction, shift: np.ndarray):
        """
        Initialize the shifted function decorator.
        
        Args:
            base_func (OptimizationFunction): The base optimization function to be shifted.
            shift (np.ndarray): The shift vector to be applied to the input.
        """
        if shift is None:
            raise ValueError("ShiftedFunction requires a shift vector. None was provided.")
        self.base = base_func
        self.dimension = base_func.dimension
        self.shift = np.asarray(shift)
        if self.shift.shape[0] != self.dimension:
            raise ValueError(f"Expected shift dimension {self.dimension}, got {self.shift.shape[0]}")
        self.constraint_penalty = base_func.constraint_penalty
    
    def evaluate(self, x):
        return self.base(x - self.shift)
    
    def evaluate_batch(self, X):
        return self.base.evaluate_batch(X - self.shift)
    
    def violations(self, x):
        return self.base.violations(x - self.shift)
    
    @property
    def initialization_bounds(self):
        return self.base.initialization_bounds
    
    @property
    def operational_bounds(self):
        return self.base.operational_bounds 