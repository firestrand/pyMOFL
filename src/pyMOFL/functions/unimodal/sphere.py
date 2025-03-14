"""
Sphere function implementation.

The Sphere function is one of the simplest unimodal benchmark functions.
It is continuous, convex, and differentiable.
"""

import numpy as np
from ...base import OptimizationFunction


class SphereFunction(OptimizationFunction):
    """
    Sphere function: f(x) = sum(x_i^2)
    
    Global minimum: f(0, 0, ..., 0) = 0
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Default bounds are [-100, 100] for each dimension.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None):
        """
        Initialize the Sphere function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension. 
                                          Defaults to [-100, 100] for each dimension.
        """
        super().__init__(dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Sphere function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Ensure x is a numpy array
        x = np.asarray(x)
        
        # Check if the input has the correct dimension
        if x.shape[0] != self.dimension:
            raise ValueError(f"Expected input dimension {self.dimension}, got {x.shape[0]}")
        
        # Compute the function value using vectorized operations
        return float(np.sum(x**2))
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Sphere function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Ensure X is a numpy array
        X = np.asarray(X)
        
        # Check if the input has the correct shape
        if X.shape[1] != self.dimension:
            raise ValueError(f"Expected input dimension {self.dimension}, got {X.shape[1]}")
        
        # Compute the function values using vectorized operations
        return np.sum(X**2, axis=1) 