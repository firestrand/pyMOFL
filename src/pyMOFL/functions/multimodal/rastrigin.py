"""
Rastrigin function implementation.

The Rastrigin function is a non-convex multimodal benchmark function with many local minima.
It is often used to test the ability of optimization algorithms to escape local optima.
"""

import numpy as np
from ...base import OptimizationFunction


class RastriginFunction(OptimizationFunction):
    """
    Rastrigin function: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Global minimum: f(0, 0, ..., 0) = 0
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Default bounds are [-5.12, 5.12] for each dimension.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None):
        """
        Initialize the Rastrigin function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension. 
                                          Defaults to [-5.12, 5.12] for each dimension.
        """
        # Set default bounds to [-5.12, 5.12] for each dimension
        if bounds is None:
            bounds = np.array([[-5.12, 5.12]] * dimension)
        
        super().__init__(dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Rastrigin function at point x.
        
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
        return float(10 * self.dimension + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Rastrigin function on a batch of points.
        
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
        return 10 * self.dimension + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=1) 