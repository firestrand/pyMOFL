"""
Rosenbrock function implementation.

The Rosenbrock function is a non-convex unimodal benchmark function.
It has a narrow, curved valley which is difficult for many optimization algorithms to navigate.
"""

import numpy as np
from ...base import OptimizationFunction


class RosenbrockFunction(OptimizationFunction):
    """
    Rosenbrock function: f(x) = sum_{i=1}^{n-1} [100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    
    Global minimum: f(1, 1, ..., 1) = 0
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Default bounds are [-30, 30] for each dimension.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None):
        """
        Initialize the Rosenbrock function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension. 
                                          Defaults to [-30, 30] for each dimension.
        """
        # Set default bounds to [-30, 30] for each dimension
        if bounds is None:
            bounds = np.array([[-30, 30]] * dimension)
        
        super().__init__(dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Rosenbrock function at point x.
        
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
        # For i=0 to n-2: 100*(x[i+1] - x[i]^2)^2 + (x[i] - 1)^2
        x_i = x[:-1]  # All elements except the last one
        x_i_plus_1 = x[1:]  # All elements except the first one
        
        term1 = 100 * (x_i_plus_1 - x_i**2)**2
        term2 = (x_i - 1)**2
        
        return float(np.sum(term1 + term2))
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Rosenbrock function on a batch of points.
        
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
        
        # Initialize the result array
        result = np.zeros(X.shape[0])
        
        # Compute the function values for each point
        for i in range(X.shape[0]):
            x = X[i]
            x_i = x[:-1]
            x_i_plus_1 = x[1:]
            
            term1 = 100 * (x_i_plus_1 - x_i**2)**2
            term2 = (x_i - 1)**2
            
            result[i] = np.sum(term1 + term2)
        
        return result 