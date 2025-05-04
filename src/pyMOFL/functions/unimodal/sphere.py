"""
Sphere function implementation.

The Sphere function is one of the simplest unimodal benchmark functions.
It is continuous, convex, and differentiable.

References:
    .. [1] Adorio, E.P., & Diliman, U.P. (2005). MVF - "Multivariate Test Functions Library in C for 
           Unconstrained Global Optimization", 2005.
    .. [2] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global 
           optimization problems". International Journal of Mathematical Modelling and Numerical 
           Optimisation, 4(2), 150-194.
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
        
    References:
        .. [1] Adorio, E.P., & Diliman, U.P. (2005). MVF - "Multivariate Test Functions Library in C for 
               Unconstrained Global Optimization", 2005.
        .. [2] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global 
               optimization problems". International Journal of Mathematical Modelling and Numerical 
               Optimisation, 4(2), 150-194.
    """
    
    def __init__(self, dimension: int, bias: float = 0.0, bounds: np.ndarray = None):
        """
        Initialize the Sphere function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension. 
                                          Defaults to [-100, 100] for each dimension.
        """
        super().__init__(dimension, bias, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Sphere function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Compute the function value using vectorized operations
        return float(np.sum(x**2) + self.bias)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Sphere function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Compute the function values using vectorized operations
        # This is more efficient than calling evaluate for each point
        return np.sum(X**2, axis=1) + self.bias