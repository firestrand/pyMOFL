"""
Rosenbrock function implementation.

The Rosenbrock function is a non-convex unimodal benchmark function.
It has a narrow, curved valley which is difficult for many optimization algorithms to navigate.

References:
    .. [1] Rosenbrock, H.H. (1960). "An automatic method for finding the greatest or least value of a function".
           The Computer Journal, 3(3), 175-184.
    .. [2] Adorio, E.P., & Diliman, U.P. (2005). MVF - "Multivariate Test Functions Library in C for 
           Unconstrained Global Optimization", 2005.
"""

import numpy as np
from ...base import OptimizationFunction


class RosenbrockFunction(OptimizationFunction):
    """
    Rosenbrock function: f(x) = sum_{i=1}^{n-1} [100*(x_{i}^2 - x_{i+1})^2 + (x_i - 1)^2]
    
    Global minimum: f(1, 1, ..., 1) = 0
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Default bounds are [-30, 30] for each dimension.
        
    References:
        .. [1] Rosenbrock, H.H. (1960). "An automatic method for finding the greatest or least value of a function".
               The Computer Journal, 3(3), 175-184.
               
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
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
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Compute the function value using the optimized formula
        term1 = 100 * (x[:-1] ** 2 - x[1:]) ** 2
        term2 = (x[:-1] - 1) ** 2
        
        return float(np.sum(term1 + term2))
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Rosenbrock function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Initialize the result array
        result = np.zeros(X.shape[0])
        
        # Compute the function values for each point
        for i, x in enumerate(X):
            term1 = 100 * (x[:-1] ** 2 - x[1:]) ** 2
            term2 = (x[:-1] - 1) ** 2
            result[i] = np.sum(term1 + term2)
        
        return result
    
    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """
        Get the global minimum of the function.
        
        Args:
            dimension (int): The dimension of the function.
            
        Returns:
            tuple: A tuple containing the global minimum point and the function value at that point.
        """
        global_min_point = np.ones(dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value 