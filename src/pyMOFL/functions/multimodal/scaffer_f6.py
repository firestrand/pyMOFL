"""
Schaffer's F6 function implementation.

This function is a challenging multimodal test function with a single global minimum
surrounded by many local minima. The function is symmetric, and the global minimum
is located at the origin.

References:
    .. [1] Schaffer, J.D. (1985). "Multiple Objective Optimization with Vector Evaluated Genetic
           Algorithms". In Proceedings of the 1st International Conference on Genetic Algorithms. 
           L. Erlbaum Associates Inc., pp. 93-100.
    .. [2] Molga, M., & Smutnicki, C. (2005). "Test functions for optimization needs".
"""

import numpy as np
from ...base import OptimizationFunction


class SchafferF6Function(OptimizationFunction):
    """
    Schaffer's F6 function: f(x) = 0.5 + (sin²(√(x₁² + x₂²)) - 0.5) / (1 + 0.001 * (x₁² + x₂²))²
    
    Global minimum: f(0, 0) = 0
    
    The function is typically evaluated in the range [-100, 100] for each dimension, 
    though it is usually restricted to 2 dimensions.
    
    Attributes:
        dimension (int): The dimensionality of the function (should be 2).
        bounds (np.ndarray): Default bounds are [-100, 100] for each dimension.
        
    References:
        .. [1] Schaffer, J.D. (1985). "Multiple Objective Optimization with Vector Evaluated Genetic
               Algorithms". In Proceedings of the 1st International Conference on Genetic Algorithms. 
               L. Erlbaum Associates Inc., pp. 93-100.
               
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
    """
    
    def __init__(self, dimension: int = 2, bounds: np.ndarray = None):
        """
        Initialize the Schaffer's F6 function.
        
        Args:
            dimension (int, optional): The dimensionality of the function. Defaults to 2.
                                      Warning: This function is typically defined for 2 dimensions,
                                      though it can be extended to higher dimensions.
            bounds (np.ndarray, optional): Bounds for each dimension. 
                                          Defaults to [-100, 100] for each dimension.
        """
        # Check dimension
        if dimension != 2:
            print(f"Warning: Schaffer's F6 function is typically defined for 2 dimensions. Using {dimension} dimensions.")
        
        # Set default bounds to [-100, 100] for each dimension
        if bounds is None:
            bounds = np.array([[-100, 100]] * dimension)
        
        super().__init__(dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Schaffer's F6 function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Compute the function value
        sum_squares = np.sum(x**2)
        numerator = np.sin(np.sqrt(sum_squares))**2 - 0.5
        denominator = (1 + 0.001 * sum_squares)**2
        
        return float(0.5 + numerator / denominator)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Schaffer's F6 function.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
                          Shape should be (n_points, dimension).
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate and preprocess the batch input
        X = self._validate_batch_input(X)
        
        # Compute the function values for all points in the batch
        sum_squares = np.sum(X**2, axis=1)
        numerator = np.sin(np.sqrt(sum_squares))**2 - 0.5
        denominator = (1 + 0.001 * sum_squares)**2
        
        return 0.5 + numerator / denominator
    
    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """
        Get the global minimum of the function.
        
        Args:
            dimension (int): The dimension of the function.
            
        Returns:
            tuple: A tuple containing the global minimum point and the function value at that point.
        """
        global_min_point = np.zeros(dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value


# For backward compatibility and simpler importing
ScafferF6Function = SchafferF6Function  # Common alternative spelling 