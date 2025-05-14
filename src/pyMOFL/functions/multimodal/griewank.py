"""
Griewank function implementation.

The Griewank function is a non-convex multimodal benchmark function with many local minima.
The product term in the function creates correlation between variables, making it non-separable.

References:
    .. [1] Griewank, A. O. (1981). "Generalized descent for global optimization". 
           Journal of Optimization Theory and Applications, 34(1), 11-39.
    .. [2] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
"""

import numpy as np
from ...base import OptimizationFunction


class GriewankFunction(OptimizationFunction):
    """
    Griewank function: f(x) = sum(x_i^2/4000) - prod(cos(x_i/sqrt(i))) + 1
    
    Global minimum: f(0, 0, ..., 0) = 0
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Default bounds are [-600, 600] for each dimension.
        
    References:
        .. [1] Griewank, A. O. (1981). "Generalized descent for global optimization". 
               Journal of Optimization Theory and Applications, 34(1), 11-39.
        .. [2] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
               "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
               optimization". Nanyang Technological University, Singapore, Tech. Rep.
               
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None):
        """
        Initialize the Griewank function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension. 
                                          Defaults to [-600, 600] for each dimension.
        """
        # Set default bounds to [-600, 600] for each dimension
        if bounds is None:
            bounds = np.array([[-600, 600]] * dimension)
        
        super().__init__(dimension, bounds)
        
        # Pre-calculate square roots of indices to avoid repeated computation
        # In the original C implementation, it's sqrt(1.0+i) where i is 0-indexed
        self._sqrt_indices = np.sqrt(np.arange(1, dimension + 1))  # sqrt(1), sqrt(2), ..., sqrt(dimension)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Griewank function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Calculate the sum term
        sum_term = np.sum(x**2) / 4000.0
        
        # Calculate the product term
        # Using sqrt(1+i) as in the original C implementation
        prod_term = np.prod(np.cos(x / self._sqrt_indices))
        
        # Combine terms
        return float(sum_term - prod_term + 1.0)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Griewank function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        n_points = X.shape[0]
        results = np.zeros(n_points)
        
        # For each point in the batch
        for p in range(n_points):
            x = X[p]
            
            # Calculate the sum term
            sum_term = np.sum(x**2) / 4000.0
            
            # Calculate the product term using the same approach as evaluate()
            prod_term = np.prod(np.cos(x / self._sqrt_indices))
            
            # Combine terms
            results[p] = sum_term - prod_term + 1.0
        
        return results
    
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