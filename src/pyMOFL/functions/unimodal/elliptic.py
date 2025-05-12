"""
High Conditioned Elliptic function implementation.

The Elliptic function is a unimodal benchmark function with high conditioning.
It is continuous, convex, and differentiable.

References:
    .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    .. [2] Hansen, N., Müller, S. D., & Koumoutsakos, P. (2003). 
           "Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)". 
           Evolutionary Computation, 11(1), 1-18.
"""

import numpy as np
from ...base import OptimizationFunction


class EllipticFunction(OptimizationFunction):
    """
    High Conditioned Elliptic function: f(x) = sum((10^6)^((i-1)/(D-1)) * x_i^2)
    
    This function is unimodal with a high condition number, making it challenging
    for optimization algorithms that are sensitive to ill-conditioning.
    
    Global minimum: f(0, 0, ..., 0) = 0
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Default bounds are [-100, 100] for each dimension.
        condition (float): Condition number for the function. Default is 10^6.
        
    References:
        .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
               "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
               optimization". Nanyang Technological University, Singapore, Tech. Rep.
        .. [2] Hansen, N., Müller, S. D., & Koumoutsakos, P. (2003). 
               "Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES)". 
               Evolutionary Computation, 11(1), 1-18.
               
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, condition: float = 1e6):
        """
        Initialize the High Conditioned Elliptic function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension. 
                                          Defaults to [-100, 100] for each dimension.
            condition (float, optional): Condition number for the function. Defaults to 10^6.
        """
        super().__init__(dimension, bounds)
        self.condition = condition
        
        # Pre-calculate the weights for each dimension to avoid repeated computation
        if dimension > 1:
            self._weights = np.power(self.condition, np.arange(dimension) / (dimension - 1))
        else:
            self._weights = np.ones(1)  # Special case for 1D
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the High Conditioned Elliptic function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Compute the function value using the pre-calculated weights
        return float(np.sum(self._weights * (x**2)))
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the High Conditioned Elliptic function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Compute the function values using the optimized formula
        return np.sum(self._weights * (X**2), axis=1)
    
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