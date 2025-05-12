"""
Ackley function implementation.

The Ackley function is a widely used multimodal test function for optimization algorithms.
It has a global minimum surrounded by an almost flat outer region with many local minima.

References:
    .. [1] Ackley, D. H. (1987). "A connectionist machine for genetic hillclimbing".
           Kluwer Academic Publishers.
    .. [2] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
"""

import numpy as np
from ...base import OptimizationFunction


class AckleyFunction(OptimizationFunction):
    """
    Ackley function: f(x) = -20·exp(-0.2·sqrt(sum(x_i^2)/D)) - exp(sum(cos(2π·x_i))/D) + 20 + e
    
    Global minimum: f(0, 0, ..., 0) = 0
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Default bounds are [-32.768, 32.768] for each dimension.
        a (float): Coefficient for the first exponential term. Default is 20.
        b (float): Coefficient for the squared term. Default is 0.2.
        c (float): Coefficient for the cosine term. Default is 2π.
        
    References:
        .. [1] Ackley, D. H. (1987). "A connectionist machine for genetic hillclimbing".
               Kluwer Academic Publishers.
        .. [2] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
               "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
               optimization". Nanyang Technological University, Singapore, Tech. Rep.
    """
    
    def __init__(self, dimension: int, bounds: np.ndarray = None, 
                 a: float = 20.0, b: float = 0.2, c: float = 2.0*np.pi):
        """
        Initialize the Ackley function.
        
        Args:
            dimension (int): The dimensionality of the function.
            bounds (np.ndarray, optional): Bounds for each dimension. 
                                          Defaults to [-32.768, 32.768] for each dimension.
            a (float, optional): Coefficient for the first exponential term. Defaults to 20.
            b (float, optional): Coefficient for the squared term. Defaults to 0.2.
            c (float, optional): Coefficient for the cosine term. Defaults to 2π.
            
        Note:
            To add a bias to the function, use the BiasedFunction decorator from the decorators module.
        """
        # Set default bounds to [-32.768, 32.768] for each dimension
        if bounds is None:
            bounds = np.array([[-32.768, 32.768]] * dimension)
        
        super().__init__(dimension, bounds)
        
        self.a = a
        self.b = b
        self.c = c
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Ackley function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        n = len(x)
        
        # First exponential term
        sum_squares = np.sum(x**2)
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_squares / n))
        
        # Second exponential term
        sum_cos = np.sum(np.cos(self.c * x))
        term2 = -np.exp(sum_cos / n)
        
        # Combine terms with constants
        return float(term1 + term2 + self.a + np.e)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Ackley function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        n_points, n_dims = X.shape
        results = np.zeros(n_points)
        
        # For each point in the batch
        for p in range(n_points):
            x = X[p]
            
            # First exponential term
            sum_squares = np.sum(x**2)
            term1 = -self.a * np.exp(-self.b * np.sqrt(sum_squares / n_dims))
            
            # Second exponential term
            sum_cos = np.sum(np.cos(self.c * x))
            term2 = -np.exp(sum_cos / n_dims)
            
            # Combine terms with constants
            results[p] = term1 + term2 + self.a + np.e
        
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