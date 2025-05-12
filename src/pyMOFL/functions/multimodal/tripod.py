"""
Tripod function implementation.

The Tripod function is a 2D benchmark function with discontinuities along 
the coordinate axes, creating steep ridges and flat valleys.

References:
    .. [1] Molga, M., & Smutnicki, C. (2005). "Test Functions for Optimization Needs."
           Technical report, Technical University of Gdańsk.
    .. [2] Clerc, M. (2012). "Standard PSO 2007/2011 Benchmark Documentation."
           Technical report, see also Zambrano-Bigiarini et al. (2013).
"""

import numpy as np
from ...base import OptimizationFunction


class TripodFunction(OptimizationFunction):
    """
    Tripod benchmark function.
    
    Mathematical definition:
    f(x1,x2) = p(x2)(1+p(x1))
             + |x1 + 50·p(x2)(1-2·p(x1))|
             + |x2 + 50(1-2·p(x2))|
    
    where p(z) = 1 if z ≥ 0, and 0 if z < 0.
    
    Global minimum: f(0, -50) = 0
    
    Attributes:
        dimension (int): Always 2 for the Tripod function.
        bounds (np.ndarray): Default bounds are [-100, 100] for each dimension.
    
    Properties:
        - Multimodal
        - Discontinuous
        - Non-separable
    
    References:
        .. [1] Molga, M., & Smutnicki, C. (2005). "Test Functions for Optimization Needs."
               Technical report, Technical University of Gdańsk.
        .. [2] Clerc, M. (2012). "Standard PSO 2007/2011 Benchmark Documentation."
               Technical report, see also Zambrano-Bigiarini et al. (2013).
               
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
    """
    
    def __init__(self, bounds: np.ndarray = None):
        """
        Initialize the Tripod function.
        
        Args:
            bounds (np.ndarray, optional): Bounds for each dimension.
                                          Defaults to [-100, 100] for each dimension.
        """
        # Tripod function is always 2D
        dimension = 2
        
        # Set default bounds to [-100, 100] for each dimension
        if bounds is None:
            bounds = np.array([[-100, 100]] * dimension)
        
        super().__init__(dimension, bounds)
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Tripod function at point x.
        
        Args:
            x (np.ndarray): A 2D point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Compute p(x1) and p(x2)
        p = np.array(x >= 0, dtype=float)
        
        # Compute the function value using the formula
        term1 = p[1] * (1 + p[0])
        term2 = np.abs(x[0] + 50 * p[1] * (1 - 2 * p[0]))
        term3 = np.abs(x[1] + 50 * (1 - 2 * p[1]))
        
        return float(term1 + term2 + term3)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Tripod function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of 2D points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)
        
        # Compute p(x1) and p(x2) for all points
        p = np.array(X >= 0, dtype=float)
        
        # Compute the function values using the formula
        term1 = p[:, 1] * (1 + p[:, 0])
        term2 = np.abs(X[:, 0] + 50 * p[:, 1] * (1 - 2 * p[:, 0]))
        term3 = np.abs(X[:, 1] + 50 * (1 - 2 * p[:, 1]))
        
        return term1 + term2 + term3
    
    @staticmethod
    def get_global_minimum(dimension: int = 2) -> tuple:
        """
        Get the global minimum of the function.
        
        Args:
            dimension (int, optional): The dimension of the function. 
                                      Defaults to 2, as Tripod is always 2D.
            
        Returns:
            tuple: A tuple containing the global minimum point and the function value at that point.
        """
        if dimension != 2:
            raise ValueError("Tripod function is only defined for 2 dimensions")
            
        global_min_point = np.array([0.0, -50.0])
        global_min_value = 0.0
        return global_min_point, global_min_value 