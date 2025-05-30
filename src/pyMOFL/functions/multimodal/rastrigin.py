"""
Rastrigin function implementation.

The Rastrigin function is a non-convex multimodal benchmark function with many local minima.
It is often used to test the ability of optimization algorithms to escape local optima.

References:
    .. [1] Rastrigin, L.A. (1974). "Systems of extremal control". Mir, Moscow.
    .. [2] Adorio, E.P., & Diliman, U.P. (2005). MVF - "Multivariate Test Functions Library in C for 
           Unconstrained Global Optimization", 2005.
    .. [3] Mühlenbein, H., Schomisch, D., & Born, J. (1991). "The parallel genetic algorithm as function 
           optimizer". Parallel Computing, 17(6-7), 619-632.
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.base import register_function


@register_function("Rastrigin")
class RastriginFunction(OptimizationFunction):
    """
    Rastrigin function: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Global minimum: f(0, 0, ..., 0) = 0
    
    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Default bounds are [-5.12, 5.12] for each dimension.
        
    References:
        .. [1] Rastrigin, L.A. (1974). "Systems of extremal control". Mir, Moscow.
        .. [2] Mühlenbein, H., Schomisch, D., & Born, J. (1991). "The parallel genetic algorithm as function 
               optimizer". Parallel Computing, 17(6-7), 619-632.
               
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
    """
    
    def __init__(self, dimension: int,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None):
        """
        Initialize the Rastrigin function.
        
        Args:
            dimension (int): The dimensionality of the function.
            initialization_bounds (Bounds, optional): Bounds for initialization. 
                                                    Defaults to [-5.12, 5.12] for each dimension.
            operational_bounds (Bounds, optional): Bounds for operational use. 
                                                  Defaults to [-5.12, 5.12] for each dimension.
        """
        default_bounds = Bounds(
            low=np.full(dimension, -5.12),
            high=np.full(dimension, 5.12),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds
        )
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Rastrigin function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        x = self._validate_input(x)
        return float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10))
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Rastrigin function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        X = self._validate_batch_input(X)
        return np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)
    
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