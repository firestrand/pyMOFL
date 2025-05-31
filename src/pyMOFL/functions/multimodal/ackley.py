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
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.base import register_function


@register_function("Ackley")
class AckleyFunction(OptimizationFunction):
    """
    Ackley function: f(x) = -20·exp(-0.2·sqrt(sum(x_i^2)/D)) - exp(sum(cos(2π·x_i))/D) + 20 + e
    
    Global minimum: f(0, 0, ..., 0) = 0
    
    Attributes:
        dimension (int): The dimensionality of the function.
        initialization_bounds (Bounds): Bounds for initialization.
        operational_bounds (Bounds): Bounds for operation.
        a (float): Coefficient for the first exponential term. Default is 20.
        b (float): Coefficient for the squared term. Default is 0.2.
        c (float): Coefficient for the cosine term. Default is 2π.
    
    References:
        .. [1] Ackley, D. H. (1987). "A connectionist machine for genetic hillclimbing".
               Kluwer Academic Publishers.
        .. [2] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
               "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
               optimization". Nanyang Technological University, Singapore, Tech. Rep.
    
    Note:
        To add a bias to the function, use the BiasedFunction decorator from the decorators module.
    """
    
    def __init__(self, dimension: int,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None,
                 a: float = 20.0, b: float = 0.2, c: float = 2.0 * np.pi):
        """
        Initialize the Ackley function.
        
        Args:
            dimension (int): The dimensionality of the function.
            initialization_bounds (Bounds, optional): Bounds for initialization. Defaults to [-32.768, 32.768] for each dimension.
            operational_bounds (Bounds, optional): Bounds for operation. Defaults to [-32.768, 32.768] for each dimension.
            a (float, optional): Coefficient for the first exponential term. Defaults to 20.
            b (float, optional): Coefficient for the squared term. Defaults to 0.2.
            c (float, optional): Coefficient for the cosine term. Defaults to 2π.
            
        Note:
            To add a bias to the function, use the BiasedFunction decorator from the decorators module.
        """
        default_bounds = Bounds(
            low=np.full(dimension, -32.768),
            high=np.full(dimension, 32.768),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds
        )
        self.a = a
        self.b = b
        self.c = c
        self._const = self.a + np.e
        self._inv_d = 1.0 / dimension

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Ackley function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        x = self._validate_input(x)

        s1 = np.dot(x, x)                           # Σ x²
        s2 = np.cos(self.c * x).sum()               # Σ cos(2πx)

        term1 = -self.a * np.exp(-self.b * np.sqrt(s1 * self._inv_d))
        term2 = -np.exp(s2 * self._inv_d)
        return float(term1 + term2 + self._const)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Ackley function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        X = self._validate_batch_input(X)

        # Σ x²  per row
        s1 = np.einsum("ij,ij->i", X, X, optimize="greedy")  # faster than (X**2).sum(axis=1)
        
        # Σ cos(2πx)  per row - create a copy of X to avoid modifying the input
        s2 = np.cos(self.c * X).sum(axis=1)  # don't reuse X's buffer to prevent side effects

        term1 = -self.a * np.exp(-self.b * np.sqrt(s1 * self._inv_d))
        term2 = -np.exp(s2 * self._inv_d)
        return term1 + term2 + self._const
    
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