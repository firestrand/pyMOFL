"""
Weierstrass function implementation.

The Weierstrass function is a multimodal benchmark function that is continuous
but nowhere differentiable. This fractal-like function has an infinite number
of local optima, making it particularly challenging for optimization algorithms.

References:
    .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    .. [2] Weierstrass, K. (1872). "Über continuirliche Functionen eines reellen Arguments, die für keinen 
           Werth des letzteren einen bestimmten Differentialquotienten besitzen".
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.base import register_function


@register_function("Weierstrass")
class WeierstrassFunction(OptimizationFunction):
    """
    Weierstrass function: 
    f(x) = sum(sum(a^k * cos(2π * b^k * (x_i + 0.5)))) - D * sum(a^k * cos(2π * b^k * 0.5))
    
    where the inner sum runs from k=0 to k_max.
    
    Global minimum: f(0, 0, ..., 0) = 0
    
    Input validation is handled by the base class (_validate_input, _validate_batch_input).
    
    Attributes:
        dimension (int): The dimensionality of the function.
        initialization_bounds (Bounds): Bounds for initialization.
        operational_bounds (Bounds): Bounds for operation.
        a (float): Base for the amplitude coefficient. Default is 0.5.
        b (float): Base for the frequency coefficient. Default is 3.0.
        k_max (int): Maximum value of k in the summation. Default is 20.
        
    References:
        .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
               "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
               optimization". Nanyang Technological University, Singapore, Tech. Rep.
        .. [2] Weierstrass, K. (1872). "Über continuirliche Functionen eines reellen Arguments, die für keinen 
               Werth des letzteren einen bestimmten Differentialquotienten besitzen".
    """
    
    def __init__(self, dimension: int,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None,
                 a: float = 0.5, b: float = 3.0, k_max: int = 20):
        """
        Initialize the Weierstrass function.
        
        Args:
            dimension (int): The dimensionality of the function.
            initialization_bounds (Bounds, optional): Bounds for initialization. 
                                                    Defaults to [-0.5, 0.5] for each dimension.
            operational_bounds (Bounds, optional): Bounds for operation. 
                                                  Defaults to [-0.5, 0.5] for each dimension.
            a (float, optional): Base for the amplitude coefficient. Defaults to 0.5.
            b (float, optional): Base for the frequency coefficient. Defaults to 3.0.
            k_max (int, optional): Maximum value of k in the summation. Defaults to 20.
        """
        default_bounds = Bounds(
            low=np.full(dimension, -0.5),
            high=np.full(dimension, 0.5),
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
        self.k_max = k_max
        # Pre-calculate constants to avoid repeated computation
        self._a_powers = np.power(self.a, np.arange(self.k_max + 1))
        self._b_powers = np.power(self.b, np.arange(self.k_max + 1))
        self._constant_term = dimension * np.sum(self._a_powers * np.cos(2 * np.pi * self._b_powers * 0.5))
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Weierstrass function at point x.
        
        Args:
            x (np.ndarray): A point in the search space.
            
        Returns:
            float: The function value at point x.
        """
        x = self._validate_input(x)
        result = 0.0
        for i in range(self.dimension):
            sum_k = 0.0
            for k in range(self.k_max + 1):
                inner_term = 2 * np.pi * self._b_powers[k] * (x[i] + 0.5)
                sum_k += self._a_powers[k] * np.cos(inner_term)
            result += sum_k
        return float(result - self._constant_term)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Weierstrass function on a batch of points.
        
        Args:
            X (np.ndarray): A batch of points in the search space.
            
        Returns:
            np.ndarray: The function values for each point.
        """
        X = self._validate_batch_input(X)
        n_points, n_dims = X.shape
        results = np.zeros(n_points)
        for p in range(n_points):
            x = X[p]
            result = 0.0
            for i in range(n_dims):
                sum_k = 0.0
                for k in range(self.k_max + 1):
                    inner_term = 2 * np.pi * self._b_powers[k] * (x[i] + 0.5)
                    sum_k += self._a_powers[k] * np.cos(inner_term)
                result += sum_k
            results[p] = result - self._constant_term
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