"""
Expanded Schaffer F6 function implementation.

This function expands the 2D Schaffer F6 function to higher dimensions
by applying it to consecutive pairs of variables.
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("SchafferF6Expanded")
@register("schafferF6Expanded")
class SchafferF6Expanded(OptimizationFunction):
    """
    Expanded Schaffer F6 function.
    
    This function applies the 2D Schaffer F6 function to consecutive pairs
    of variables and sums the results. It's commonly used in CEC benchmarks.
    
    The function is defined as:
        f(x) = sum(F6(x_i, x_{i+1})) for i = 1 to D-1
             + F6(x_D, x_1)
    
    where:
        F6(x, y) = 0.5 + (sin^2(sqrt(x^2 + y^2)) - 0.5) / (1 + 0.001 * (x^2 + y^2))^2
    
    Global minimum: f(0, 0, ..., 0) = 0
    
    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, defaults to [-100, 100]^d.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, defaults to [-100, 100]^d.
    """
    
    def __init__(self, dimension: int,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None):
        default_init_bounds = Bounds(
            low=np.full(dimension, -100.0),
            high=np.full(dimension, 100.0),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        default_oper_bounds = Bounds(
            low=np.full(dimension, -100.0),
            high=np.full(dimension, 100.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_init_bounds,
            operational_bounds=operational_bounds or default_oper_bounds
        )
    
    def _schaffer_f6(self, x: float, y: float) -> float:
        """Apply Schaffer F6 function to a pair of variables."""
        sum_squares = x**2 + y**2
        numerator = np.sin(np.sqrt(sum_squares))**2 - 0.5
        denominator = (1 + 0.001 * sum_squares)**2
        return 0.5 + numerator / denominator
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the expanded Schaffer F6 function at a single point.
        
        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (dimension,).
            
        Returns
        -------
        float
            Function value at x.
        """
        x = self._validate_input(x)
        
        total = 0.0
        # Apply Schaffer F6 to consecutive pairs
        for i in range(self.dimension - 1):
            total += self._schaffer_f6(x[i], x[i + 1])
        
        # Apply Schaffer F6 to the last and first variables (circular)
        total += self._schaffer_f6(x[-1], x[0])
        
        return float(total)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the expanded Schaffer F6 function.
        
        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_points, dimension).
            
        Returns
        -------
        np.ndarray
            Function values of shape (n_points,).
        """
        X = self._validate_batch_input(X)
        n_points = X.shape[0]
        
        result = np.zeros(n_points)
        
        # Apply Schaffer F6 to consecutive pairs
        for i in range(self.dimension - 1):
            sum_squares = X[:, i]**2 + X[:, i + 1]**2
            numerator = np.sin(np.sqrt(sum_squares))**2 - 0.5
            denominator = (1 + 0.001 * sum_squares)**2
            result += 0.5 + numerator / denominator
        
        # Apply Schaffer F6 to the last and first variables (circular)
        sum_squares = X[:, -1]**2 + X[:, 0]**2
        numerator = np.sin(np.sqrt(sum_squares))**2 - 0.5
        denominator = (1 + 0.001 * sum_squares)**2
        result += 0.5 + numerator / denominator
        
        return result
    
    @property
    def bounds(self) -> np.ndarray:
        """
        Legacy-compatible property for test and composite compatibility.
        Returns the operational bounds as a (dimension, 2) array [[low, high], ...].
        Prefer using .operational_bounds for new code.
        """
        return np.stack([self.operational_bounds.low, self.operational_bounds.high], axis=1)
    
    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """
        Get the global minimum of the function.
        
        Parameters
        ----------
        dimension : int
            The dimension of the function.
            
        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        global_min_point = np.zeros(dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value