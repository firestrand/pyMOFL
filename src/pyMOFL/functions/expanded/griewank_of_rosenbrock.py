"""
Griewank of Rosenbrock (F8F2) expanded function implementation.

This function is created by composing Griewank function with Rosenbrock function,
commonly used in CEC benchmark suites.
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("GriewankOfRosenbrock")
@register("griewankOfRosenbrock")
class GriewankOfRosenbrock(OptimizationFunction):
    """
    Griewank of Rosenbrock (F8F2) expanded function.
    
    This function applies Griewank function to pairs of variables transformed
    by the Rosenbrock function. It's commonly used in CEC benchmarks as F8F2.
    
    The function is defined as:
        F8F2(x) = sum(F8(F2(x_i, x_{i+1}))) for i = 1 to D-1
                + F8(F2(x_D, x_1))
    
    where:
        F2(x, y) = 100 * (x^2 - y)^2 + (x - 1)^2  (Rosenbrock)
        F8(z) = z^2/4000 - cos(z) + 1              (Griewank)
    
    Global minimum: f(1, 1, ..., 1) = 0
    
    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, defaults to [-5, 5]^d.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, defaults to [-5, 5]^d.
    """
    
    def __init__(self, dimension: int,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None):
        default_init_bounds = Bounds(
            low=np.full(dimension, -5.0),
            high=np.full(dimension, 5.0),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        default_oper_bounds = Bounds(
            low=np.full(dimension, -5.0),
            high=np.full(dimension, 5.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_init_bounds,
            operational_bounds=operational_bounds or default_oper_bounds
        )
    
    def _rosenbrock(self, x: float, y: float) -> float:
        """Apply Rosenbrock function F2 to a pair of variables."""
        return 100.0 * (x**2 - y)**2 + (x - 1.0)**2
    
    def _griewank(self, z: float) -> float:
        """Apply Griewank function F8 to a single value."""
        return z**2 / 4000.0 - np.cos(z) + 1.0
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Griewank of Rosenbrock function at a single point.
        
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
        # Apply F8(F2) to consecutive pairs
        for i in range(self.dimension - 1):
            rosenbrock_val = self._rosenbrock(x[i], x[i + 1])
            total += self._griewank(rosenbrock_val)
        
        # Apply F8(F2) to the last and first variables (circular)
        rosenbrock_val = self._rosenbrock(x[-1], x[0])
        total += self._griewank(rosenbrock_val)
        
        return float(total)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Griewank of Rosenbrock function.
        
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
        
        # Apply F8(F2) to consecutive pairs
        for i in range(self.dimension - 1):
            rosenbrock_vals = 100.0 * (X[:, i]**2 - X[:, i + 1])**2 + (X[:, i] - 1.0)**2
            result += rosenbrock_vals**2 / 4000.0 - np.cos(rosenbrock_vals) + 1.0
        
        # Apply F8(F2) to the last and first variables (circular)
        rosenbrock_vals = 100.0 * (X[:, -1]**2 - X[:, 0])**2 + (X[:, -1] - 1.0)**2
        result += rosenbrock_vals**2 / 4000.0 - np.cos(rosenbrock_vals) + 1.0
        
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
        global_min_point = np.ones(dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value