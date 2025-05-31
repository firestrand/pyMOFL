"""
Griewank function implementation.

The Griewank function is a non-convex multimodal benchmark function with many local minima.
The product term in the function creates correlation between variables, making it non-separable.

References
----------
.. [1] Griewank, A. O. (1981). "Generalized descent for global optimization". 
       Journal of Optimization Theory and Applications, 34(1), 11-39.
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

@register_function("Griewank")
class GriewankFunction(OptimizationFunction):
    """
    Griewank function.

    The function is defined as:
        f(x) = sum(x_i^2/4000) - prod(cos(x_i/sqrt(i))) + 1

    Global minimum: f(0, 0, ..., 0) = 0

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for random initialization. If None, defaults to [-600, 600]^d.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. If None, defaults to [-600, 600]^d.

    References
    ----------
    .. [1] Griewank, A. O. (1981). "Generalized descent for global optimization". 
           Journal of Optimization Theory and Applications, 34(1), 11-39.
    .. [2] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    """
    def __init__(self, dimension: int,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None):
        default_init_bounds = Bounds(
            low=np.full(dimension, -600.0),
            high=np.full(dimension, 600.0),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        default_oper_bounds = Bounds(
            low=np.full(dimension, -600.0),
            high=np.full(dimension, 600.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_init_bounds,
            operational_bounds=operational_bounds or default_oper_bounds
        )
        self._inv_sqrt = 1.0 / np.sqrt(np.arange(1, dimension + 1, dtype=np.float64))

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Griewank function at a single point.

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
        s = np.dot(x, x)
        p = np.cos(x * self._inv_sqrt).prod()
        return 1.0 + s * 0.00025 - p

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Griewank function.

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
        s = np.sum(X ** 2, axis=1)
        p = np.prod(np.cos(X * self._inv_sqrt), axis=1)
        return 1.0 + s * 0.00025 - p

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