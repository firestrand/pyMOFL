"""
Rosenbrock function implementation.

The Rosenbrock function is a non-convex unimodal benchmark function.
It has a narrow, curved valley which is difficult for many optimization algorithms to navigate.

References
----------
.. [1] Rosenbrock, H.H. (1960). "An automatic method for finding the greatest or least value of a function".
       The Computer Journal, 3(3), 175-184.
.. [2] Adorio, E.P., & Diliman, U.P. (2005). MVF - "Multivariate Test Functions Library in C for 
       Unconstrained Global Optimization", 2005.
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.base import register_function

@register_function("Rosenbrock")
class RosenbrockFunction(OptimizationFunction):
    """
    Rosenbrock function.

    The function is defined as:
        f(x) = sum_{i=1}^{n-1} [100*(x_{i}^2 - x_{i+1})^2 + (x_i - 1)^2]

    Global minimum: f(1, 1, ..., 1) = 0

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for random initialization. If None, defaults to [-30, 30]^d.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. If None, defaults to [-30, 30]^d.

    References
    ----------
    .. [1] Rosenbrock, H.H. (1960). "An automatic method for finding the greatest or least value of a function".
           The Computer Journal, 3(3), 175-184.
    .. [2] Adorio, E.P., & Diliman, U.P. (2005). MVF - "Multivariate Test Functions Library in C for 
           Unconstrained Global Optimization", 2005.
    """
    def __init__(self, dimension: int,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None):
        default_init_bounds = Bounds(
            low=np.full(dimension, -30.0),
            high=np.full(dimension, 30.0),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        default_oper_bounds = Bounds(
            low=np.full(dimension, -30.0),
            high=np.full(dimension, 30.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_init_bounds,
            operational_bounds=operational_bounds or default_oper_bounds
        )

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Rosenbrock function at a single point.

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
        term1 = 100 * (x[:-1] ** 2 - x[1:]) ** 2
        term2 = (x[:-1] - 1) ** 2
        return float(np.sum(term1 + term2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Rosenbrock function.

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
        term1 = 100 * (X[:, :-1] ** 2 - X[:, 1:]) ** 2
        term2 = (X[:, :-1] - 1) ** 2
        return np.sum(term1 + term2, axis=1)

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