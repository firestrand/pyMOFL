"""
Sphere function implementation.

The Sphere function is one of the simplest unimodal benchmark functions.
It is continuous, convex, and differentiable.

References
----------
.. [1] Adorio, E.P., & Diliman, U.P. (2005). MVF - "Multivariate Test Functions Library in C for 
       Unconstrained Global Optimization", 2005.
.. [2] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global 
       optimization problems". International Journal of Mathematical Modelling and Numerical 
       Optimisation, 4(2), 150-194.
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.base import register_function

@register_function("Sphere")
class SphereFunction(OptimizationFunction):
    """
    Sphere function.

    The Sphere function is defined as:
        f(x) = sum(x_i^2)
    where x is a d-dimensional vector.

    Global minimum: f(0, 0, ..., 0) = 0

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for random initialization. If None, defaults to [-100, 100]^d.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. If None, defaults to [-100, 100]^d.

    References
    ----------
    .. [1] Adorio, E.P., & Diliman, U.P. (2005). MVF - "Multivariate Test Functions Library in C for 
           Unconstrained Global Optimization", 2005.
    .. [2] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global 
           optimization problems". International Journal of Mathematical Modelling and Numerical 
           Optimisation, 4(2), 150-194.
    """
    def __init__(self, dimension: int,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None):
        # Sensible defaults: [-100, 100] for each dimension
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

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Sphere function at a single point.

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
        return float(np.sum(x**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Sphere function.

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
        return np.sum(X**2, axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """
        Get the global minimum point and value for the Sphere function.

        Parameters
        ----------
        dimension : int
            The dimensionality of the function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        global_min_point = np.zeros(dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value