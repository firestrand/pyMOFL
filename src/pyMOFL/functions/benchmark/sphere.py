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

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Sphere")
@register("sphere")
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

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        # Sensible defaults: [-100, 100] for each dimension
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the sphere function value."""
        x = self._validate_input(x)
        return float(np.sum(x**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute sphere function for batch."""
        X = self._validate_batch_input(X)
        return np.sum(X**2, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """
        Get the global minimum point and value for the Sphere function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        global_min_point = np.zeros(self.dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value
