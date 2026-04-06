"""
Quintic function implementation.

The Quintic function is a multimodal function with multiple global minima.

References
----------
.. [1] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
       optimization problems". International Journal of Mathematical Modelling and Numerical
       Optimisation, 4(2), 150-194.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Quintic")
@register("quintic")
class QuinticFunction(OptimizationFunction):
    """
    Quintic function.

    f(x) = sum(|x_i^5 - 3*x_i^4 + 4*x_i^3 + 2*x_i^2 - 10*x_i - 4|)

    Properties: Continuous, Differentiable, Separable, Scalable, Multimodal
    Domain: [-10, 10]^D
    Global minimum: f(-1, ..., -1) = 0 or f(2, ..., 2) = 0
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -10.0),
            high=np.full(dimension, 10.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the Quintic function value."""
        x = self._validate_input(x)
        terms = x**5 - 3.0 * x**4 + 4.0 * x**3 + 2.0 * x**2 - 10.0 * x - 4.0
        return float(np.sum(np.abs(terms)))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Quintic function for a batch of points."""
        X = self._validate_batch_input(X)
        terms = X**5 - 3.0 * X**4 + 4.0 * X**3 + 2.0 * X**2 - 10.0 * X - 4.0
        return np.sum(np.abs(terms), axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Quintic function.

        The function has two global minima at x = (-1, ..., -1) and x = (2, ..., 2),
        both with f* = 0. We return (-1, ..., -1) as the canonical optimum.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.full(self.dimension, -1.0), 0.0
