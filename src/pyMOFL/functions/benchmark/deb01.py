"""
Deb's First function (Deb01) implementation.

Deb01 is a highly multimodal function with many regularly distributed local minima.

References
----------
.. [1] Deb, K. (2001). "Multi-Objective Optimization using Evolutionary Algorithms."
       John Wiley & Sons.
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


@register("Deb01")
@register("deb01")
class Deb01Function(OptimizationFunction):
    """
    Deb's First function (Deb01).

    f(x) = -(1/D) * sum(sin(5*pi*x_i)^6)

    Properties: Continuous, Differentiable, Separable, Scalable, Multimodal
    Domain: [-1, 1]^D
    Global minimum: f* = -1 (at multiple points where sin(5*pi*x_i) = +/-1)
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -1.0),
            high=np.full(dimension, 1.0),
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
        """Compute the Deb01 function value."""
        x = self._validate_input(x)
        return float(-(1.0 / self.dimension) * np.sum(np.sin(5.0 * np.pi * x) ** 6))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Deb01 function for a batch of points."""
        X = self._validate_batch_input(X)
        return -(1.0 / self.dimension) * np.sum(np.sin(5.0 * np.pi * X) ** 6, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Deb01 function.

        The global minimum of -1 is achieved when sin(5*pi*x_i)^6 = 1 for all i,
        i.e., when 5*pi*x_i = pi/2 + k*pi, giving x_i = (2k+1)/10.
        We use x_i = 0.1 (k=0) as the canonical optimum.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.full(self.dimension, 0.1), -1.0
