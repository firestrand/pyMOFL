"""
Deb's Third function (Deb03) implementation.

Deb03 is a variant of Deb01 with a nonlinear argument transformation,
creating an irregularly spaced multimodal landscape.

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


@register("Deb03")
@register("deb03")
class Deb03Function(OptimizationFunction):
    """
    Deb's Third function (Deb03).

    f(x) = -(1/D) * sum(sin(5*pi*(x_i^0.75 - 0.05))^6)

    Properties: Continuous, Differentiable, Separable, Scalable, Multimodal
    Domain: [0, 1]^D
    Global minimum: f* = -1
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, 0.0),
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
        """Compute the Deb03 function value."""
        x = self._validate_input(x)
        return float(-(1.0 / self.dimension) * np.sum(np.sin(5.0 * np.pi * (x**0.75 - 0.05)) ** 6))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Deb03 function for a batch of points."""
        X = self._validate_batch_input(X)
        return -(1.0 / self.dimension) * np.sum(np.sin(5.0 * np.pi * (X**0.75 - 0.05)) ** 6, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Deb03 function.

        The minimum of -1 is achieved when sin(5*pi*(x_i^0.75 - 0.05))^6 = 1
        for all i, i.e., when 5*pi*(x_i^0.75 - 0.05) = pi/2 + k*pi.
        For k=0: x_i^0.75 = 0.15, so x_i = 0.15^(4/3).

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        # 5*pi*(x^0.75 - 0.05) = pi/2 => x^0.75 = 0.15 => x = 0.15^(4/3)
        x_opt = 0.15 ** (4.0 / 3.0)
        return np.full(self.dimension, x_opt), -1.0
