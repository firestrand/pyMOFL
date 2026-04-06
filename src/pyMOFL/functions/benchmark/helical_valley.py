"""
Helical Valley (Fletcher-Powell) benchmark function.

A 3-dimensional test function with a helical-shaped valley.

References
----------
.. [1] Fletcher, R. & Powell, M.J.D. (1963). "A rapidly convergent descent method for
       minimization". The Computer Journal, 6(2), 163-168.
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


@register("HelicalValley")
@register("helical_valley")
class HelicalValleyFunction(OptimizationFunction):
    """
    Helical Valley (Fletcher-Powell) function (3D).

    Mathematical definition:
        f(x) = 100*((x3 - 10*theta)^2 + (sqrt(x1^2+x2^2) - 1)^2) + x3^2
        theta = (1/(2*pi)) * arctan2(x2, x1)

    Global minimum: f(1, 0, 0) = 0

    Properties: Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal
    """

    def __init__(
        self,
        dimension: int = 3,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 3:
            raise ValueError("Helical Valley function is Non-Scalable and requires dimension=3")

        default_bounds = Bounds(
            low=np.full(3, -10.0),
            high=np.full(3, 10.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    @staticmethod
    def _theta(x1: float, x2: float) -> float:
        """Compute theta using atan2 for proper quadrant handling."""
        return np.arctan2(x2, x1) / (2.0 * np.pi)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate Helical Valley function."""
        x = self._validate_input(x)
        x1, x2, x3 = x[0], x[1], x[2]
        theta = self._theta(x1, x2)
        r = np.sqrt(x1**2 + x2**2)
        return float(100.0 * ((x3 - 10.0 * theta) ** 2 + (r - 1.0) ** 2) + x3**2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Helical Valley function for batch."""
        X = self._validate_batch_input(X)
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        theta = np.arctan2(x2, x1) / (2.0 * np.pi)
        r = np.sqrt(x1**2 + x2**2)
        return 100.0 * ((x3 - 10.0 * theta) ** 2 + (r - 1.0) ** 2) + x3**2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([1.0, 0.0, 0.0]), 0.0
