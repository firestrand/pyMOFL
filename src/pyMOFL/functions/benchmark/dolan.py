"""
Dolan benchmark function.

A 5-dimensional test function with mixed trigonometric and polynomial terms.

References
----------
.. [1] Dolan, E.D. & More, J.J. (2002). "Benchmarking optimization software with performance
       profiles". Mathematical Programming, 91(2), 201-213.
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


@register("Dolan")
@register("dolan")
class DolanFunction(OptimizationFunction):
    """
    Dolan function (5D).

    Mathematical definition:
        f(x) = |(x1 + 1.7*x2)*sin(x1) - 1.5*x3 - 0.1*x4*cos(x4 + x5 - x1)
                + 0.2*x5^2 - x2 - 1|

    Global minimum: f(8.39, 4.81, 7.35, 68.88, 3.85) ~ 0 (approximately)

    Properties: Continuous, Non-Differentiable, Non-Separable, Non-Scalable, Multimodal
    """

    def __init__(
        self,
        dimension: int = 5,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 5:
            raise ValueError("Dolan function is Non-Scalable and requires dimension=5")

        default_bounds = Bounds(
            low=np.full(5, -100.0),
            high=np.full(5, 100.0),
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
        """Evaluate Dolan function."""
        x = self._validate_input(x)
        x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
        inner = (
            (x1 + 1.7 * x2) * np.sin(x1)
            - 1.5 * x3
            - 0.1 * x4 * np.cos(x4 + x5 - x1)
            + 0.2 * x5**2
            - x2
            - 1.0
        )
        return float(np.abs(inner))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Dolan function for batch."""
        X = self._validate_batch_input(X)
        x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        inner = (
            (x1 + 1.7 * x2) * np.sin(x1)
            - 1.5 * x3
            - 0.1 * x4 * np.cos(x4 + x5 - x1)
            + 0.2 * x5**2
            - x2
            - 1.0
        )
        return np.abs(inner)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        min_point = np.array([8.39, 4.81, 7.35, 68.88, 3.85])
        min_value = self.evaluate(min_point)
        return min_point, min_value
