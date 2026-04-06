"""
Zimmerman benchmark function.

A 2-dimensional constrained test function using penalty-based reformulation.

References
----------
.. [1] Zimmerman, W. (1985). Constrained optimization benchmark.
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


@register("Zimmerman")
@register("zimmerman")
class ZimmermanFunction(OptimizationFunction):
    """
    Zimmerman function (2D).

    Piecewise penalty-based formulation:
        p1 = 9 - x1 - x2
        p2 = (x1-3)^2 + (x2-2)^2 - 16
        p3 = x1*x2 - 14
        penalty(p) = 100*(1+p) if p > 0 else 0
        f(x) = max(p1, penalty(p2), penalty(p3))

    Global minimum: f(7, 2) = 0

    Properties: Continuous, Non-Differentiable, Non-Separable, Non-Scalable
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Zimmerman function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.full(2, 0.0),
            high=np.full(2, 100.0),
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
    def _penalty(p: float) -> float:
        """Compute penalty for constraint violation."""
        if p > 0:
            return 100.0 * (1.0 + p)
        return 0.0

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate Zimmerman function."""
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]

        p1 = 9.0 - x1 - x2
        p2 = (x1 - 3.0) ** 2 + (x2 - 2.0) ** 2 - 16.0
        p3 = x1 * x2 - 14.0

        return float(max(p1, self._penalty(p2), self._penalty(p3)))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate Zimmerman function for batch."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]

        p1 = 9.0 - x1 - x2
        p2 = (x1 - 3.0) ** 2 + (x2 - 2.0) ** 2 - 16.0
        p3 = x1 * x2 - 14.0

        pen2 = np.where(p2 > 0, 100.0 * (1.0 + p2), 0.0)
        pen3 = np.where(p3 > 0, 100.0 * (1.0 + p3), 0.0)

        return np.maximum(np.maximum(p1, pen2), pen3)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([7.0, 2.0]), 0.0
