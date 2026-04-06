"""Freudenstein Roth function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("freudenstein_roth")
class FreudensteinRothFunction(OptimizationFunction):
    """
    Freudenstein Roth function (2D).

    f(x) = (x1 - 13 + ((5-x2)*x2 - 2)*x2)^2 + (x1 - 29 + ((x2+1)*x2 - 14)*x2)^2

    Global minimum: f(5, 4) = 0
    Bounds: [-10, 10]^2
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Freudenstein Roth function requires dimension=2")
        default_bounds = Bounds(
            low=np.full(2, -10.0),
            high=np.full(2, 10.0),
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
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        f1 = x1 - 13 + ((5 - x2) * x2 - 2) * x2
        f2 = x1 - 29 + ((x2 + 1) * x2 - 14) * x2
        return float(f1**2 + f2**2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        f1 = x1 - 13 + ((5 - x2) * x2 - 2) * x2
        f2 = x1 - 29 + ((x2 + 1) * x2 - 14) * x2
        return f1**2 + f2**2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([5.0, 4.0]), 0.0
