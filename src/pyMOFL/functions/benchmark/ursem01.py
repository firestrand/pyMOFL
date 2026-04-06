"""Ursem01 function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("ursem01")
class Ursem01Function(OptimizationFunction):
    """
    Ursem01 function (2D).

    f(x) = -sin(2*x1 - pi/2) - 3*cos(x2) - x1/2

    Global minimum: f(1.69714, 0) ≈ -4.8168
    Bounds: x1 in [-2.5, 3], x2 in [-2, 2]
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Ursem01 function requires dimension=2")
        default_bounds = Bounds(
            low=np.array([-2.5, -2.0]),
            high=np.array([3.0, 2.0]),
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
        return float(-np.sin(2 * x1 - np.pi / 2) - 3 * np.cos(x2) - x1 / 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return -np.sin(2 * x1 - np.pi / 2) - 3 * np.cos(x2) - x1 / 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        point = np.array([1.69714, 0.0])
        return point, float(self.evaluate(point))
