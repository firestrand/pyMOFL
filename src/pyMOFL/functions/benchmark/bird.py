"""Bird function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("bird")
class BirdFunction(OptimizationFunction):
    """
    Bird function (2D).

    f(x) = sin(x1)*exp((1-cos(x2))^2) + cos(x2)*exp((1-sin(x1))^2) + (x1-x2)^2

    Global minimum: f(4.70104, 3.15294) ≈ -106.764537
    Bounds: [-2π, 2π]^2
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Bird function requires dimension=2")
        default_bounds = Bounds(
            low=np.full(2, -2 * np.pi),
            high=np.full(2, 2 * np.pi),
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
        return float(
            np.sin(x1) * np.exp((1 - np.cos(x2)) ** 2)
            + np.cos(x2) * np.exp((1 - np.sin(x1)) ** 2)
            + (x1 - x2) ** 2
        )

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return (
            np.sin(x1) * np.exp((1 - np.cos(x2)) ** 2)
            + np.cos(x2) * np.exp((1 - np.sin(x1)) ** 2)
            + (x1 - x2) ** 2
        )

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        point = np.array([4.70104, 3.15294])
        return point, float(self.evaluate(point))
