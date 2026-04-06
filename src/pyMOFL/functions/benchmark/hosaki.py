"""Hosaki function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("hosaki")
class HosakiFunction(OptimizationFunction):
    """
    Hosaki function (2D).

    f(x) = (1 - 8*x1 + 7*x1^2 - 7*x1^3/3 + x1^4/4) * x2^2 * exp(-x2)

    Global minimum: f(4, 2) ≈ -2.3458
    Bounds: [0, 10]^2
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Hosaki function requires dimension=2")
        default_bounds = Bounds(
            low=np.full(2, 0.0),
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
        return float((1 - 8 * x1 + 7 * x1**2 - 7 * x1**3 / 3 + x1**4 / 4) * x2**2 * np.exp(-x2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return (1 - 8 * x1 + 7 * x1**2 - 7 * x1**3 / 3 + x1**4 / 4) * x2**2 * np.exp(-x2)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        point = np.array([4.0, 2.0])
        return point, float(self.evaluate(point))
