"""Zettl function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("zettl")
class ZettlFunction(OptimizationFunction):
    """
    Zettl function (2D).

    f(x) = x1/4 + (x1^2 - 2*x1 + x2^2)^2

    Global minimum: f(-0.02990, 0) ≈ -0.003791
    Bounds: [-1, 5]^2
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Zettl function requires dimension=2")
        default_bounds = Bounds(
            low=np.full(2, -1.0),
            high=np.full(2, 5.0),
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
        return float(x1 / 4 + (x1**2 - 2 * x1 + x2**2) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return x1 / 4 + (x1**2 - 2 * x1 + x2**2) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        point = np.array([-0.02990, 0.0])
        return point, float(self.evaluate(point))
