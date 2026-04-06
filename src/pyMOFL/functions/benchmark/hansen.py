"""Hansen function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("hansen")
class HansenFunction(OptimizationFunction):
    """
    Hansen function (2D).

    f(x) = [sum_{i=0}^{4} (i+1)*cos(i*x1 + i + 1)] * [sum_{j=0}^{4} (j+1)*cos((j+2)*x2 + j + 1)]

    Global minimum: f(-7.58989, -7.70831) ≈ -176.5418
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
            raise ValueError("Hansen function requires dimension=2")
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
        i = np.arange(5)
        sum1 = np.sum((i + 1) * np.cos(i * x1 + i + 1))
        sum2 = np.sum((i + 1) * np.cos((i + 2) * x2 + i + 1))
        return float(sum1 * sum2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        i = np.arange(5)
        # Shape: (batch, 5)
        sum1 = np.sum((i + 1) * np.cos(np.outer(x1, i) + i + 1), axis=1)
        sum2 = np.sum((i + 1) * np.cos(np.outer(x2, i + 2) + i + 1), axis=1)
        return sum1 * sum2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        point = np.array([-7.58989583, -7.70831466])
        return point, float(self.evaluate(point))
