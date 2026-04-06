"""Cross-Leg Table function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("cross_leg_table")
class CrossLegTableFunction(OptimizationFunction):
    """
    Cross-Leg Table function (2D).

    f(x) = -1 / (|sin(x1)*sin(x2)*exp(|100 - sqrt(x1^2+x2^2)/pi|)| + 1)^0.1

    Global minimum: f(0, 0) = -1
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
            raise ValueError("Cross-Leg Table function requires dimension=2")
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
        inner = np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi)
        val = np.abs(np.sin(x1) * np.sin(x2) * np.exp(inner))
        return float(-1.0 / (val + 1) ** 0.1)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        inner = np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi)
        val = np.abs(np.sin(x1) * np.sin(x2) * np.exp(inner))
        return -1.0 / (val + 1) ** 0.1

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([0.0, 0.0]), -1.0
