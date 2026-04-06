"""Damavandi function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("damavandi")
class DamavandiFunction(OptimizationFunction):
    """
    Damavandi function (2D).

    f(x) = (1 - |sinc(pi*(x1-2)) * sinc(pi*(x2-2))|^5) * (2 + (x1-7)^2 + 2*(x2-7)^2)

    where sinc(t) = sin(t)/t

    Global minimum: f(2, 2) = 0
    Bounds: [0, 14]^2
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Damavandi function requires dimension=2")
        default_bounds = Bounds(
            low=np.full(2, 0.0),
            high=np.full(2, 14.0),
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
        sinc1 = np.sinc(x1 - 2)  # np.sinc(x) = sin(pi*x)/(pi*x)
        sinc2 = np.sinc(x2 - 2)
        factor1 = 1 - np.abs(sinc1 * sinc2) ** 5
        factor2 = 2 + (x1 - 7) ** 2 + 2 * (x2 - 7) ** 2
        return float(factor1 * factor2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        sinc1 = np.sinc(x1 - 2)
        sinc2 = np.sinc(x2 - 2)
        factor1 = 1 - np.abs(sinc1 * sinc2) ** 5
        factor2 = 2 + (x1 - 7) ** 2 + 2 * (x2 - 7) ** 2
        return factor1 * factor2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([2.0, 2.0]), 0.0
