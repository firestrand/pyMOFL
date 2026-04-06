"""Giunta function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("giunta")
class GiuntaFunction(OptimizationFunction):
    """
    Giunta function (2D).

    f(x) = 0.6 + sum(sin^2(1 - 16*xi/15) - sin(4 - 64*xi/15)/50 - sin(1 - 16*xi/15))

    Global minimum: f(0.4673, 0.4673) ≈ 0.06447
    Bounds: [-1, 1]^2
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Giunta function requires dimension=2")
        default_bounds = Bounds(
            low=np.full(2, -1.0),
            high=np.full(2, 1.0),
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
        t = 16.0 * x / 15.0
        result = 0.6 + np.sum(
            np.sin(1 - t) ** 2 - np.sin(4 - 64.0 * x / 15.0) / 50.0 - np.sin(1 - t)
        )
        return float(result)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        t = 16.0 * X / 15.0
        terms = np.sin(1 - t) ** 2 - np.sin(4 - 64.0 * X / 15.0) / 50.0 - np.sin(1 - t)
        return 0.6 + np.sum(terms, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        point = np.array([0.4673200277395354, 0.4673200277395354])
        return point, float(self.evaluate(point))
