"""Parsopoulos function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("parsopoulos")
class ParsopoulosFunction(OptimizationFunction):
    """
    Parsopoulos function (2D).

    f(x) = cos(x1)^2 + sin(x2)^2

    Global minimum: f(π/2, 0) = 0
    Bounds: [-5, 5]^2
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Parsopoulos function requires dimension=2")
        default_bounds = Bounds(
            low=np.full(2, -5.0),
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
        return float(np.cos(x[0]) ** 2 + np.sin(x[1]) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        return np.cos(X[:, 0]) ** 2 + np.sin(X[:, 1]) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([np.pi / 2, 0.0]), 0.0
