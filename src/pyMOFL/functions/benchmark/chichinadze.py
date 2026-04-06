"""Chichinadze function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("chichinadze")
class ChichinadzeFunction(OptimizationFunction):
    """
    Chichinadze function (2D).

    f(x) = x1^2 - 12*x1 + 8*sin(5*pi*x1/2) + 10*cos(pi*x1/2) + 11
           - 0.2*sqrt(5)/exp(0.5*(x2-0.5)^2)

    Global minimum: f(6.189866, 0.5) ≈ -42.9444
    Bounds: [-30, 30]^2
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Chichinadze function requires dimension=2")
        default_bounds = Bounds(
            low=np.full(2, -30.0),
            high=np.full(2, 30.0),
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
            x1**2
            - 12 * x1
            + 8 * np.sin(5 * np.pi * x1 / 2)
            + 10 * np.cos(np.pi * x1 / 2)
            + 11
            - 0.2 * np.sqrt(5) / np.exp(0.5 * (x2 - 0.5) ** 2)
        )

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return (
            x1**2
            - 12 * x1
            + 8 * np.sin(5 * np.pi * x1 / 2)
            + 10 * np.cos(np.pi * x1 / 2)
            + 11
            - 0.2 * np.sqrt(5) / np.exp(0.5 * (x2 - 0.5) ** 2)
        )

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        point = np.array([6.189866, 0.5])
        return point, float(self.evaluate(point))
