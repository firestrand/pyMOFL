"""Leon function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("leon")
class LeonFunction(OptimizationFunction):
    """
    Leon function (2D).

    f(x) = (1 - x1)^2 + 100*(x2 - x1^2)^2

    Note: This is equivalent to Rosenbrock in 2D.
    Global minimum: f(1, 1) = 0
    Bounds: [-1.2, 1.2]^2
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Leon function requires dimension=2")
        default_bounds = Bounds(
            low=np.full(2, -1.2),
            high=np.full(2, 1.2),
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
        return float((1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([1.0, 1.0]), 0.0
