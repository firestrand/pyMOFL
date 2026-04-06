"""El-Attar-Vidyasagar-Dutta function implementation."""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("el_attar")
class ElAttarVidyasagarDuttaFunction(OptimizationFunction):
    """
    El-Attar-Vidyasagar-Dutta function (2D).

    f(x) = (x1^2 + x2 - 10)^2 + (x1 + x2^2 - 7)^2 + (x1^2 + x2^3 - 1)^2

    Global minimum: f(3.4091868222, -2.1714330361) ≈ 1.7128
    Bounds: [-100, 100]^2
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("El-Attar function requires dimension=2")
        default_bounds = Bounds(
            low=np.full(2, -100.0),
            high=np.full(2, 100.0),
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
        return float((x1**2 + x2 - 10) ** 2 + (x1 + x2**2 - 7) ** 2 + (x1**2 + x2**3 - 1) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return (x1**2 + x2 - 10) ** 2 + (x1 + x2**2 - 7) ** 2 + (x1**2 + x2**3 - 1) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        point = np.array([3.4091868222, -2.1714330361])
        return point, float(self.evaluate(point))
