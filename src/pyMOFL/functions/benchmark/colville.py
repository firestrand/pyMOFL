"""Colville benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Colville")
@register("colville")
class ColvilleFunction(OptimizationFunction):
    """
    Colville function (4D).

    f(x) = 100*(x1^2 - x2)^2 + (x1 - 1)^2 + (x3 - 1)^2 + 90*(x3^2 - x4)^2
           + 10.1*((x2 - 1)^2 + (x4 - 1)^2) + 19.8*(x2 - 1)*(x4 - 1)

    Global minimum: f(1, 1, 1, 1) = 0

    References
    ----------
    .. [1] Colville, A.R. (1968). "A comparative study of nonlinear programming codes".
    """

    def __init__(
        self,
        dimension: int = 4,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 4:
            raise ValueError("Colville function is Non-Scalable and requires dimension=4")

        default_bounds = Bounds(
            low=np.full(4, -10.0),
            high=np.full(4, 10.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
        )

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        return float(
            100.0 * (x1**2 - x2) ** 2
            + (x1 - 1.0) ** 2
            + (x3 - 1.0) ** 2
            + 90.0 * (x3**2 - x4) ** 2
            + 10.1 * ((x2 - 1.0) ** 2 + (x4 - 1.0) ** 2)
            + 19.8 * (x2 - 1.0) * (x4 - 1.0)
        )

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        return (
            100.0 * (x1**2 - x2) ** 2
            + (x1 - 1.0) ** 2
            + (x3 - 1.0) ** 2
            + 90.0 * (x3**2 - x4) ** 2
            + 10.1 * ((x2 - 1.0) ** 2 + (x4 - 1.0) ** 2)
            + 19.8 * (x2 - 1.0) * (x4 - 1.0)
        )

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([1.0, 1.0, 1.0, 1.0]), 0.0
