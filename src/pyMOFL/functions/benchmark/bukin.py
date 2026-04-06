"""Bukin N.6 benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Bukin6")
@register("bukin6")
class Bukin6Function(OptimizationFunction):
    """
    Bukin N.6 function (2D).

    f(x) = 100 * sqrt(|x2 - 0.01*x1^2|) + 0.01 * |x1 + 10|

    Global minimum: f(-10, 1) = 0

    Domain: x1 in [-15, -5], x2 in [-3, 3]

    References
    ----------
    .. [1] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions
           for global optimization problems".
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Bukin N.6 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-15.0, -3.0]),
            high=np.array([-5.0, 3.0]),
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
        x1, x2 = x[0], x[1]
        return float(100.0 * np.sqrt(np.abs(x2 - 0.01 * x1**2)) + 0.01 * np.abs(x1 + 10.0))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return 100.0 * np.sqrt(np.abs(x2 - 0.01 * x1**2)) + 0.01 * np.abs(x1 + 10.0)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([-10.0, 1.0]), 0.0
