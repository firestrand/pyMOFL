"""Eggholder benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Eggholder")
@register("eggholder")
class EggholderFunction(OptimizationFunction):
    """
    Eggholder function (2D).

    f(x) = -(x2+47)*sin(sqrt(|x2 + x1/2 + 47|)) - x1*sin(sqrt(|x1 - (x2+47)|))

    Global minimum: f(512, 404.2319) ~ -959.6407

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
            raise ValueError("Eggholder function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-512.0, -512.0]),
            high=np.array([512.0, 512.0]),
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
        return float(
            -(x2 + 47.0) * np.sin(np.sqrt(np.abs(x2 + x1 / 2.0 + 47.0)))
            - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47.0))))
        )

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return -(x2 + 47.0) * np.sin(np.sqrt(np.abs(x2 + x1 / 2.0 + 47.0))) - x1 * np.sin(
            np.sqrt(np.abs(x1 - (x2 + 47.0)))
        )

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        min_point = np.array([512.0, 404.2319])
        min_value = self.evaluate(min_point)
        return min_point, min_value
