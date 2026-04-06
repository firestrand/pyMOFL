"""Holder Table benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("HolderTable")
@register("holder_table")
class HolderTableFunction(OptimizationFunction):
    """
    Holder Table function (2D).

    f(x) = -|sin(x1)*cos(x2)*exp(|1 - sqrt(x1^2+x2^2)/pi|)|

    Four symmetric global minima at (+-8.05502, +-9.66459) with f* ~ -19.2085.

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
            raise ValueError("Holder Table function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-10.0, -10.0]),
            high=np.array([10.0, 10.0]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
        )

    def _compute_single(self, x1: float, x2: float) -> float:
        inner = np.abs(1.0 - np.sqrt(x1**2 + x2**2) / np.pi)
        return float(-np.abs(np.sin(x1) * np.cos(x2) * np.exp(inner)))

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        return self._compute_single(x[0], x[1])

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        inner = np.abs(1.0 - np.sqrt(x1**2 + x2**2) / np.pi)
        return -np.abs(np.sin(x1) * np.cos(x2) * np.exp(inner))

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        min_point = np.array([8.05502, 9.66459])
        min_value = self._compute_single(min_point[0], min_point[1])
        return min_point, min_value
