"""Quartic (De Jong 4) benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Quartic")
@register("quartic")
class QuarticFunction(OptimizationFunction):
    """
    Quartic (De Jong 4) function, without noise.

    f(x) = sum(i * x_i^4)

    Global minimum: f(0, ..., 0) = 0

    References
    ----------
    .. [1] De Jong, K.A. (1975). "An analysis of the behavior of a class of genetic
           adaptive systems".
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -1.28),
                high=np.full(dimension, 1.28),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -1.28),
                high=np.full(dimension, 1.28),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        i = np.arange(1, self.dimension + 1, dtype=float)
        return float(np.sum(i * x**4))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        i = np.arange(1, self.dimension + 1, dtype=float)
        return np.sum(i * X**4, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.zeros(self.dimension), 0.0
