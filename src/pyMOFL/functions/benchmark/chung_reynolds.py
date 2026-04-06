"""Chung-Reynolds benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("ChungReynolds")
@register("chung_reynolds")
class ChungReynoldsFunction(OptimizationFunction):
    """
    Chung-Reynolds function.

    f(x) = (sum(x_i^2))^2

    Global minimum: f(0, ..., 0) = 0

    References
    ----------
    .. [1] Chung, C.J. & Reynolds, R.G. (1996). "A testbed for solving optimization
           problems using cultural algorithms".
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
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
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
        return float(np.sum(x**2) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        return np.sum(X**2, axis=1) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.zeros(self.dimension), 0.0
