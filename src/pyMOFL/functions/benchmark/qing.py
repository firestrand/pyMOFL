"""Qing benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Qing")
@register("qing")
class QingFunction(OptimizationFunction):
    """
    Qing function.

    f(x) = sum((x_i^2 - i)^2)

    Global minimum: f(sqrt(1), sqrt(2), ..., sqrt(D)) = 0

    References
    ----------
    .. [1] Qing, A. (2006). "Dynamic differential evolution strategy and applications in
           electromagnetic inverse scattering problems".
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
                low=np.full(dimension, -500.0),
                high=np.full(dimension, 500.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -500.0),
                high=np.full(dimension, 500.0),
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
        return float(np.sum((x**2 - i) ** 2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        i = np.arange(1, self.dimension + 1, dtype=float)
        return np.sum((X**2 - i) ** 2, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        x_star = np.sqrt(np.arange(1, self.dimension + 1, dtype=float))
        return x_star, 0.0
