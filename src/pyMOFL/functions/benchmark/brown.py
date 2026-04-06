"""Brown benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Brown")
@register("brown")
class BrownFunction(OptimizationFunction):
    """
    Brown function.

    f(x) = sum((x_i^2)^(x_{i+1}^2 + 1) + (x_{i+1}^2)^(x_i^2 + 1))

    Global minimum: f(0, ..., 0) = 0

    References
    ----------
    .. [1] Brown, A.A. & Bartholomew-Biggs, M.C. (1989). "Some effective methods for
           unconstrained optimization based on the solution of systems of ordinary
           differential equations".
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
                low=np.full(dimension, -1.0),
                high=np.full(dimension, 4.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -1.0),
                high=np.full(dimension, 4.0),
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
        x_sq = x**2
        return float(np.sum(x_sq[:-1] ** (x_sq[1:] + 1.0) + x_sq[1:] ** (x_sq[:-1] + 1.0)))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        X_sq = X**2
        return np.sum(
            X_sq[:, :-1] ** (X_sq[:, 1:] + 1.0) + X_sq[:, 1:] ** (X_sq[:, :-1] + 1.0),
            axis=1,
        )

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.zeros(self.dimension), 0.0
