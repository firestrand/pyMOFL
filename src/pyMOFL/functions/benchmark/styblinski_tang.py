"""Styblinski-Tang benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("StyblinskiTang")
@register("styblinski_tang")
class StyblinskiTangFunction(OptimizationFunction):
    """
    Styblinski-Tang function.

    f(x) = 0.5 * sum(x_i^4 - 16*x_i^2 + 5*x_i)

    Global minimum: f(-2.903534, ..., -2.903534) = -39.16617 * D

    References
    ----------
    .. [1] Styblinski, M.A. & Tang, T.S. (1990). "Experiments in nonconvex optimization:
           Stochastic approximation with function smoothing and simulated annealing".
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
                low=np.full(dimension, -5.0),
                high=np.full(dimension, 5.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -5.0),
                high=np.full(dimension, 5.0),
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
        return float(0.5 * np.sum(x**4 - 16.0 * x**2 + 5.0 * x))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        return 0.5 * np.sum(X**4 - 16.0 * X**2 + 5.0 * X, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        x_star = np.full(self.dimension, -2.903534)
        f_star = float(0.5 * np.sum(x_star**4 - 16.0 * x_star**2 + 5.0 * x_star))
        return x_star, f_star
