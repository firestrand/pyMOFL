"""Beale benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Beale")
@register("beale")
class BealeFunction(OptimizationFunction):
    """
    Beale function (2D).

    f(x) = (1.5 - x1*(1-x2))^2 + (2.25 - x1*(1-x2^2))^2 + (2.625 - x1*(1-x2^3))^2

    Global minimum: f(3, 0.5) = 0

    References
    ----------
    .. [1] Beale, E.M.L. (1958). "On an iterative method for finding a local minimum
           of a function of more than one variable".
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Beale function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-4.5, -4.5]),
            high=np.array([4.5, 4.5]),
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
        t1 = (1.5 - x1 * (1.0 - x2)) ** 2
        t2 = (2.25 - x1 * (1.0 - x2**2)) ** 2
        t3 = (2.625 - x1 * (1.0 - x2**3)) ** 2
        return float(t1 + t2 + t3)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        t1 = (1.5 - x1 * (1.0 - x2)) ** 2
        t2 = (2.25 - x1 * (1.0 - x2**2)) ** 2
        t3 = (2.625 - x1 * (1.0 - x2**3)) ** 2
        return t1 + t2 + t3

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([3.0, 0.5]), 0.0
