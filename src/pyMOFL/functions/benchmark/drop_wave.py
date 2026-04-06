"""Drop-Wave benchmark function."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("DropWave")
@register("drop_wave")
class DropWaveFunction(OptimizationFunction):
    """
    Drop-Wave function (2D).

    f(x) = -(1 + cos(12*sqrt(x1^2 + x2^2))) / (0.5*(x1^2 + x2^2) + 2)

    Global minimum: f(0, 0) = -1

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
            raise ValueError("Drop-Wave function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-5.12, -5.12]),
            high=np.array([5.12, 5.12]),
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
        r_sq = x1**2 + x2**2
        return float(-(1.0 + np.cos(12.0 * np.sqrt(r_sq))) / (0.5 * r_sq + 2.0))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        r_sq = x1**2 + x2**2
        return -(1.0 + np.cos(12.0 * np.sqrt(r_sq))) / (0.5 * r_sq + 2.0)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([0.0, 0.0]), -1.0
