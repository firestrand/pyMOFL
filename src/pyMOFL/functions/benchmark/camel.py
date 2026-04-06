"""Camel benchmark functions (Six-Hump and Three-Hump variants)."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("SixHumpCamel")
@register("six_hump_camel")
class SixHumpCamelFunction(OptimizationFunction):
    """
    Six-Hump Camel function (2D).

    f(x) = (4 - 2.1*x1^2 + x1^4/3)*x1^2 + x1*x2 + (-4 + 4*x2^2)*x2^2

    Two global minima at (0.0898, -0.7126) and (-0.0898, 0.7126) with f* = -1.0316285.

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
            raise ValueError("Six-Hump Camel function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-5.0, -5.0]),
            high=np.array([5.0, 5.0]),
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
            (4.0 - 2.1 * x1**2 + x1**4 / 3.0) * x1**2 + x1 * x2 + (-4.0 + 4.0 * x2**2) * x2**2
        )

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return (4.0 - 2.1 * x1**2 + x1**4 / 3.0) * x1**2 + x1 * x2 + (-4.0 + 4.0 * x2**2) * x2**2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        min_point = np.array([0.0898, -0.7126])
        min_value = self.evaluate(min_point)
        return min_point, min_value


@register("ThreeHumpCamel")
@register("three_hump_camel")
class ThreeHumpCamelFunction(OptimizationFunction):
    """
    Three-Hump Camel function (2D).

    f(x) = 2*x1^2 - 1.05*x1^4 + x1^6/6 + x1*x2 + x2^2

    Global minimum: f(0, 0) = 0

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
            raise ValueError("Three-Hump Camel function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-5.0, -5.0]),
            high=np.array([5.0, 5.0]),
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
        return float(2.0 * x1**2 - 1.05 * x1**4 + x1**6 / 6.0 + x1 * x2 + x2**2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return 2.0 * x1**2 - 1.05 * x1**4 + x1**6 / 6.0 + x1 * x2 + x2**2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([0.0, 0.0]), 0.0
