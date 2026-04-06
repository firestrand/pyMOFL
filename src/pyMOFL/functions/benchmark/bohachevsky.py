"""Bohachevsky benchmark functions (variants 1, 2, 3)."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


def _default_bounds():
    return Bounds(
        low=np.array([-100.0, -100.0]),
        high=np.array([100.0, 100.0]),
        mode=BoundModeEnum.OPERATIONAL,
        qtype=QuantizationTypeEnum.CONTINUOUS,
    )


@register("Bohachevsky1")
@register("bohachevsky1")
class Bohachevsky1Function(OptimizationFunction):
    """
    Bohachevsky function variant 1 (2D).

    f(x) = x1^2 + 2*x2^2 - 0.3*cos(3*pi*x1) - 0.4*cos(4*pi*x2) + 0.7

    Global minimum: f(0, 0) = 0
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Bohachevsky1 function is Non-Scalable and requires dimension=2")

        default = _default_bounds()
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default,
            operational_bounds=operational_bounds or default,
        )

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        return float(
            x1**2
            + 2.0 * x2**2
            - 0.3 * np.cos(3.0 * np.pi * x1)
            - 0.4 * np.cos(4.0 * np.pi * x2)
            + 0.7
        )

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return (
            x1**2
            + 2.0 * x2**2
            - 0.3 * np.cos(3.0 * np.pi * x1)
            - 0.4 * np.cos(4.0 * np.pi * x2)
            + 0.7
        )

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([0.0, 0.0]), 0.0


@register("Bohachevsky2")
@register("bohachevsky2")
class Bohachevsky2Function(OptimizationFunction):
    """
    Bohachevsky function variant 2 (2D).

    f(x) = x1^2 + 2*x2^2 - 0.3*cos(3*pi*x1)*cos(4*pi*x2) + 0.3

    Global minimum: f(0, 0) = 0
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Bohachevsky2 function is Non-Scalable and requires dimension=2")

        default = _default_bounds()
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default,
            operational_bounds=operational_bounds or default,
        )

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        return float(
            x1**2 + 2.0 * x2**2 - 0.3 * np.cos(3.0 * np.pi * x1) * np.cos(4.0 * np.pi * x2) + 0.3
        )

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return x1**2 + 2.0 * x2**2 - 0.3 * np.cos(3.0 * np.pi * x1) * np.cos(4.0 * np.pi * x2) + 0.3

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([0.0, 0.0]), 0.0


@register("Bohachevsky3")
@register("bohachevsky3")
class Bohachevsky3Function(OptimizationFunction):
    """
    Bohachevsky function variant 3 (2D).

    f(x) = x1^2 + 2*x2^2 - 0.3*cos(3*pi*x1 + 4*pi*x2) + 0.3

    Global minimum: f(0, 0) = 0
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Bohachevsky3 function is Non-Scalable and requires dimension=2")

        default = _default_bounds()
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default,
            operational_bounds=operational_bounds or default,
        )

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        x1, x2 = x[0], x[1]
        return float(x1**2 + 2.0 * x2**2 - 0.3 * np.cos(3.0 * np.pi * x1 + 4.0 * np.pi * x2) + 0.3)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        return x1**2 + 2.0 * x2**2 - 0.3 * np.cos(3.0 * np.pi * x1 + 4.0 * np.pi * x2) + 0.3

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.array([0.0, 0.0]), 0.0
