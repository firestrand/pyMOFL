"""Hartmann benchmark functions (3D and 6D variants)."""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Hartmann3")
@register("hartmann3")
class Hartmann3Function(OptimizationFunction):
    """
    Hartmann 3D function.

    f(x) = -sum_i alpha_i * exp(-sum_j A_ij * (x_j - P_ij)^2)

    Global minimum: f(0.114614, 0.555649, 0.852547) ~ -3.8628

    References
    ----------
    .. [1] Picheny, V., Wagner, T. & Ginsbourger, D. (2012). "A benchmark of kriging-based
           infill criteria for noisy optimization".
    """

    _ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
    _A = np.array(
        [
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
        ]
    )
    _P = np.array(
        [
            [0.3689, 0.1170, 0.2673],
            [0.4699, 0.4387, 0.7470],
            [0.1091, 0.8732, 0.5547],
            [0.0382, 0.5743, 0.8828],
        ]
    )

    def __init__(
        self,
        dimension: int = 3,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 3:
            raise ValueError("Hartmann3 function is Non-Scalable and requires dimension=3")

        default_bounds = Bounds(
            low=np.zeros(3),
            high=np.ones(3),
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
        outer = 0.0
        for i in range(4):
            inner = np.sum(self._A[i] * (x - self._P[i]) ** 2)
            outer += self._ALPHA[i] * np.exp(-inner)
        return float(-outer)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        # X: (N, 3), A: (4, 3), P: (4, 3)
        # diff: (N, 4, 3) = X[:, None, :] - P[None, :, :]
        diff = X[:, np.newaxis, :] - self._P[np.newaxis, :, :]
        inner = np.sum(self._A * diff**2, axis=2)  # (N, 4)
        return -np.sum(self._ALPHA * np.exp(-inner), axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        min_point = np.array([0.114614, 0.555649, 0.852547])
        min_value = self.evaluate(min_point)
        return min_point, min_value


@register("Hartmann6")
@register("hartmann6")
class Hartmann6Function(OptimizationFunction):
    """
    Hartmann 6D function.

    f(x) = -sum_i alpha_i * exp(-sum_j A_ij * (x_j - P_ij)^2)

    Global minimum: f(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573) ~ -3.3224

    References
    ----------
    .. [1] Picheny, V., Wagner, T. & Ginsbourger, D. (2012). "A benchmark of kriging-based
           infill criteria for noisy optimization".
    """

    _ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
    _A = np.array(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ]
    )
    _P = np.array(
        [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ]
    )

    def __init__(
        self,
        dimension: int = 6,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 6:
            raise ValueError("Hartmann6 function is Non-Scalable and requires dimension=6")

        default_bounds = Bounds(
            low=np.zeros(6),
            high=np.ones(6),
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
        outer = 0.0
        for i in range(4):
            inner = np.sum(self._A[i] * (x - self._P[i]) ** 2)
            outer += self._ALPHA[i] * np.exp(-inner)
        return float(-outer)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        diff = X[:, np.newaxis, :] - self._P[np.newaxis, :, :]
        inner = np.sum(self._A * diff**2, axis=2)
        return -np.sum(self._ALPHA * np.exp(-inner), axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        min_point = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        min_value = self.evaluate(min_point)
        return min_point, min_value
