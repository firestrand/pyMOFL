"""
Multimodal niching benchmark functions (CEC 2013 / CEC 2015 Niching benchmark).

Implements specialized multimodal niching base functions, including trap functions
and expanded multimodal landscapes designed to evaluate algorithms on locating
and maintaining multiple global and local optima.

References
----------
.. [1] Li, X., Engelbrecht, A., & Epitropakis, M. G. (2013). "Benchmark Functions for the
       CEC'2013 Special Session and Competition on Niching Methods for Multimodal Function
       Optimization." Technical Report, RMIT University.
.. [2] Epitropakis, M. G., Li, X., & Engelbrecht, A. (2015). "Competition on Niching Methods
       for Multimodal Optimization." IEEE CEC 2015.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("ExpandedTwoPeakTrap")
@register("expanded_two_peak_trap")
@register("TwoPeakTrap")
@register("two_peak_trap")
class ExpandedTwoPeakTrapFunction(OptimizationFunction):
    """Expanded Two-Peak Trap benchmark function.

    Base 1D function on [0, 20] with one global optimum and one local trap:
        trap(x) = (160/15) * (15 - x)  if x < 15
                = (200/5) * (x - 15)   if x >= 15
    Expanded formulation for minimization:
        f(x) = sum_{i=1}^D (200.0 - trap(x_i))
    """

    def __init__(
        self,
        dimension: int = 1,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.zeros(dimension, dtype=np.float64),
                high=np.full(dimension, 20.0, dtype=np.float64),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.zeros(dimension, dtype=np.float64),
                high=np.full(dimension, 20.0, dtype=np.float64),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

    def _trap1d(self, x: NDArray) -> NDArray:
        arr = np.asarray(x, dtype=np.float64)
        res = np.zeros_like(arr, dtype=np.float64)
        c_low = arr < 0.0
        c1 = (arr >= 0.0) & (arr < 15.0)
        c2 = (arr >= 15.0) & (arr <= 20.0)
        c_high = arr > 20.0
        res[c_low] = 40.0 + arr[c_low] ** 2
        res[c1] = -(160.0 / 15.0) * (15.0 - arr[c1]) + 200.0
        res[c2] = -40.0 * (arr[c2] - 15.0) + 200.0
        res[c_high] = (arr[c_high] - 20.0) ** 2
        return res

    def evaluate(self, x: NDArray) -> float:
        x = self._validate_input(x)
        return float(np.sum(self._trap1d(x)))

    def evaluate_batch(self, X: NDArray) -> NDArray:
        X = self._validate_batch_input(X)
        return np.sum(self._trap1d(X), axis=-1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.full(self.dimension, 20.0), 0.0


@register("ExpandedFiveUnevenPeakTrap")
@register("expanded_five_uneven_peak_trap")
@register("FiveUnevenPeakTrap")
@register("five_uneven_peak_trap")
class ExpandedFiveUnevenPeakTrapFunction(OptimizationFunction):
    """Expanded Five-Uneven-Peak Trap benchmark function.

    Base 1D piecewise linear function on [0, 30] featuring two global peaks (at 0 and 30)
    and three local peaks (at 5.0, 12.5, 22.5).
    """

    def __init__(
        self,
        dimension: int = 1,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.zeros(dimension, dtype=np.float64),
                high=np.full(dimension, 30.0, dtype=np.float64),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.zeros(dimension, dtype=np.float64),
                high=np.full(dimension, 30.0, dtype=np.float64),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

    def _trap1d(self, arr: NDArray) -> NDArray:
        arr = np.asarray(arr, dtype=np.float64)
        res = np.zeros_like(arr, dtype=np.float64)
        c0 = arr < 0.0
        c1 = (arr >= 0.0) & (arr < 2.5)
        c2 = (arr >= 2.5) & (arr < 5.0)
        c3 = (arr >= 5.0) & (arr < 7.5)
        c4 = (arr >= 7.5) & (arr < 12.5)
        c5 = (arr >= 12.5) & (arr < 17.5)
        c6 = (arr >= 17.5) & (arr < 22.5)
        c7 = (arr >= 22.5) & (arr < 27.5)
        c8 = (arr >= 27.5) & (arr <= 30.0)
        c9 = arr > 30.0

        res[c0] = -200.0 + arr[c0] ** 2
        res[c1] = -80.0 * (2.5 - arr[c1])
        res[c2] = -64.0 * (arr[c2] - 2.5)
        res[c3] = -64.0 * (7.5 - arr[c3])
        res[c4] = -28.0 * (arr[c4] - 7.5)
        res[c5] = -28.0 * (17.5 - arr[c5])
        res[c6] = -32.0 * (arr[c6] - 17.5)
        res[c7] = -32.0 * (27.5 - arr[c7])
        res[c8] = -80.0 * (arr[c8] - 27.5)
        res[c9] = -200.0 + (arr[c9] - 30.0) ** 2
        return res + 200.0

    def evaluate(self, x: NDArray) -> float:
        x = self._validate_input(x)
        return float(np.sum(self._trap1d(x)))

    def evaluate_batch(self, X: NDArray) -> NDArray:
        X = self._validate_batch_input(X)
        return np.sum(self._trap1d(X), axis=-1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.zeros(self.dimension), 0.0


@register("ExpandedEqualMinima")
@register("expanded_equal_minima")
@register("EqualMinima")
@register("equal_minima")
class ExpandedEqualMinimaFunction(OptimizationFunction):
    """Expanded Equal Minima benchmark function.

    Base 1D function on [0, 1]:
        f(x) = sum_{i=1}^D (1.0 - sin(5 * pi * x_i)^6)
    Has 5^D equal global minima with value 0.0.
    """

    def __init__(
        self,
        dimension: int = 1,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.zeros(dimension, dtype=np.float64),
                high=np.ones(dimension, dtype=np.float64),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.zeros(dimension, dtype=np.float64),
                high=np.ones(dimension, dtype=np.float64),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

    def _eval_arr(self, arr: NDArray) -> NDArray:
        arr = np.asarray(arr, dtype=np.float64)
        in_bounds = (arr >= 0.0) & (arr <= 1.0)
        return np.where(in_bounds, 1.0 - np.sin(5.0 * np.pi * arr) ** 6, 1.0 + arr**2)

    def evaluate(self, x: NDArray) -> float:
        x = self._validate_input(x)
        return float(np.sum(self._eval_arr(x)))

    def evaluate_batch(self, X: NDArray) -> NDArray:
        X = self._validate_batch_input(X)
        return np.sum(self._eval_arr(X), axis=-1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.full(self.dimension, 0.1), 0.0


@register("ExpandedDecreasingMinima")
@register("expanded_decreasing_minima")
@register("DecreasingMinima")
@register("decreasing_minima")
class ExpandedDecreasingMinimaFunction(OptimizationFunction):
    """Expanded Decreasing Minima benchmark function.

    Base 1D function on [0, 1] with exponentially modulated uneven minima:
        f(x) = sum_{i=1}^D (1.0 - exp(-2 * ln(2) * ((x_i - 0.1)/0.8)^2) * sin(5 * pi * x_i)^6)
    """

    def __init__(
        self,
        dimension: int = 1,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.zeros(dimension, dtype=np.float64),
                high=np.ones(dimension, dtype=np.float64),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.zeros(dimension, dtype=np.float64),
                high=np.ones(dimension, dtype=np.float64),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

    def _eval_arr(self, arr: NDArray) -> NDArray:
        arr = np.asarray(arr, dtype=np.float64)
        in_bounds = (arr >= 0.0) & (arr <= 1.0)
        val = 1.0 - np.exp(-2.0 * np.log(2.0) * ((arr - 0.1) / 0.8) ** 2) * (
            np.sin(5.0 * np.pi * arr) ** 6
        )
        return np.where(in_bounds, val, 1.0 + arr**2)

    def evaluate(self, x: NDArray) -> float:
        x = self._validate_input(x)
        return float(np.sum(self._eval_arr(x)))

    def evaluate_batch(self, X: NDArray) -> NDArray:
        X = self._validate_batch_input(X)
        return np.sum(self._eval_arr(X), axis=-1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        return np.full(self.dimension, 0.1), 0.0


@register("ExpandedUnevenMinima")
@register("expanded_uneven_minima")
@register("UnevenMinima")
@register("uneven_minima")
class ExpandedUnevenMinimaFunction(OptimizationFunction):
    """Expanded Uneven Minima benchmark function.

    Base 1D function on [0, 1]:
        f(x) = sum_{i=1}^D (1.0 - sin(5 * pi * (x_i^0.75 - 0.05))^6)
    """

    def __init__(
        self,
        dimension: int = 1,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.zeros(dimension, dtype=np.float64),
                high=np.ones(dimension, dtype=np.float64),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.zeros(dimension, dtype=np.float64),
                high=np.ones(dimension, dtype=np.float64),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

    def _eval_arr(self, arr: NDArray) -> NDArray:
        arr = np.asarray(arr, dtype=np.float64)
        in_bounds = (arr >= 0.0) & (arr <= 1.0)
        safe_arr = np.maximum(arr, 0.0)
        val = 1.0 - np.sin(5.0 * np.pi * (safe_arr**0.75 - 0.05)) ** 6
        return np.where(in_bounds, val, 1.0 + arr**2)

    def evaluate(self, x: NDArray) -> float:
        x = self._validate_input(x)
        return float(np.sum(self._eval_arr(x)))

    def evaluate_batch(self, X: NDArray) -> NDArray:
        X = self._validate_batch_input(X)
        return np.sum(self._eval_arr(X), axis=-1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        x_opt = 0.15 ** (1.0 / 0.75)
        return np.full(self.dimension, x_opt), 0.0


@register("ExpandedHimmelblau")
@register("expanded_himmelblau")
class ExpandedHimmelblauFunction(OptimizationFunction):
    """Expanded Himmelblau multimodal benchmark function.

    Expands 2D Himmelblau function across non-overlapping 2D pairs:
        f(x) = sum_{i=0, step 2}^{D-2} ((x_i^2 + x_{i+1} - 11)^2 + (x_i + x_{i+1}^2 - 7)^2)
    Domain: [-6, 6]^D.
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension < 2 or dimension % 2 != 0:
            raise ValueError(
                f"ExpandedHimmelblauFunction requires even dimension >= 2, got {dimension}"
            )
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -6.0, dtype=np.float64),
                high=np.full(dimension, 6.0, dtype=np.float64),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -6.0, dtype=np.float64),
                high=np.full(dimension, 6.0, dtype=np.float64),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

    def evaluate(self, x: NDArray) -> float:
        x = self._validate_input(x)
        if self.dimension % 2 != 0:
            raise ValueError(
                f"ExpandedHimmelblauFunction requires even dimension, got {self.dimension}"
            )
        xi = x[0::2]
        xip1 = x[1::2]
        terms = (xi**2 + xip1 - 11.0) ** 2 + (xi + xip1**2 - 7.0) ** 2
        return float(np.sum(terms))

    def evaluate_batch(self, X: NDArray) -> NDArray:
        X = self._validate_batch_input(X)
        if self.dimension % 2 != 0:
            raise ValueError(
                f"ExpandedHimmelblauFunction requires even dimension, got {self.dimension}"
            )
        Xi = X[:, 0::2]
        Xip1 = X[:, 1::2]
        terms = (Xi**2 + Xip1 - 11.0) ** 2 + (Xi + Xip1**2 - 7.0) ** 2
        return np.sum(terms, axis=-1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        if self.dimension == 2:
            return np.array([3.0, 2.0]), 0.0
        opt_2d = np.tile([3.0, 2.0], self.dimension // 2)
        return opt_2d, 0.0


@register("ExpandedSixHumpCamel")
@register("expanded_six_hump_camel")
class ExpandedSixHumpCamelFunction(OptimizationFunction):
    """Expanded Six-Hump Camel benchmark function.

    Expands 2D Six-Hump Camel back function across non-overlapping 2D pairs:
        f(x) = sum_{i=0, step 2}^{D-2} 4 * ((4 - 2.1*x_i^2 + x_i^4/3)*x_i^2 + x_i*x_{i+1} + (-4 + 4*x_{i+1}^2)*x_{i+1}^2) + 4.126514 * (D / 2)
    Domain: [-2, 2]^D.
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension < 2 or dimension % 2 != 0:
            raise ValueError(
                f"ExpandedSixHumpCamelFunction requires even dimension >= 2, got {dimension}"
            )
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -2.0, dtype=np.float64),
                high=np.full(dimension, 2.0, dtype=np.float64),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -2.0, dtype=np.float64),
                high=np.full(dimension, 2.0, dtype=np.float64),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

    def evaluate(self, x: NDArray) -> float:
        x = self._validate_input(x)
        if self.dimension % 2 != 0:
            raise ValueError(
                f"ExpandedSixHumpCamelFunction requires even dimension, got {self.dimension}"
            )
        xi = x[0::2]
        xip1 = x[1::2]
        x2 = xi**2
        x4 = xi**4
        y2 = xip1**2
        terms = ((4.0 - 2.1 * x2 + x4 / 3.0) * x2 + xi * xip1 + (4.0 * y2 - 4.0) * y2) * 4.0
        return float(np.sum(terms) + 4.126514 * (self.dimension / 2.0))

    def evaluate_batch(self, X: NDArray) -> NDArray:
        X = self._validate_batch_input(X)
        if self.dimension % 2 != 0:
            raise ValueError(
                f"ExpandedSixHumpCamelFunction requires even dimension, got {self.dimension}"
            )
        Xi = X[:, 0::2]
        Xip1 = X[:, 1::2]
        X2 = Xi**2
        X4 = Xi**4
        Y2 = Xip1**2
        terms = ((4.0 - 2.1 * X2 + X4 / 3.0) * X2 + Xi * Xip1 + (4.0 * Y2 - 4.0) * Y2) * 4.0
        return np.sum(terms, axis=-1) + 4.126514 * (self.dimension / 2.0)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        if self.dimension == 2:
            return np.array([0.089842, -0.712656]), 0.0
        opt_2d = np.tile([0.089842, -0.712656], self.dimension // 2)
        return opt_2d, 0.0


@register("ModifiedVincent")
@register("modified_vincent")
class ModifiedVincentFunction(OptimizationFunction):
    """Modified Vincent multimodal benchmark function.

    Defined on [0.25, 10]^D:
        f(x) = (1.0 / D) * sum_{i=1}^D (1.0 - sin(10.0 * ln(x_i)))
    Features 6^D global optima where f(x) = 0.0.
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, 0.25, dtype=np.float64),
                high=np.full(dimension, 10.0, dtype=np.float64),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, 0.25, dtype=np.float64),
                high=np.full(dimension, 10.0, dtype=np.float64),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

    def _eval_terms(self, x: NDArray) -> NDArray:
        arr = np.asarray(x, dtype=np.float64)
        in_bounds = (arr >= 0.25) & (arr <= 10.0)
        low = arr < 0.25
        high = arr > 10.0

        terms = np.zeros_like(arr, dtype=np.float64)
        safe_x = np.maximum(arr, 1e-15)
        terms[in_bounds] = -np.sin(10.0 * np.log(safe_x[in_bounds]))
        terms[low] = (0.25 - arr[low]) ** 2 - np.sin(10.0 * np.log(2.5))
        terms[high] = (arr[high] - 10.0) ** 2 - np.sin(10.0 * np.log(10.0))
        return terms + 1.0

    def evaluate(self, x: NDArray) -> float:
        x = self._validate_input(x)
        return float(np.sum(self._eval_terms(x)) / self.dimension)

    def evaluate_batch(self, X: NDArray) -> NDArray:
        X = self._validate_batch_input(X)
        return np.sum(self._eval_terms(X), axis=-1) / self.dimension

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        # At 10 * ln(x) = pi/2 -> ln(x) = pi/20 -> x = exp(pi/20) ≈ 1.17009
        x_opt = float(np.exp(np.pi / 20.0))
        return np.full(self.dimension, x_opt), 0.0
