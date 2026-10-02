"""
FastFractal DoubleDip benchmark function.

Implements the multi-modal, non-separable fractal landscape from CEC 2008 (Function 7),
originally contributed by Aleš Zamuda and based on Cara MacNish's randomized
self-similar fractal landscapes.

References
----------
.. [1] Tang, K., Yao, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007).
       "Benchmark Functions for the CEC'2008 Special Session and Competition on Large Scale
       Global Optimization". Nature Inspired Computation and Applications Laboratory (NICAL),
       USTC, China, Tech. Rep.
.. [2] MacNish, C. (2007). "Towards Unbiased Benchmarking of Evolutionary and Hybrid Algorithms
       for Real-valued Optimisation". Connection Science, 19(4), 361-385.
.. [3] MacNish, C. (2006). "Benchmarking Evolutionary and Hybrid Algorithms using Randomized
       Self-Similar Landscapes". Proc. 6th Int. Conf. on Simulated Evolution and Learning
       (SEAL'06), LNCS 4247, pp. 361-368. Springer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register

DOUBLE_TABLE_SIZE: int = 16384  # 0x3fff + 1
INT_TABLE_SIZE: int = 256  # 0xff + 1
_MASK_32: int = 0xFFFFFFFF
_MAX_INT_FLOAT: float = 4294967295.0
_LCG_A: int = 1664525
_LCG_C: int = 1013904223


def _double_dip_value(point: float, centre: float, scale: float) -> float:
    """Evaluate 1D unit DoubleDip base function value."""
    x = (point - centre) / scale
    if -0.5 < x < 0.5:
        xs = 4.0 * x * x
        return (-96.0 * xs * xs * xs + 193.0 * xs * xs - 98.0 * xs + 1.0) * scale
    return 0.0


def _double_dip_twist(y: float) -> float:
    """Compute twisting displacement dx based on coordinate y."""
    y = math.fmod(y, 1.0)
    ys = y * y
    if y > 0.0:
        return 4.0 * (ys * ys - 2.0 * ys * y + ys)
    return 4.0 * (ys * ys + 2.0 * ys * y + ys)


class _RanTable:
    """Deterministic pseudo-random lookup tables matching MacNish's RanTable / RanQD1."""

    def __init__(self, ave_int: int = 1, index: int = 1) -> None:
        self.double_table_size = DOUBLE_TABLE_SIZE
        self.int_table_size = INT_TABLE_SIZE

        # Populate double table using RanQD1(index)
        idum = index
        idum = (_LCG_A * idum + _LCG_C) & _MASK_32
        self.double_table = np.zeros(DOUBLE_TABLE_SIZE, dtype=np.float64)
        for i in range(DOUBLE_TABLE_SIZE):
            idum = (_LCG_A * idum + _LCG_C) & _MASK_32
            self.double_table[i] = idum / _MAX_INT_FLOAT

        # Populate integer table using RanQD1(index)
        idum = index
        idum = (_LCG_A * idum + _LCG_C) & _MASK_32
        self.int_table = np.zeros(INT_TABLE_SIZE, dtype=np.int32)
        upper_bound = 2 * ave_int
        for i in range(INT_TABLE_SIZE):
            idum = (_LCG_A * idum + _LCG_C) & _MASK_32
            val = idum / _MAX_INT_FLOAT
            self.int_table[i] = math.floor(val * (upper_bound + 1))

        self.double_table_index = 0
        self.int_table_index = 0

    def set_seed(self, seed: int) -> None:
        self.double_table_index = seed & (DOUBLE_TABLE_SIZE - 1)
        self.int_table_index = seed & (INT_TABLE_SIZE - 1)

    def next_double(self) -> float:
        self.double_table_index = (self.double_table_index + 1) & (DOUBLE_TABLE_SIZE - 1)
        return float(self.double_table[self.double_table_index])

    def next_integer(self) -> int:
        self.int_table_index = (self.int_table_index + 1) & (INT_TABLE_SIZE - 1)
        return int(self.int_table[self.int_table_index])


class _Fractal1DEngine:
    """Recursive 1D fractal evaluation engine."""

    def __init__(self, fractal_depth: int = 3, density: int = 1, index: int = 1) -> None:
        self.fractal_depth = fractal_depth
        self.density = density
        self.index = index
        self.ran_table = _RanTable(ave_int=density, index=index)

    def set_index(self, index: int) -> None:
        self.index = index
        self.ran_table.set_seed(index)

    def evaluate(self, x: float) -> float:
        x = math.fmod(x, 1.0)
        if x <= 0.0:
            x = x + 1.0
        if self.fractal_depth < 1:
            return 0.0
        return self._get_depth_local(x, 1, self.index, 1)

    def _get_depth_local(self, x: float, rec_depth: int, seed: int, span: int) -> float:
        depth = 0.0
        scale = 1.0 / span
        square = math.ceil(x * span)
        for offset in (-1, 0, 1):
            x1 = x
            square1 = square + offset
            if square1 == 0:
                square1 = span
                x1 = x1 + 1.0
            elif square1 > span:
                square1 = 1
                x1 = x1 - 1.0
            depth += self._get_depth_wrt_square(x1, square1, seed, scale)
        if rec_depth < self.fractal_depth:
            new_seed = (span + seed) & (DOUBLE_TABLE_SIZE - 1)
            new_span = span << 1
            depth += self._get_depth_local(x, rec_depth + 1, new_seed, new_span)
        return depth

    def _get_depth_wrt_square(self, x: float, square: int, seed: int, scale: float) -> float:
        depth = 0.0
        square_seed = square - 1
        local_seed = (seed + square_seed) & (DOUBLE_TABLE_SIZE - 1)
        self.ran_table.set_seed(local_seed)
        num_units = self.ran_table.next_integer()
        for _ in range(num_units):
            diameter = 1.0 / (2.0 - self.ran_table.next_double()) * scale
            centre = (square - self.ran_table.next_double()) * scale
            dist = x - centre
            if dist * dist < diameter * diameter * 0.25:
                depth += _double_dip_value(x, centre, diameter)
        return depth

    def evaluate_array(self, x_arr: NDArray[np.float64], seed: int) -> NDArray[np.float64]:
        """Vectorized evaluation of 1D fractal function on an array of points."""
        x = np.fmod(x_arr, 1.0)
        x = np.where(x <= 0.0, x + 1.0, x)
        depth = np.zeros_like(x, dtype=np.float64)
        if self.fractal_depth < 1:
            return depth

        def _rec_batch(
            curr_x: NDArray[np.float64], rec_depth: int, s: int, span: int
        ) -> NDArray[np.float64]:
            d = np.zeros_like(curr_x, dtype=np.float64)
            scale = 1.0 / span
            square = np.ceil(curr_x * span).astype(np.int64)
            for offset in (-1, 0, 1):
                x1 = curr_x.copy()
                sq1 = square + offset
                wrap_rhs = sq1 == 0
                sq1[wrap_rhs] = span
                x1[wrap_rhs] += 1.0
                wrap_lhs = sq1 > span
                sq1[wrap_lhs] = 1
                x1[wrap_lhs] -= 1.0

                unique_sqs = np.unique(sq1)
                for u_sq in unique_sqs:
                    mask = sq1 == u_sq
                    if not np.any(mask):
                        continue
                    local_seed = (s + u_sq - 1) & (DOUBLE_TABLE_SIZE - 1)
                    i_idx = (local_seed + 1) & (INT_TABLE_SIZE - 1)
                    num_units = self.ran_table.int_table[i_idx]
                    d_idx = local_seed
                    sub_x = x1[mask]
                    d_sub = np.zeros_like(sub_x)
                    for _ in range(num_units):
                        d_idx = (d_idx + 1) & (DOUBLE_TABLE_SIZE - 1)
                        rand_diam = self.ran_table.double_table[d_idx]
                        diameter = 1.0 / (2.0 - rand_diam) * scale

                        d_idx = (d_idx + 1) & (DOUBLE_TABLE_SIZE - 1)
                        rand_centre = self.ran_table.double_table[d_idx]
                        centre = (u_sq - rand_centre) * scale

                        dist = sub_x - centre
                        cond = dist * dist < diameter * diameter * 0.25
                        if np.any(cond):
                            u = dist[cond] / diameter
                            xs = 4.0 * u * u
                            d_sub[cond] += (
                                -96.0 * xs * xs * xs + 193.0 * xs * xs - 98.0 * xs + 1.0
                            ) * diameter
                    d[mask] += d_sub

            if rec_depth < self.fractal_depth:
                new_seed = (span + s) & (DOUBLE_TABLE_SIZE - 1)
                new_span = span << 1
                d += _rec_batch(curr_x, rec_depth + 1, new_seed, new_span)
            return d

        return _rec_batch(x, 1, seed, 1)


@register("FastFractalDoubleDip")
@register("fast_fractal_double_dip")
@register("DoubleDip")
@register("FastFractal")
class FastFractalDoubleDip(OptimizationFunction):
    r"""
    FastFractal "DoubleDip" function (CEC 2008 F7).

    A multi-modal, non-separable benchmark landscape constructed from recursive 1D
    double-dip polynomial wavelets with nonlinear coordinate twisting:

    .. math::

        F_7(\mathbf{x}) = \sum_{i=1}^D \text{fractal1D}\left(x_i + \text{twist}(x_{(i \bmod D) + 1})\right)

    where:

    .. math::

        \text{twist}(y) = 4(y^4 - 2y^3 + y^2)

    and the 1D fractal component is recursively evaluated across multi-resolution
    octaves using deterministic Knuth LCG pseudo-random parameter lookup tables.

    Parameters
    ----------
    dimension : int, default=1000
        Dimensionality of the problem (:math:`D \ge 2`).
    initialization_bounds : Bounds, optional
        Bounds for random initialization. Defaults to :math:`[-1, 1]^D`.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. Defaults to :math:`[-1, 1]^D`.
    fractal_depth : int, default=3
        Recursive depth of fractal resolution levels (octaves).
    density : int, default=1
        Average number of base functions per unit interval per resolution level.
    seed_index : int, default=1
        Initial seed/index parameter for random lookup tables.

    References
    ----------
    .. [1] Tang, K., Yao, X., Suganthan, P. N., MacNish, C., Chen, Y. P., Chen, C. M., & Yang, Z. (2007).
           "Benchmark Functions for the CEC'2008 Special Session and Competition on Large Scale
           Global Optimization". Nature Inspired Computation and Applications Laboratory (NICAL),
           USTC, China, Tech. Rep.
    .. [2] MacNish, C. (2007). "Towards Unbiased Benchmarking of Evolutionary and Hybrid Algorithms
           for Real-valued Optimisation". Connection Science, 19(4), 361-385.
    """

    def __init__(
        self,
        dimension: int = 1000,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        fractal_depth: int = 3,
        density: int = 1,
        seed_index: int = 1,
        **kwargs: Any,
    ) -> None:
        if dimension < 2:
            raise ValueError(f"FastFractalDoubleDip requires dimension >= 2, got {dimension}")

        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -1.0),
                high=np.full(dimension, 1.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -1.0),
                high=np.full(dimension, 1.0),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )

        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
            **kwargs,
        )

        self.fractal_depth = fractal_depth
        self.density = density
        self.seed_index = seed_index
        self._engine = _Fractal1DEngine(
            fractal_depth=fractal_depth, density=density, index=seed_index
        )

    def evaluate(self, x: NDArray[Any]) -> float:
        """Evaluate the FastFractal DoubleDip function at point x."""
        x = self._validate_input(x)
        d = self.dimension
        depth = 0.0
        last_x = float(x[-1])

        for i in range(d):
            xi = float(x[i])
            dx = _double_dip_twist(last_x)
            seed = 6 * i + 1
            self._engine.set_index(seed)
            depth += self._engine.evaluate(xi + dx)
            last_x = xi

        return float(depth)

    def evaluate_batch(self, X: NDArray[Any]) -> NDArray[np.float64]:
        """Vectorized batch evaluation of the FastFractal DoubleDip function."""
        X = self._validate_batch_input(X)
        d = self.dimension
        n_points = X.shape[0]
        results = np.zeros(n_points, dtype=np.float64)

        last_x = X[:, -1].astype(np.float64)
        for i in range(d):
            xi = X[:, i].astype(np.float64)
            # Compute twist vectorized
            y = np.fmod(last_x, 1.0)
            ys = y * y
            dx = np.where(
                y > 0.0,
                4.0 * (ys * ys - 2.0 * ys * y + ys),
                4.0 * (ys * ys + 2.0 * ys * y + ys),
            )
            seed = 6 * i + 1
            results += self._engine.evaluate_array(xi + dx, seed)
            last_x = xi

        return results

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Global minimum is analytically unknown for FastFractal landscapes."""
        raise NotImplementedError(
            "FastFractalDoubleDip global optimum is analytically unknown as per CEC 2008 specification."
        )
