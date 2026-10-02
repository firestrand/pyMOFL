"""
Schwefel x*sin(sqrt(|x|)) function (BBOB f20 base / Schwefel 2.26).

f(x) = 418.9829 * D - sum(x_i * sin(sqrt(|x_i|)))

The traditional Schwefel function with a deceptive global structure:
the second-best optimum is far from the global one and much easier
to find, making this a challenging benchmark for global optimization.

References
----------
.. [1] Schwefel, H.-P. (1981). "Numerical optimization of computer models."
       John Wiley & Sons.
.. [2] Hansen, N., Finck, S., Ros, R., & Auger, A. (2009).
       "Real-Parameter Black-Box Optimization Benchmarking 2009:
       Noiseless Functions Definitions." INRIA Technical Report RR-6829.
"""

import numpy as np
from numpy.typing import NDArray

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register

# The constant per dimension that offsets the sum at the global optimum.
# 418.9828872724339 is the standard value used in the literature.
_SCHWEFEL_CONSTANT = 418.9828872724339

# The approximate global optimum coordinate.
_SCHWEFEL_OPTIMUM_COORD = 420.9687462275036


@register("SchwefelSin")
@register("schwefel_sin")
class SchwefelSinFunction(OptimizationFunction):
    """
    Schwefel x*sin(sqrt(|x|)) function.

    f(x) = 418.9829 * D - sum(x_i * sin(sqrt(|x_i|)))

    Global minimum: f(420.9687..., ..., 420.9687...) ≈ 0

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for random initialization. Defaults to [-500, 500]^D.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. Defaults to [-500, 500]^D.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        boundary_handling: bool = False,
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
        self._boundary_handling = boundary_handling

    def evaluate(self, x: NDArray) -> float:
        """Compute the Schwefel x*sin(sqrt(|x|)) function value."""
        x = self._validate_input(x)
        if not self._boundary_handling:
            return float(
                _SCHWEFEL_CONSTANT * self.dimension - np.sum(x * np.sin(np.sqrt(np.abs(x))))
            )
        # CEC 2014 boundary handling: clamp + penalty for |x_i| > 500
        D = self.dimension
        f = 0.0
        mask_pos = x > 500
        mask_neg = x < -500
        mask_std = ~(mask_pos | mask_neg)
        # Standard region
        x_std = x[mask_std]
        f -= np.sum(x_std * np.sin(np.sqrt(np.abs(x_std))))
        # z > 500: clamp via fmod
        if np.any(mask_pos):
            z_pos = x[mask_pos]
            clamped = 500.0 - np.fmod(z_pos, 500.0)
            f -= np.sum(clamped * np.sin(np.sqrt(clamped)))
            f += np.sum(((z_pos - 500.0) / 100.0) ** 2) / D
        # z < -500: clamp via fmod
        if np.any(mask_neg):
            z_neg = x[mask_neg]
            mod_val = np.fmod(np.abs(z_neg), 500.0)
            clamped = 500.0 - mod_val
            f -= np.sum((-500.0 + mod_val) * np.sin(np.sqrt(clamped)))
            f += np.sum(((z_neg + 500.0) / 100.0) ** 2) / D
        f += _SCHWEFEL_CONSTANT * D
        return float(f)

    def evaluate_batch(self, X: NDArray, out: NDArray | None = None) -> NDArray:
        """Compute Schwefel function for batch."""
        X = self._validate_batch_input(X)
        if not self._boundary_handling:
            val = _SCHWEFEL_CONSTANT * self.dimension - np.sum(
                X * np.sin(np.sqrt(np.abs(X))), axis=1
            )
            if out is not None:
                out[:] = val
                return out
            return val

        # CEC 2014 boundary handling vectorized across (N, D)
        D = self.dimension
        term = np.zeros_like(X)
        penalty = np.zeros_like(X)

        mask_pos = X > 500.0
        mask_neg = X < -500.0
        mask_std = ~(mask_pos | mask_neg)

        if np.any(mask_std):
            x_std = X[mask_std]
            term[mask_std] = -x_std * np.sin(np.sqrt(np.abs(x_std)))

        if np.any(mask_pos):
            z_pos = X[mask_pos]
            clamped_pos = 500.0 - np.fmod(z_pos, 500.0)
            term[mask_pos] = -clamped_pos * np.sin(np.sqrt(clamped_pos))
            penalty[mask_pos] = ((z_pos - 500.0) / 100.0) ** 2

        if np.any(mask_neg):
            z_neg = X[mask_neg]
            mod_val = np.fmod(np.abs(z_neg), 500.0)
            clamped_neg = 500.0 - mod_val
            term[mask_neg] = -(-500.0 + mod_val) * np.sin(np.sqrt(clamped_neg))
            penalty[mask_neg] = ((z_neg + 500.0) / 100.0) ** 2

        total = np.sum(term, axis=1) + (np.sum(penalty, axis=1) / D) + (_SCHWEFEL_CONSTANT * D)
        if out is not None:
            out[:] = total
            return out
        return total

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """
        Get the global minimum point and value.

        The global optimum is at x_i ≈ 420.9687 for all i.
        The function value is approximately 0 (not exactly, due to
        the irrational nature of the optimum coordinate).

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        global_min_point = np.full(self.dimension, _SCHWEFEL_OPTIMUM_COORD)
        # Compute the actual value at the optimum coordinate
        global_min_value = float(
            _SCHWEFEL_CONSTANT * self.dimension
            - self.dimension
            * _SCHWEFEL_OPTIMUM_COORD
            * np.sin(np.sqrt(np.abs(_SCHWEFEL_OPTIMUM_COORD)))
        )
        return global_min_point, global_min_value
