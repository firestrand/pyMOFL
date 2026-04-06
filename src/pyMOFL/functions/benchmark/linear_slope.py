"""
Linear Slope function (BBOB f5 base).

f(x) = sum(5 * |s_i| - s_i * x_i)
where s_i = sign_i * 10^((i-1)/(D-1))

The optimum lies at the domain boundary: x_i = sign_i * 5.
This is the only purely linear function in the BBOB suite.

References
----------
.. [1] Hansen, N., Finck, S., Ros, R., & Auger, A. (2009).
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


@register("LinearSlope")
@register("linear_slope")
class LinearSlopeFunction(OptimizationFunction):
    """
    Linear Slope function.

    f(x) = sum(5 * |s_i| - s_i * x_i)
    where s_i = sign_i * 10^((i-1)/(D-1))

    Global minimum: f(sign_i * 5, ...) = 0

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    sign_vector : np.ndarray, optional
        A vector of +1/-1 values determining slope directions.
        If None, defaults to all +1 (positive slopes).
    initialization_bounds : Bounds, optional
        Bounds for random initialization. Defaults to [-5, 5]^D.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. Defaults to [-5, 5]^D.
    """

    def __init__(
        self,
        dimension: int,
        sign_vector: NDArray | None = None,
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

        if sign_vector is not None:
            sv = np.asarray(sign_vector, dtype=np.float64)
            if sv.shape != (dimension,):
                raise ValueError(
                    f"sign_vector shape {sv.shape} doesn't match dimension {dimension}"
                )
            self._sign_vector = np.sign(sv)
            if np.any(self._sign_vector == 0):
                raise ValueError(
                    "sign_vector must not contain zeros (np.sign(0) = 0 breaks ±1 semantics)"
                )
        else:
            self._sign_vector = np.ones(dimension, dtype=np.float64)

        # Slope coefficients: s_i = sign_i * sqrt(10)^(i/(D-1))
        # Matches COCO convention: pow(sqrt(10.0), i / (D-1))
        if dimension == 1:
            exponents = np.array([0.0])
        else:
            exponents = np.arange(dimension, dtype=np.float64) / (dimension - 1)
        self._s = self._sign_vector * np.power(np.sqrt(10.0), exponents)

    def evaluate(self, x: NDArray) -> float:
        """Compute the Linear Slope function value."""
        x = self._validate_input(x)
        return float(np.sum(5.0 * np.abs(self._s) - self._s * x))

    def evaluate_batch(self, X: NDArray) -> NDArray:
        """Compute Linear Slope function for batch."""
        X = self._validate_batch_input(X)
        return np.sum(5.0 * np.abs(self._s) - self._s * X, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum point and value.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        global_min_point = self._sign_vector * 5.0
        global_min_value = 0.0
        return global_min_point, global_min_value
