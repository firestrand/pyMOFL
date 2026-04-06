"""
Sharp Ridge function (BBOB f13 base).

f(x) = x_1^2 + 100 * sqrt(sum(x_i^2, i=2..D))

The ridge along x_2 = ... = x_D = 0 is non-differentiable due to the
sqrt. The gradient toward the ridge does not flatten out, making
gradient-based approaches ineffective near the ridge.

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


@register("SharpRidge")
@register("sharp_ridge")
class SharpRidgeFunction(OptimizationFunction):
    """
    Sharp Ridge function.

    f(x) = x_1^2 + 100 * sqrt(sum(x_i^2, i=2..D))

    Global minimum: f(0, ..., 0) = 0

    Parameters
    ----------
    dimension : int
        The dimensionality of the function (must be >= 2).
    initialization_bounds : Bounds, optional
        Bounds for random initialization. Defaults to [-5, 5]^D.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. Defaults to [-5, 5]^D.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension < 2:
            raise ValueError("SharpRidgeFunction requires dimension >= 2")
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

    def evaluate(self, x: NDArray) -> float:
        """Compute the Sharp Ridge function value."""
        x = self._validate_input(x)
        return float(x[0] ** 2 + 100.0 * np.sqrt(np.sum(x[1:] ** 2)))

    def evaluate_batch(self, X: NDArray) -> NDArray:
        """Compute Sharp Ridge function for batch."""
        X = self._validate_batch_input(X)
        return X[:, 0] ** 2 + 100.0 * np.sqrt(np.sum(X[:, 1:] ** 2, axis=1))

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """
        Get the global minimum point and value.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        global_min_point = np.zeros(self.dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value
