"""
Attractive Sector function (BBOB f6 base).

f(x) = sum((s_i * x_i)^2)^0.9
where s_i = 100 if x_i * x_opt_i > 0, else 1.

Only a hypercone of roughly (1/2)^D volume fraction of the search space
has the favorable s_i = 100 scaling, making the function highly asymmetric.

The T_osz transformation and Q*Lambda*R rotations are not included in
this base function — they are applied externally via ComposedFunction
when constructing the full BBOB f6 suite function.

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


@register("AttractiveSector")
@register("attractive_sector")
class AttractiveSectorFunction(OptimizationFunction):
    """
    Attractive Sector function.

    f(x) = sum((s_i * x_i)^2)^0.9
    where s_i = 100 if x_i * x_opt_i > 0, else 1.

    Global minimum: f(0) = 0

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    x_opt : np.ndarray, optional
        Reference direction for sector weighting. Defaults to ones(D).
    initialization_bounds : Bounds, optional
        Bounds for random initialization. Defaults to [-5, 5]^D.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. Defaults to [-5, 5]^D.
    """

    def __init__(
        self,
        dimension: int,
        x_opt: NDArray | None = None,
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

        if x_opt is not None:
            self._x_opt = np.asarray(x_opt, dtype=np.float64)
            if self._x_opt.shape != (dimension,):
                raise ValueError(
                    f"x_opt shape {self._x_opt.shape} doesn't match dimension {dimension}"
                )
        else:
            self._x_opt = np.ones(dimension, dtype=np.float64)

    def evaluate(self, x: NDArray) -> float:
        """Compute the Attractive Sector function value."""
        x = self._validate_input(x)
        s = np.where(x * self._x_opt > 0, 100.0, 1.0)
        inner_sum = np.sum((s * x) ** 2)
        return float(inner_sum**0.9)

    def evaluate_batch(self, X: NDArray) -> NDArray:
        """Compute Attractive Sector function for batch."""
        X = self._validate_batch_input(X)
        s = np.where(X * self._x_opt > 0, 100.0, 1.0)
        inner_sums = np.sum((s * X) ** 2, axis=1)
        return inner_sums**0.9

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
