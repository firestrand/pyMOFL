"""
MaxAbsolute function: returns max(|x_i|) for a vector x.

Can be used as a base function (dimension required) or as a wrapper
around a base function that returns a vector, in which case it returns
the max absolute value of the vector output.
"""

from __future__ import annotations

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("yao_liu_04")
class MaxAbsolute(OptimizationFunction):
    def __init__(
        self,
        dimension: int | None = None,
        base_function: OptimizationFunction | None = None,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
    ) -> None:
        if base_function is None and dimension is None:
            raise ValueError("Either dimension or base_function must be provided")
        if base_function is not None and dimension is None:
            dimension = base_function.dimension

        assert dimension is not None, "Either dimension or base_function must be provided"

        # Default bounds: [-100, 100]^d
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )
        self.base_function = base_function
        # Convenience bounds array to satisfy tests
        assert self.initialization_bounds is not None
        self.bounds = np.column_stack(
            (
                self.initialization_bounds.low,
                self.initialization_bounds.high,
            )
        )

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        if self.base_function is None:
            return float(np.max(np.abs(x)))
        # Base may return a vector (identity in tests) or scalar; handle both
        y = self.base_function.evaluate(x)
        y = np.asarray(y)
        if y.ndim == 0:
            return float(abs(y))
        return float(np.max(np.abs(y)))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        if self.base_function is None:
            return np.max(np.abs(X), axis=1)
        Y = self.base_function.evaluate_batch(X)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            return np.abs(Y)
        return np.max(np.abs(Y), axis=1)
