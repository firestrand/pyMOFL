"""
MaxAbsolute: Unified base/decorator for max-absolute transformation.

This class computes f(x) = max(|x|) as a base function, or f(x) = max(|base(x)|) as a decorator.
Useful for CEC and other benchmarks where the max-absolute value is a transformation or a base function.

References:
    - Used in Schwefel's Problem 2.6 and other composite benchmarks.

Usage:
    # As a base function
    f = MaxAbsolute(dimension=10, ...)
    value = f(x)

    # As a decorator
    base = SomeFunction(...)
    f = MaxAbsolute(base_function=base)
    value = f(x)

Note:
    This class is composable with other decorators (bias, shift, etc.).
    For scalar base functions, this is equivalent to abs(base(x)).
"""

import numpy as np
from ..core.composable_function import OutputTransformingFunction
from ..core.bounds import Bounds
from ..core.bound_mode_enum import BoundModeEnum
from ..core.quantization_type_enum import QuantizationTypeEnum

class MaxAbsolute(OutputTransformingFunction):
    """
    MaxAbsolute: Output-transforming decorator for max-absolute transformation.

    Inherits from OutputTransformingFunction. Subclasses should only implement _apply and _apply_batch, never override evaluate or evaluate_batch.
    """
    def __init__(self, base_function=None, dimension=None, initialization_bounds=None, operational_bounds=None):
        if base_function is None:
            assert dimension is not None, "Must specify dimension if no base_function"
            default_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS
            )
            if initialization_bounds is None:
                initialization_bounds = default_bounds
            if operational_bounds is None:
                operational_bounds = default_bounds
        super().__init__(base_function=base_function, dimension=dimension, initialization_bounds=initialization_bounds, operational_bounds=operational_bounds)

    def _apply(self, y):
        y = np.asarray(y)
        if y.ndim == 0:
            return float(abs(y))
        return float(np.max(np.abs(y)))

    def _apply_batch(self, Y):
        Y = np.asarray(Y)
        if Y.ndim == 1:
            return np.abs(Y)
        return np.max(np.abs(Y), axis=1) 