"""
Quantized: Input-transforming decorator or base for quantization (integer or step) of input variables.

Supports both base and decorator usage. Inherits from InputTransformingFunction.

Usage:
    # As a decorator
    base = SphereFunction(...)
    f = Quantized(base_function=base, qtype=QuantizationTypeEnum.INTEGER)
    value = f(x)

    # As a base function
    f = Quantized(dimension=3, qtype=QuantizationTypeEnum.STEP, step=0.5)
    value = f(x)

Supports per-variable quantization if qtype is an array.
"""

import numpy as np
from typing import Optional, Any
from pyMOFL.core.composable_function import InputTransformingFunction
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction

class Quantized(InputTransformingFunction):
    """
    Input-transforming decorator or base for quantizing input variables before evaluation.

    Parameters
    ----------
    base_function : OptimizationFunction, optional
        The function to decorate. If None, acts as a base function.
    dimension : int, optional
        The dimensionality (required if base_function is None).
    qtype : QuantizationTypeEnum or np.ndarray
        Quantization type (integer, step, or per-variable array).
    step : float or np.ndarray, optional
        Step size for STEP quantization (ignored otherwise). Default is 1.0.
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, uses base function's bounds or defaults.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, updates base function's bounds with new qtype/step or defaults.
    """
    def __init__(self,
                 base_function: Optional[OptimizationFunction] = None,
                 dimension: Optional[int] = None,
                 qtype: Any = QuantizationTypeEnum.INTEGER,
                 step: Any = 1.0,
                 initialization_bounds: Optional[Bounds] = None,
                 operational_bounds: Optional[Bounds] = None):
        if base_function is not None:
            # Decorator mode: always create new operational_bounds with requested quantization
            base_op_bounds = base_function.operational_bounds
            if base_op_bounds is not None:
                op_bounds = Bounds(
                    low=base_op_bounds.low.copy(),
                    high=base_op_bounds.high.copy(),
                    mode=base_op_bounds.mode,
                    qtype=qtype,
                    step=step,
                )
            else:
                dim = getattr(base_function, 'dimension', None)
                assert dim is not None, "Base function must have dimension if no operational_bounds"
                op_bounds = Bounds(
                    low=np.full(dim, -100.0),
                    high=np.full(dim, 100.0),
                    mode=BoundModeEnum.OPERATIONAL,
                    qtype=qtype,
                    step=step,
                )
            super().__init__(
                base_function=base_function,
                initialization_bounds=initialization_bounds or base_function.initialization_bounds,
                operational_bounds=operational_bounds or op_bounds,
            )
        else:
            # Base mode: require dimension and bounds
            assert dimension is not None, "Must specify dimension if no base_function"
            if initialization_bounds is None:
                initialization_bounds = Bounds(
                    low=np.full(dimension, -100.0),
                    high=np.full(dimension, 100.0),
                    mode=BoundModeEnum.INITIALIZATION,
                    qtype=qtype,
                    step=step,
                )
            if operational_bounds is None:
                operational_bounds = Bounds(
                    low=np.full(dimension, -100.0),
                    high=np.full(dimension, 100.0),
                    mode=BoundModeEnum.OPERATIONAL,
                    qtype=qtype,
                    step=step,
                )
            super().__init__(
                base_function=None,
                dimension=dimension,
                initialization_bounds=initialization_bounds,
                operational_bounds=operational_bounds,
            )
        self.qtype = qtype
        self.step = step

    def _apply(self, x: np.ndarray) -> np.ndarray:
        """
        Quantize the input vector according to qtype and step.

        Parameters
        ----------
        x : np.ndarray
            Input vector.
        Returns
        -------
        np.ndarray
            Quantized input vector.
        """
        return self.operational_bounds.project(x)

    def _apply_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Quantize a batch of input vectors according to qtype and step.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_points, dimension).
        Returns
        -------
        np.ndarray
            Quantized input array.
        """
        return np.vstack([self._apply(x) for x in X]) 