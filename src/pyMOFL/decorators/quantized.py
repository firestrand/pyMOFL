from __future__ import annotations
from typing import Optional, Any
import numpy as np

from pyMOFL.core.composable_function       import InputTransformingFunction
from pyMOFL.core.bounds                    import Bounds
from pyMOFL.core.bound_mode_enum           import BoundModeEnum
from pyMOFL.core.quantization_type_enum    import QuantizationTypeEnum
from pyMOFL.core.function                  import OptimizationFunction


class Quantized(InputTransformingFunction):
    """
    Decorator / base that **only quantises** its input.

    • No clipping or projection is done here.  
    • When `base_function` is supplied and `operational_bounds` is None, the
      decorator defers to `base_function.operational_bounds`.  
    • If bounds are still absent, quantisation of STEP uses 0-centred lattice.
    """

    # ────────────────────────────────────────────────────────────────
    # constructor
    # ────────────────────────────────────────────────────────────────
    def __init__(self,
                 base_function: Optional[OptimizationFunction] = None,
                 dimension:     Optional[int]                 = None,
                 qtype: Any = QuantizationTypeEnum.INTEGER,
                 step:  Any = 1.0,
                 initialization_bounds: Optional[Bounds] = None,
                 operational_bounds:   Optional[Bounds] = None):

        self.qtype = qtype
        self.step  = step

        if base_function is not None:                               # decorator mode
            op_bounds = (operational_bounds
                          if operational_bounds is not None
                          else base_function.operational_bounds)    # ← defer
            super().__init__(
                base_function         = base_function,
                initialization_bounds = initialization_bounds or base_function.initialization_bounds,
                operational_bounds    = op_bounds,                  # may be None
            )
        else:                                                        # base-function mode
            if dimension is None:
                raise ValueError("Must specify 'dimension' when no base_function.")

            if initialization_bounds is None:
                initialization_bounds = Bounds(
                    low = np.full(dimension, -100.0),
                    high= np.full(dimension,  100.0),
                    mode= BoundModeEnum.INITIALIZATION,
                    qtype=qtype, step=step,
                )

            # honour caller’s explicit choice: op_bounds may be None
            super().__init__(
                base_function         = None,
                dimension             = dimension,
                initialization_bounds = initialization_bounds,
                operational_bounds    = operational_bounds,         # may be None
            )

    # ────────────────────────────────────────────────────────────────
    # helper: round-half-away-from-zero
    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _nearest(v: np.ndarray, s: np.ndarray) -> np.ndarray:
        return np.sign(v) * np.floor(np.abs(v) / s + 0.5) * s

    # ────────────────────────────────────────────────────────────────
    # single-vector quantisation
    # ────────────────────────────────────────────────────────────────
    def _apply(self, x: np.ndarray) -> np.ndarray:
        """
        Quantise `x` according to qtype/step and return it **without clipping**.
        """
        xq    = np.asarray(x, dtype=float)
        qtype = self.qtype
        step  = np.asarray(self.step)

        # reference origin for STEP mode
        if self.operational_bounds is not None:
            low_ref = np.asarray(self.operational_bounds.low)
        else:
            low_ref = 0.0                        # unbounded → centre lattice at 0

        # per-variable qtype
        if isinstance(qtype, (list, tuple, np.ndarray)):
            xq = xq.copy()
            for i, qt in enumerate(qtype):
                if qt == QuantizationTypeEnum.INTEGER:
                    xq[i] = np.rint(xq[i])
                elif qt == QuantizationTypeEnum.STEP:
                    origin = low_ref if np.isscalar(low_ref) else low_ref[i]
                    st     = step    if np.isscalar(step)    else step[i]
                    xq[i] = self._nearest(xq[i] - origin, st) + origin
        else:
            if qtype == QuantizationTypeEnum.INTEGER:
                xq = np.rint(xq)
            elif qtype == QuantizationTypeEnum.STEP:
                xq = self._nearest(xq - low_ref, step) + low_ref

        return xq   # ← never clipped

    # ────────────────────────────────────────────────────────────────
    # batch wrapper
    # ────────────────────────────────────────────────────────────────
    def _apply_batch(self, X: np.ndarray) -> np.ndarray:
        return np.vstack([self._apply(x) for x in X])
