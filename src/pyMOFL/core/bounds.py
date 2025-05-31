"""
Bounds dataclass for representing variable bounds, quantization, and enforcement mode.
"""
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from .bound_mode_enum import BoundModeEnum
from .quantization_type_enum import QuantizationTypeEnum
from typing import Any

@dataclass(frozen=True)
class Bounds:
    """
    Represents bounds, quantization, and enforcement mode for optimization variables.

    Attributes
    ----------
    low : NDArray
        Lower bounds for each variable.
    high : NDArray
        Upper bounds for each variable.
    mode : BoundModeEnum
        Whether these bounds are for initialization or operational use.
    qtype : QuantizationTypeEnum
        Quantization type (continuous, integer, step).
    step : float
        Step size for STEP quantization (ignored otherwise).
    """
    low: NDArray[Any]
    high: NDArray[Any]
    mode: BoundModeEnum = BoundModeEnum.OPERATIONAL
    qtype: QuantizationTypeEnum = QuantizationTypeEnum.CONTINUOUS
    step: float = 1.0

    def project(self, x: NDArray[Any]) -> NDArray[Any]:
        """
        Repair helper that snaps/clips input into the legal region according to quantization and bounds.
        Supports per-variable quantization if qtype is an array.
        """
        x_proj = np.array(x, dtype=float)
        # Handle per-variable quantization
        if isinstance(self.qtype, np.ndarray) or (hasattr(self.qtype, '__len__') and not isinstance(self.qtype, str)):
            x_proj = x_proj.copy()
            for i, q in enumerate(self.qtype):
                if q == QuantizationTypeEnum.INTEGER:
                    x_proj[i] = np.rint(x_proj[i])
                elif q == QuantizationTypeEnum.STEP:
                    x_proj[i] = np.round((x_proj[i] - self.low[i]) / self.step) * self.step + self.low[i]
            return np.clip(x_proj, self.low, self.high)
        else:
            if self.qtype == QuantizationTypeEnum.INTEGER:
                x_proj = np.rint(x_proj)
            elif self.qtype == QuantizationTypeEnum.STEP:
                x_proj = np.round((x_proj - self.low) / self.step) * self.step + self.low
            return np.clip(x_proj, self.low, self.high) 