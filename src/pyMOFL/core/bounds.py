"""
Bounds dataclass for representing variable bounds as metadata only.
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
    Represents bounds for optimization variables as metadata only.

    Attributes
    ----------
    low : NDArray
        Lower bounds for each variable.
    high : NDArray
        Upper bounds for each variable.
    mode : BoundModeEnum
        Whether these bounds are for initialization or operational use.
    qtype : QuantizationTypeEnum
        Quantization type (continuous, integer, step). Metadata only; not enforced.
    step : float
        Step size for STEP quantization (metadata only).
    
    Note
    ----
    This class is a pure data object. It does not perform any enforcement, quantization, or projection.
    Enforcement/quantization is opt-in via the Quantized decorator.
    """
    low: NDArray[Any]
    high: NDArray[Any]
    mode: BoundModeEnum = BoundModeEnum.OPERATIONAL
    qtype: QuantizationTypeEnum = QuantizationTypeEnum.CONTINUOUS
    step: float = 1.0 