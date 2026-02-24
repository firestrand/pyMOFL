"""
Core components for the pyMOFL optimization framework.

This module provides the fundamental building blocks:
- OptimizationFunction: Base class for all optimization functions
- Bounds: Metadata for function bounds
- BoundModeEnum: Initialization vs operational bounds
- QuantizationTypeEnum: Continuous, integer, or binary domains
"""

from .bound_mode_enum import BoundModeEnum
from .bounds import Bounds
from .function import OptimizationFunction
from .quantization_type_enum import QuantizationTypeEnum

__all__ = [
    "BoundModeEnum",
    "Bounds",
    "OptimizationFunction",
    "QuantizationTypeEnum",
]
