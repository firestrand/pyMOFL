"""
QuantizedFunction decorator for applying quantization to an OptimizationFunction.
"""
from typing import Any
import numpy as np
from numpy.typing import NDArray
from .function import OptimizationFunction
from .bounds import Bounds
from .quantization_type_enum import QuantizationTypeEnum

class QuantizedFunction(OptimizationFunction):
    """
    Decorator that applies quantization (integer or step) to an OptimizationFunction.
    """
    def __init__(self, base_function: OptimizationFunction, qtype: QuantizationTypeEnum, step: float = 1.0):
        self.base_function = base_function
        # Replace quantization type and step in operational_bounds
        base_op_bounds = base_function.operational_bounds
        if base_op_bounds is not None:
            self.operational_bounds = Bounds(
                low=base_op_bounds.low,
                high=base_op_bounds.high,
                mode=base_op_bounds.mode,
                qtype=qtype,
                step=step
            )
        else:
            self.operational_bounds = None
        self.initialization_bounds = base_function.initialization_bounds
        self.constraint_penalty = base_function.constraint_penalty

    def evaluate(self, z: NDArray[Any]) -> float:
        return self.base_function(self.operational_bounds.project(z) if self.operational_bounds else z)

    def violations(self, x: NDArray[Any]) -> float:
        return self.base_function.violations(x) 