"""
Quantized wrapper: applies input quantization before evaluating base function.

Supports integer quantization and step-based quantization.
"""

from __future__ import annotations

import numpy as np

from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum


class Quantized(OptimizationFunction):
    def __init__(
        self,
        base_function: OptimizationFunction,
        qtype: QuantizationTypeEnum = QuantizationTypeEnum.CONTINUOUS,
        step: float | None = None,
    ) -> None:
        super().__init__(
            dimension=base_function.dimension,
            initialization_bounds=base_function.initialization_bounds,
            operational_bounds=base_function.operational_bounds,
        )
        self.base_function = base_function
        self.qtype = qtype
        self.step = step

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        if self.qtype == QuantizationTypeEnum.CONTINUOUS:
            return x
        if self.qtype == QuantizationTypeEnum.INTEGER:
            return np.rint(x)
        # STEP: quantize to nearest multiple of step if provided, else passthrough
        if self.qtype == QuantizationTypeEnum.STEP and self.step:
            return np.rint(x / self.step) * self.step
        return x

    def evaluate(self, x: np.ndarray) -> float:
        x = self._validate_input(x)
        z = self._quantize(x)
        return float(self.base_function.evaluate(z))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_batch_input(X)
        Z = self._quantize(X)
        return self.base_function.evaluate_batch(Z)
