"""
Power transformation for scalar function outputs.

A scalar-to-scalar transformation that applies a power operation to the output.
Formula: y' = y^exponent
"""

from __future__ import annotations

import numpy as np

from .base import ScalarTransform


class PowerTransform(ScalarTransform):
    """
    Power transformation for scalar function outputs.

    Computes y' = y^exponent.
    Used in GNBG functions for applying basin linearity (lambda parameter).

    Parameters
    ----------
    exponent : float
        The exponent to apply to the scalar value.
    """

    def __init__(self, exponent: float):
        """
        Initialize the power transform.

        Args:
            exponent: The exponent to apply.
        """
        self.exponent = float(exponent)

    def __call__(self, y: float) -> float:
        """Apply the power transformation."""
        # Using np.power to handle potential edge cases gracefully
        # In GNBG, y is always non-negative (sum of squares).
        return float(np.power(max(0.0, float(y)), self.exponent))

    def transform_batch(self, Y: np.ndarray) -> np.ndarray:
        """Apply the power transformation to a batch of scalars."""
        Y = np.asarray(Y, dtype=np.float64)
        # Ensure non-negative inputs as in GNBG context
        return np.power(np.maximum(0.0, Y), self.exponent)
