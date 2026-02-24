"""
Bias transformation - adds constant to output.

A scalar-to-scalar transformation function.
Mathematical form: bias(y) = y + bias_value
"""

import numpy as np

from .base import ScalarTransform


class BiasTransform(ScalarTransform):
    """
    Bias transformation function.

    Takes a scalar input and returns a biased scalar.
    """

    def __init__(self, bias: float):
        """
        Initialize bias transformation.

        Args:
            bias: The bias value to add
        """
        self.bias = float(bias)

    def __call__(self, y: float) -> float:
        """
        Apply bias transformation.

        Args:
            y: Input scalar

        Returns:
            Biased scalar: y + bias
        """
        return y + self.bias

    def transform_batch(self, Y: np.ndarray) -> np.ndarray:
        """
        Apply bias transformation to batch.

        Args:
            Y: Batch of scalars

        Returns:
            Batch of biased scalars: Y + bias
        """
        return np.asarray(Y) + self.bias
