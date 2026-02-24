"""
Noise transformation for adding noise to function output.

Based on CEC 2005 benchmark specification.
"""

import numpy as np

from .base import ScalarTransform


class NoiseTransform(ScalarTransform):
    """
    Adds noise to the function output.

    The CEC 2005 functions F4, F17, F24, F25 use noise defined as:
    f_noisy(x) = f(x) * (1 + 0.4 * |N(0,1)|)

    where N(0,1) is a standard normal random variable.

    Parameters
    ----------
    noise_level : float
        Noise level coefficient (default 0.4 for CEC 2005)
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(self, noise_level: float = 0.4, seed: int | None = None):
        """
        Initialize noise transform.

        Args:
            noise_level: Noise level coefficient (default 0.4)
            seed: Random seed for reproducibility
        """
        self.noise_level = noise_level
        if seed is not None:
            np.random.seed(seed)

    def __call__(self, value: float | np.ndarray) -> float | np.ndarray:  # type: ignore[override]
        """
        Apply noise to scalar value(s).

        Args:
            value: Scalar or array of function values

        Returns:
            Value(s) with noise applied
        """
        # Generate noise using absolute value of normal distribution
        # This matches the CEC 2005 specification: 1 + 0.4 * |N(0,1)|
        if isinstance(value, np.ndarray):
            noise_factor = 1.0 + self.noise_level * np.abs(np.random.randn(*value.shape))
        else:
            noise_factor = 1.0 + self.noise_level * np.abs(np.random.randn())

        return value * noise_factor

    def __repr__(self) -> str:
        return f"NoiseTransform(noise_level={self.noise_level})"
