"""
Noise transformation for adding noise to function output.

Based on CEC 2005 benchmark specification.
"""

import numpy as np

from ._rng import resolve_generator
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
        Legacy process-global random seed. Mutually exclusive with rng.
    rng : numpy.random.Generator, optional
        Explicit caller-owned stream; avoids changing global random state.
    """

    def __init__(
        self,
        noise_level: float = 0.4,
        seed: int | None = None,
        *,
        rng: np.random.Generator | None = None,
    ):
        """
        Initialize noise transform.

        Args:
            noise_level: Noise level coefficient (default 0.4)
            seed: Random seed for reproducibility
            rng: Explicit stream, mutually exclusive with seed
        """
        self.noise_level = noise_level
        self._rng = resolve_generator(seed, rng) if rng is not None else None
        if seed is not None and rng is None:
            np.random.seed(seed)

    def _normal(self, shape: tuple[int, ...] | None = None) -> float | np.ndarray:
        """Preserve legacy draws or consume the explicitly supplied stream."""
        if self._rng is None:
            return np.random.randn(*(shape or ()))
        return self._rng.standard_normal(shape)

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
            noise_factor = 1.0 + self.noise_level * np.abs(self._normal(value.shape))
        else:
            noise_factor = 1.0 + self.noise_level * np.abs(self._normal())

        return value * noise_factor

    def transform_batch(self, Y: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """
        Apply noise to batch of scalar values.

        Args:
            Y: Array of function values
            out: Optional pre-allocated buffer of shape matching Y

        Returns:
            Values with noise applied
        """
        Y_arr = np.asarray(Y, dtype=np.float64)
        noise_factor = 1.0 + self.noise_level * np.abs(self._normal(Y_arr.shape))
        return np.multiply(Y_arr, noise_factor, out=out)

    def __repr__(self) -> str:
        return f"NoiseTransform(noise_level={self.noise_level})"
