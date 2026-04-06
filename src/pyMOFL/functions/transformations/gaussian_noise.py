"""Gaussian (log-normal multiplicative) noise transform.

COCO formula: f_noisy = f * exp(beta * N(0,1))
"""

import numpy as np

from .base import ScalarTransform


class GaussianNoiseTransform(ScalarTransform):
    """Log-normal multiplicative noise (COCO Gaussian noise model).

    Applies: f_noisy = f * exp(beta * N(0,1))

    Parameters
    ----------
    beta : float
        Noise intensity. beta=0 gives identity.
    seed : int, optional
        Seed for np.random.Generator reproducibility.
    """

    def __init__(self, beta: float, seed: int | None = None):
        self.beta = beta
        self._rng = np.random.default_rng(seed)

    def __call__(self, y: float) -> float:
        n = self._rng.standard_normal()
        return float(y * np.exp(self.beta * n))

    def transform_batch(self, Y: np.ndarray) -> np.ndarray:
        n = self._rng.standard_normal(Y.shape)
        return Y * np.exp(self.beta * n)

    def __repr__(self) -> str:
        return f"GaussianNoiseTransform(beta={self.beta})"
