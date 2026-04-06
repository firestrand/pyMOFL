"""Cauchy (heavy-tailed additive) noise transform.

COCO formula: f_noisy = f + alpha * max(0, 1000 + I_{U<p} * N1/(|N2|+1e-99))
where U ~ Uniform(0,1), N1, N2 ~ Normal(0,1).
"""

import numpy as np

from .base import ScalarTransform


class CauchyNoiseTransform(ScalarTransform):
    """Heavy-tailed additive noise (COCO Cauchy noise model).

    The noise term is always non-negative (due to max(0, ...)),
    so f_noisy >= f.

    Parameters
    ----------
    alpha : float
        Noise intensity scale.
    p : float
        Probability of Cauchy-like perturbation (0 to 1).
    seed : int, optional
        Seed for np.random.Generator reproducibility.
    """

    def __init__(self, alpha: float, p: float, seed: int | None = None):
        self.alpha = alpha
        self.p = p
        self._rng = np.random.default_rng(seed)

    def __call__(self, y: float) -> float:
        u = self._rng.uniform()
        n1 = self._rng.standard_normal()
        n2 = self._rng.standard_normal()
        indicator = 1.0 if u < self.p else 0.0
        cauchy_term = indicator * n1 / (abs(n2) + 1e-99)
        noise = self.alpha * max(0.0, 1000.0 + cauchy_term)
        return float(y + noise)

    def transform_batch(self, Y: np.ndarray) -> np.ndarray:
        n = Y.shape[0]
        u = self._rng.uniform(size=n)
        n1 = self._rng.standard_normal(size=n)
        n2 = self._rng.standard_normal(size=n)
        indicator = (u < self.p).astype(float)
        cauchy_term = indicator * n1 / (np.abs(n2) + 1e-99)
        noise = self.alpha * np.maximum(0.0, 1000.0 + cauchy_term)
        return Y + noise

    def __repr__(self) -> str:
        return f"CauchyNoiseTransform(alpha={self.alpha}, p={self.p})"
