"""Uniform (scale-dependent) noise transform.

COCO formula: f_noisy = f * U^beta * max(1, (1e9/(|f|+1e-99))^(alpha * U'))
where U, U' ~ Uniform(0,1).
"""

import numpy as np

from .base import ScalarTransform


class UniformNoiseTransform(ScalarTransform):
    """Scale-dependent uniform noise (COCO uniform noise model).

    For large |f|, the max term ≈ 1 so noise ≈ f * U^beta.
    For small |f|, the (1e9/|f|)^(alpha*U') term amplifies.

    Parameters
    ----------
    alpha : float
        Controls scale-dependent amplification.
    beta : float
        Controls base multiplicative noise.
    seed : int, optional
        Seed for np.random.Generator reproducibility.
    """

    def __init__(self, alpha: float, beta: float, seed: int | None = None):
        self.alpha = alpha
        self.beta = beta
        self._rng = np.random.default_rng(seed)

    def __call__(self, y: float) -> float:
        if y == 0.0:
            return 0.0
        u1 = self._rng.uniform()
        u2 = self._rng.uniform()
        scale = u1**self.beta * max(1.0, (1e9 / (abs(y) + 1e-99)) ** (self.alpha * u2))
        return float(y * scale)

    def transform_batch(self, Y: np.ndarray) -> np.ndarray:
        result = np.copy(Y)
        nonzero = Y != 0.0
        n = int(np.sum(nonzero))
        if n > 0:
            u1 = self._rng.uniform(size=n)
            u2 = self._rng.uniform(size=n)
            abs_y = np.abs(Y[nonzero]) + 1e-99
            scale = u1**self.beta * np.maximum(1.0, (1e9 / abs_y) ** (self.alpha * u2))
            result[nonzero] = Y[nonzero] * scale
        return result

    def __repr__(self) -> str:
        return f"UniformNoiseTransform(alpha={self.alpha}, beta={self.beta})"
