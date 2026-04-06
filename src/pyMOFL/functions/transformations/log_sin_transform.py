"""
Log-space sinusoidal transformation.

A vector-to-vector transformation that introduces sinusoidal modulation
in the logarithmic space of element magnitudes. This is used to introduce
ruggedness and asymmetry.

Mathematical form (element-wise):
  T(0) = 0
  T(a > 0) = exp(ln(a) + mu_1 * sin(omega_1 * ln(a)) + sin(omega_2 * ln(a)))
  T(a < 0) = -exp(ln(|a|) + mu_2 * sin(omega_3 * ln(|a|)) + sin(omega_4 * ln(|a|)))
"""

from __future__ import annotations

import numpy as np

from .base import VectorTransform


class LogSinTransform(VectorTransform):
    """
    Log-space sinusoidal transformation.

    Introduces configurable ruggedness and asymmetry into a landscape
    by modulating magnitudes via logarithmic sine terms.

    Parameters
    ----------
    mu : tuple[float, float], default=(1.0, 1.0)
        Multipliers for the primary sine term (positive, negative signs).
    omega : tuple[float, float, float, float], default=(1.0, 1.0, 1.0, 1.0)
        Frequencies for the sine terms (primary_pos, secondary_pos, primary_neg, secondary_neg).
    """

    def __init__(
        self,
        mu: tuple[float, float] = (1.0, 1.0),
        omega: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ):
        """Initialize the log-sin transform."""
        self.mu = np.asarray(mu, dtype=np.float64)
        self.omega = np.asarray(omega, dtype=np.float64)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply the transform."""
        x = np.asarray(x, dtype=np.float64)
        return self._apply(x)

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """Apply the transform to a batch."""
        X = np.asarray(X, dtype=np.float64)
        return self._apply(X)

    def _apply(self, arr: np.ndarray) -> np.ndarray:
        """Vectorized element-wise application."""
        result = np.zeros_like(arr, dtype=np.float64)
        nonzero = arr != 0.0
        if not np.any(nonzero):
            return result

        vals = arr[nonzero]
        abs_vals = np.abs(vals)
        ln_vals = np.log(abs_vals)

        pos_mask = vals > 0.0
        neg_mask = vals < 0.0

        mu1, mu2 = self.mu
        w1, w2, w3, w4 = self.omega

        transformed_vals = np.empty_like(vals)

        if np.any(pos_mask):
            ln_p = ln_vals[pos_mask]
            transformed_vals[pos_mask] = np.exp(ln_p + mu1 * np.sin(w1 * ln_p) + np.sin(w2 * ln_p))

        if np.any(neg_mask):
            ln_n = ln_vals[neg_mask]
            transformed_vals[neg_mask] = -np.exp(ln_n + mu2 * np.sin(w3 * ln_n) + np.sin(w4 * ln_n))

        result[nonzero] = transformed_vals
        return result
