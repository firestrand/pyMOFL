"""
Boundary penalty (f_pen from Hansen et al. 2009, BBOB).

A PenaltyTransform (vector→scalar) that computes a quadratic penalty
for elements exceeding a boundary.
Formula: f_pen(x) = sum(max(0, |x_i| - bound)^2)
"""

import numpy as np

from .base import PenaltyTransform


class BoundaryPenaltyTransform(PenaltyTransform):
    """BBOB boundary penalty f_pen.

    Parameters
    ----------
    bound : float
        Elements with |x_i| > bound incur penalty. Default 5.0.
    """

    def __init__(self, bound: float = 5.0) -> None:
        self.bound = float(bound)

    def __call__(self, x: np.ndarray) -> float:
        """Compute penalty for a single vector."""
        x = np.asarray(x, dtype=np.float64)
        violations = np.maximum(0.0, np.abs(x) - self.bound)
        return float(np.sum(violations**2))

    def compute_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute penalties for a batch of vectors.

        Returns
        -------
        ndarray of shape (N,)
        """
        X = np.asarray(X, dtype=np.float64)
        violations = np.maximum(0.0, np.abs(X) - self.bound)
        return np.sum(violations**2, axis=1)
