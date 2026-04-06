"""LinearConstraint — g(x) = a^T (x - shift * a/||a||) <= 0."""

from __future__ import annotations

import numpy as np


class LinearConstraint:
    """A single linear inequality constraint.

    g(x) = a^T (x - shift * a_hat) <= 0

    where a_hat = a / ||a|| is the unit normal.

    Parameters
    ----------
    normal : np.ndarray
        Constraint normal vector a.
    shift : float
        Distance along normal to shift the boundary.
        shift=0 means the boundary passes through the origin.
    is_active : bool
        Whether this constraint is active (binding at optimum).
    """

    def __init__(
        self,
        normal: np.ndarray,
        shift: float = 0.0,
        is_active: bool = True,
    ):
        self.normal = np.asarray(normal, dtype=np.float64)
        self.shift = float(shift)
        self.is_active = is_active
        norm = np.linalg.norm(self.normal)
        self._unit_normal = self.normal / norm if norm > 0 else self.normal
        self._shift_vector = self.shift * self._unit_normal

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate constraint value g(x).

        Returns <= 0 if feasible, > 0 if infeasible.
        """
        return float(self.normal @ (x - self._shift_vector))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate constraint for multiple points."""
        shifted = X - self._shift_vector
        return shifted @ self.normal

    def is_feasible(self, x: np.ndarray) -> bool:
        """Check if point satisfies this constraint."""
        return self.evaluate(x) <= 0.0

    def __repr__(self) -> str:
        return (
            f"LinearConstraint(dim={len(self.normal)}, shift={self.shift}, active={self.is_active})"
        )
