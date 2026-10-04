"""Tripod variant with the pinned SPSO source's sign(0)=0 axis semantics."""

from typing import override

import numpy as np
from numpy.typing import NDArray

from pyMOFL.registry import register

from .tripod import TripodFunction


@register("spso_tripod")
class SPSOTripodFunction(TripodFunction):
    """Source-backed D2 Tripod; inherited bounds and optimum (0,-50), value 0.

    Unlike the generic Tripod, source sign(0)=0 gives half coefficients on
    axes. See docs/spso-reference-review.md for pinned source identities.
    """

    @override
    def evaluate(self, x: NDArray[np.float64]) -> float:
        """Evaluate a source-native vector of shape (2,)."""
        x = self._validate_input(x)
        return float(self.evaluate_batch(x[None, :])[0])

    @override
    def evaluate_batch(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate source-native rows without modifying input."""
        X = self._validate_batch_input(X)
        x1, x2 = X[:, 0], X[:, 1]
        s11, s12 = (1 - np.sign(x1)) / 2, (1 + np.sign(x1)) / 2
        s21, s22 = (1 - np.sign(x2)) / 2, (1 + np.sign(x2)) / 2
        return s21 * (np.abs(x1) + np.abs(x2 + 50)) + s22 * (
            s11 * (1 + np.abs(x1 + 50) + np.abs(x2 - 50))
            + s12 * (2 + np.abs(x1 - 50) + np.abs(x2 - 50))
        )
