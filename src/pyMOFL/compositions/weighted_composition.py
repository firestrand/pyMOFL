"""
Weighted composition of multiple optimization functions.

Provides a generic, distance-based weighted composition that supports
Gaussian weighting with optional dominance suppression (CEC-style),
per-component biases, and non-continuous preprocessing.
"""

from __future__ import annotations

import numpy as np

from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction


class WeightedComposition(OptimizationFunction):
    """
    Generic weighted composition of multiple optimization functions.

    Computes Gaussian weights based on distance from input point to each
    component's optimum, then returns a weighted sum of component outputs.

    Parameters
    ----------
    dimension : int
        Dimensionality of the search space.
    components : list[OptimizationFunction]
        Component functions, each fully self-contained (may include
        internal transforms and normalization via ComposedFunction).
    optima : list[np.ndarray]
        Optimum point for each component, used for Gaussian weight
        computation. Shape of each array: (dimension,).
    sigmas : list[float]
        Gaussian width parameter per component.
    biases : list[float] or None
        Per-component bias added inside the weight multiplication.
        If None, defaults to zeros.
    global_bias : float
        Constant added to the total weighted sum.
    dominance_suppression : bool
        If True, applies CEC-style winner-takes-most suppression
        to the Gaussian weights before normalization.
    non_continuous : bool
        If True, applies non-continuous rounding to the input vector
        before weight computation and all component evaluations.
        Rounding is centered on optima[0].
    initialization_bounds : Bounds or None
        Bounds for initialization.
    operational_bounds : Bounds or None
        Bounds for operation.
    """

    def __init__(
        self,
        *,
        dimension: int,
        components: list[OptimizationFunction],
        optima: list[np.ndarray],
        sigmas: list[float],
        biases: list[float] | None = None,
        global_bias: float = 0.0,
        dominance_suppression: bool = False,
        non_continuous: bool = False,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
    ) -> None:
        ncomp = len(components)
        if len(optima) != ncomp:
            raise ValueError("components and optima length mismatch")
        if len(sigmas) != ncomp:
            raise ValueError("components and sigmas length mismatch")
        if biases is not None and len(biases) != ncomp:
            raise ValueError("components and biases length mismatch")

        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

        self.components = components
        self.optima = [np.asarray(o, dtype=np.float64) for o in optima]
        self.sigmas = [float(s) for s in sigmas]
        self.biases = [float(b) for b in biases] if biases is not None else [0.0] * ncomp
        self.global_bias = float(global_bias)
        self.dominance_suppression = bool(dominance_suppression)
        self.non_continuous = bool(non_continuous)

    def _compute_weights(self, x: np.ndarray) -> np.ndarray:
        """Compute Gaussian weights from raw distances to component optima."""
        n = len(self.components)
        d2 = np.array(
            [np.sum((x - self.optima[i]) ** 2) for i in range(n)],
            dtype=np.float64,
        )
        w = np.exp(-(d2 / (2.0 * self.dimension * (np.array(self.sigmas) ** 2))))

        if self.dominance_suppression:
            maxw = np.max(w)
            if maxw > 0.0:
                mask = w != maxw
                w[mask] *= 1.0 - maxw**10.0

        s = np.sum(w)
        if s > 0.0:
            w = w / s
        else:
            w[:] = 1.0 / n
        return w

    def _noncontinuous_map(self, x: np.ndarray) -> np.ndarray:
        """Non-continuous rounding centered on first optimum.

        For dimensions where |x_j - o1_j| >= 0.5, applies round(2*x_j)/2
        using C-style round-half-away-from-zero.
        """
        o1 = self.optima[0]
        y = x.copy()
        mask = np.abs(x - o1) >= 0.5
        y[mask] = np.copysign(np.floor(np.abs(2.0 * x[mask]) + 0.5), x[mask]) / 2.0
        return y

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the weighted composition at point x."""
        x = self._validate_input(x)

        # Non-continuous preprocessing: applied to all weights and components
        x_eval = self._noncontinuous_map(x) if self.non_continuous else x

        w = self._compute_weights(x_eval)

        total = self.global_bias
        for i, comp in enumerate(self.components):
            f = comp.evaluate(x_eval)
            total += float(w[i] * (f + self.biases[i]))
        return float(total)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the weighted composition for a batch of points."""
        X = self._validate_batch_input(X)
        return np.array([self.evaluate(row) for row in X], dtype=np.float64)
