"""Per-coordinate source half-up quantization, separate from legacy np.rint."""

from typing import override

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import VectorTransform


class HalfUpQuantizationTransform(VectorTransform):
    """Map coordinate x to q*floor(.5+x/q) for q>1e-40; otherwise passthrough.

    Parameters
    ----------
    steps
        Finite nonnegative 1D per-coordinate steps, copied at construction.
        Zero denotes continuous input. This transform never clips bounds.
    """

    def __init__(self, steps: ArrayLike) -> None:
        self.steps = np.array(steps, dtype=np.float64, copy=True)
        if self.steps.ndim != 1 or not np.all(np.isfinite(self.steps)) or np.any(self.steps < 0):
            raise ValueError("steps must be a finite nonnegative 1D vector")

    @override
    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Quantize one vector without modifying its input."""
        if x.shape != self.steps.shape:
            raise ValueError("input must have the same shape as steps")
        return self.transform_batch(x[None, :])[0]

    @override
    def transform_batch(
        self, X: NDArray[np.float64], out: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Quantize rows with step-zero passthrough and optional output reuse."""
        if X.ndim != 2 or X.shape[1] != len(self.steps):
            raise ValueError("input must have shape (N, len(steps))")
        active = self.steps > 1e-40
        result = np.array(X, dtype=np.float64, copy=True)
        steps = self.steps[active]
        result[:, active] = steps * np.floor(0.5 + result[:, active] / steps)
        if out is not None:
            np.copyto(out, result)
            return out
        return result
