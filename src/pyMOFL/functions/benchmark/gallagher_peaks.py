"""
Gallagher's Gaussian Peaks function (BBOB f21/f22 base).

f(x) = 10 - max_i(w_i * exp(-1/(2D) * (x - y_i)^T C_i (x - y_i)))

Creates n_peaks Gaussian peaks with random positions, weights, and
per-peak conditioning matrices. Parametrized by n_peaks:
  - 101 peaks for BBOB f21
  - 21 peaks for BBOB f22

References
----------
.. [1] Hansen, N., Finck, S., Ros, R., & Auger, A. (2009).
       "Real-Parameter Black-Box Optimization Benchmarking 2009:
       Noiseless Functions Definitions." INRIA Technical Report RR-6829.
"""

import numpy as np
from numpy.typing import NDArray

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("GallagherPeaks")
@register("gallagher_peaks")
class GallagherPeaksFunction(OptimizationFunction):
    """
    Gallagher's Gaussian Peaks function.

    f(x) = 10 - max_i(w_i * exp(-1/(2D) * (x - y_i)^T C_i (x - y_i)))

    Global minimum: f(y_0) = 0 (at the first peak, which has weight w_0 = 10).

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    n_peaks : int
        Number of Gaussian peaks (101 for BBOB f21, 21 for BBOB f22).
    seed : int
        Random seed for reproducible peak generation.
    initialization_bounds : Bounds, optional
        Bounds for random initialization. Defaults to [-5, 5]^D.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. Defaults to [-5, 5]^D.
    """

    def __init__(
        self,
        dimension: int,
        n_peaks: int = 101,
        seed: int = 0,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if n_peaks < 1:
            raise ValueError(f"n_peaks must be >= 1, got {n_peaks}")
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -5.0),
                high=np.full(dimension, 5.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -5.0),
                high=np.full(dimension, 5.0),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

        self._n_peaks = n_peaks
        self._seed = seed
        self._generate_peaks(dimension, n_peaks, seed)

    def _generate_peaks(self, dimension: int, n_peaks: int, seed: int) -> None:
        """Generate peak positions, weights, and conditioning matrices."""
        rng = np.random.default_rng(seed)

        # Peak weights: w_0 = 10, w_i = 1.1 + 8*(i-1)/(n_peaks-2) for i>=1
        self._weights = np.empty(n_peaks)
        self._weights[0] = 10.0
        if n_peaks > 2:
            self._weights[1:] = 1.1 + 8.0 * np.arange(n_peaks - 1) / (n_peaks - 2)
        elif n_peaks == 2:
            self._weights[1] = 1.1

        # Peak positions: y_0 in [-4,4]^D, y_i in [-5,5]^D
        self._positions = np.empty((n_peaks, dimension))
        self._positions[0] = rng.uniform(-4.0, 4.0, size=dimension)
        if n_peaks > 1:
            self._positions[1:] = rng.uniform(-5.0, 5.0, size=(n_peaks - 1, dimension))

        # Conditioning matrices C_i = Lambda^{alpha_i} / alpha_i^{1/4}
        # alpha values: alpha_0 = 1000^0.5 (conditioning ~31.6 for global peak)
        # alpha_i from {1000^(2j/(n_peaks-2))} for j=0..n_peaks-2, randomly assigned
        if n_peaks > 2:
            alpha_exponents = 2.0 * np.arange(n_peaks - 1) / (n_peaks - 2)
        elif n_peaks == 2:
            alpha_exponents = np.array([0.0])
        else:
            alpha_exponents = np.array([])

        alphas = np.empty(n_peaks)
        alphas[0] = 1000.0**0.5  # ~31.6 for global peak
        if n_peaks > 1:
            remaining_alphas = 1000.0**alpha_exponents
            rng.shuffle(remaining_alphas)
            alphas[1:] = remaining_alphas

        # Build diagonal conditioning vectors (with random permutation per peak)
        # C_i diagonal = Lambda^{alpha_i}_{jj} / alpha_i^{1/4}
        self._c_diag = np.empty((n_peaks, dimension))
        for i in range(n_peaks):
            if dimension == 1:
                diag = np.array([1.0])
            else:
                diag = alphas[i] ** (np.arange(dimension) / (dimension - 1))
            diag /= alphas[i] ** 0.25
            perm = rng.permutation(dimension)
            self._c_diag[i] = diag[perm]

    def evaluate(self, x: NDArray) -> float:
        """Compute the Gallagher Peaks function value."""
        x = self._validate_input(x)
        D = self.dimension

        # Compute max over peaks: w_i * exp(-1/(2D) * sum(c_ij * (x_j - y_ij)^2))
        max_val = 0.0
        for i in range(self._n_peaks):
            diff = x - self._positions[i]
            exponent = -1.0 / (2.0 * D) * np.sum(self._c_diag[i] * diff**2)
            val = self._weights[i] * np.exp(exponent)
            if val > max_val:
                max_val = val

        return float(10.0 - max_val)

    def evaluate_batch(self, X: NDArray) -> NDArray:
        """Compute Gallagher Peaks function for batch."""
        X = self._validate_batch_input(X)
        n = X.shape[0]
        D = self.dimension

        # Vectorized: for each peak, compute all batch points at once
        results = np.full(n, 10.0)
        max_vals = np.zeros(n)

        for i in range(self._n_peaks):
            diff = X - self._positions[i]  # (n, D)
            exponents = -1.0 / (2.0 * D) * np.sum(self._c_diag[i] * diff**2, axis=1)
            vals = self._weights[i] * np.exp(exponents)
            np.maximum(max_vals, vals, out=max_vals)

        return results - max_vals

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum point and value.

        The global minimum is at y_0 (the first peak), where the function
        value is 10 - w_0 = 10 - 10 = 0.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return self._positions[0].copy(), 0.0
