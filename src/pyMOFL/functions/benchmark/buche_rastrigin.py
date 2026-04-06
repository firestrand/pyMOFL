"""
Büche-Rastrigin function (BBOB f4 base).

A non-separable variant of Rastrigin with per-element asymmetric scaling
that breaks the symmetry of the standard Rastrigin function.

f(x) = sum_i(s_i * (x_i^2 - 10*cos(2*pi*x_i)) + 10)

where s_i depends on whether x_i > 0 and the index i:
  s_i = 10^(0.5 * i/(D-1))  for even i when x_i > 0
  s_i = 10^(0.5 * i/(D-1))  always for odd i
  For even i when x_i <= 0: additional factor of 10

References
----------
.. [1] Hansen, N., et al. (2009). "Real-parameter black-box optimization
       benchmarking 2009: Noiseless functions definitions."
       INRIA Technical Report RR-6829.
"""

import numpy as np
from numpy.typing import NDArray

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("BucheRastrigin")
@register("buche_rastrigin")
class BucheRastriginFunction(OptimizationFunction):
    """
    Büche-Rastrigin function.

    A variant of the Rastrigin function with per-element asymmetric scaling.
    Even-indexed variables are scaled differently for positive vs negative values,
    breaking the symmetry of the standard Rastrigin.

    Global minimum: f(0, ..., 0) = 0
    Domain: [-5, 5]^D
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -5.0),
            high=np.full(dimension, 5.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )
        # Pre-compute per-element exponents: 10^(0.5 * i/(D-1))
        if dimension == 1:
            self._alpha = np.array([1.0])
        else:
            exponents = 0.5 * np.arange(dimension) / (dimension - 1)
            self._alpha = np.power(10.0, exponents)

    def _compute_scaling(self, x: NDArray) -> NDArray:
        """Compute per-element asymmetric scaling factors.

        For even indices (0-based): scale by additional factor of 10 when x_i <= 0.
        """
        s = self._alpha.copy()
        even_mask = np.arange(self.dimension) % 2 == 0
        neg_mask = x <= 0
        s[even_mask & neg_mask] *= 10.0
        return s

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the Büche-Rastrigin function value."""
        x = self._validate_input(x)
        s = self._compute_scaling(x)
        z = s * x
        return float(np.sum(z**2 - 10.0 * np.cos(2.0 * np.pi * z) + 10.0))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Büche-Rastrigin function for batch."""
        X = self._validate_batch_input(X)
        # Vectorized scaling for batch
        S = np.tile(self._alpha, (X.shape[0], 1))
        even_mask = np.arange(self.dimension) % 2 == 0
        neg_mask = X <= 0
        S[:, even_mask] = np.where(
            neg_mask[:, even_mask],
            S[:, even_mask] * 10.0,
            S[:, even_mask],
        )
        Z = S * X
        return np.sum(Z**2 - 10.0 * np.cos(2.0 * np.pi * Z) + 10.0, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.zeros(self.dimension), 0.0
