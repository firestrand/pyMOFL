"""
Step Ellipsoidal function (BBOB f7 base).

f(z) = 0.1 * max(|z_hat_1|/10000, sum_i(10^(2i/(D-1)) * z_tilde_i^2))

where:
    z_hat_i = Lambda_i * z_i  with Lambda_i = 10^(i/(2(D-1)))
    z_tilde_i = floor(0.5 + z_hat_i)  if |z_hat_i| > 0.5
    z_tilde_i = floor(0.5 + 10*z_hat_i) / 10  if |z_hat_i| <= 0.5

The floor/step operation creates plateaus in the landscape.
Two external rotation matrices (Q, R) are applied via transforms in the config chain.

References
----------
.. [1] Hansen, N., et al. (2009). "Real-parameter black-box optimization
       benchmarking 2009: Noiseless functions definitions."
       INRIA Technical Report RR-6829.
"""

import numpy as np

from pyMOFL.core.bounds import BoundModeEnum, Bounds, QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.registry import register


@register("StepEllipsoid")
@register("step_ellipsoid")
class StepEllipsoidFunction(OptimizationFunction):
    """
    Step Ellipsoidal function.

    Combines ill-conditioning with a step/plateau structure.
    The conditional floor operation creates flat regions.

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
        # Lambda conditioning: 10^(i/(2(D-1)))
        if dimension == 1:
            self._lambda = np.array([1.0])
            self._cond = np.array([1.0])
        else:
            self._lambda = np.power(10.0, np.arange(dimension) / (2.0 * (dimension - 1)))
            # Condition numbers for the sum: 10^(2i/(D-1))
            self._cond = np.power(10.0, 2.0 * np.arange(dimension) / (dimension - 1))

    def _apply_step(self, z_hat: np.ndarray) -> np.ndarray:
        """Apply the conditional floor/step operation."""
        z_tilde = np.where(
            np.abs(z_hat) > 0.5,
            np.floor(0.5 + z_hat),
            np.floor(0.5 + 10.0 * z_hat) / 10.0,
        )
        return z_tilde

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the Step Ellipsoidal function value."""
        x = self._validate_input(x)
        z_hat = self._lambda * x
        z_tilde = self._apply_step(z_hat)
        sum_term = float(np.sum(self._cond * z_tilde**2))
        first_term = np.abs(z_hat[0]) / 1e4 if self.dimension > 0 else 0.0
        return 0.1 * max(first_term, sum_term)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Step Ellipsoidal function for batch."""
        X = self._validate_batch_input(X)
        Z_hat = X * self._lambda  # (N, D)
        Z_tilde = self._apply_step(Z_hat)
        sum_terms = np.sum(self._cond * Z_tilde**2, axis=1)
        first_terms = np.abs(Z_hat[:, 0]) / 1e4
        return 0.1 * np.maximum(first_terms, sum_terms)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.zeros(self.dimension), 0.0
