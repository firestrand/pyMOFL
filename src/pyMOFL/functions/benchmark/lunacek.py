"""
Lunacek Bi-Rastrigin function implementation.

The Lunacek Bi-Rastrigin function creates a two-basin landscape by combining
two spherical basins (centered at μ₀ and μ₁) with Rastrigin-style cosine
modulation. The parameter s is dimension-dependent.

References
----------
.. [1] Lunacek, M. & Whitley, D. (2006). "The dispersion metric and the CMA
       evolution strategy." GECCO.
.. [2] Liang, J.J., et al. (2013). "Problem definitions and evaluation criteria
       for the CEC 2013 special session on real-parameter optimization."
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("LunacekBiRastrigin")
@register("lunacek_bi_rastrigin")
class LunacekBiRastriginFunction(OptimizationFunction):
    """
    Lunacek Bi-Rastrigin function.

    Parameters: μ₀ = 2.5, d = 1.0, s = 1 - 1/(2√(D+20) - 8.2), μ₁ = -√((μ₀²-d)/s)

    f(x) = min(Σ(xᵢ-μ₀)², d·D + s·Σ(xᵢ-μ₁)²) + 10·(D - Σcos(2π(xᵢ-μ₀)))

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-100, 100]^D
    Global minimum: f(μ₀, ..., μ₀) = 0
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -100.0),
            high=np.full(dimension, 100.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )
        # Dimension-dependent parameters
        self._mu0 = 2.5
        self._d = 1.0
        self._s = 1.0 - 1.0 / (2.0 * np.sqrt(dimension + 20.0) - 8.2)
        self._mu1 = -np.sqrt((self._mu0**2 - self._d) / self._s)

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Lunacek Bi-Rastrigin."""
        x = self._validate_input(x)
        sum1 = np.sum((x - self._mu0) ** 2)
        sum2 = self._d * self.dimension + self._s * np.sum((x - self._mu1) ** 2)
        rastrigin = 10.0 * (self.dimension - np.sum(np.cos(2.0 * np.pi * (x - self._mu0))))
        return float(min(sum1, sum2) + rastrigin)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Lunacek Bi-Rastrigin for a batch of points."""
        X = self._validate_batch_input(X)
        sum1 = np.sum((X - self._mu0) ** 2, axis=1)
        sum2 = self._d * self.dimension + self._s * np.sum((X - self._mu1) ** 2, axis=1)
        rastrigin = 10.0 * (self.dimension - np.sum(np.cos(2.0 * np.pi * (X - self._mu0)), axis=1))
        return np.minimum(sum1, sum2) + rastrigin

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """Get the global minimum of the Lunacek Bi-Rastrigin function.

        Parameters
        ----------
        dimension : int
            The dimension of the function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        return np.full(dimension, 2.5), 0.0
