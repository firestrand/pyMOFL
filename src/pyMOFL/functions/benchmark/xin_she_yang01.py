"""
Xin-She Yang's First function (XinSheYang01) implementation.

This function uses random coefficients multiplied by power terms, creating
a stochastic multimodal landscape.

References
----------
.. [1] Yang, X.S. (2010). "Nature-Inspired Metaheuristic Algorithms."
       Second Edition, Luniver Press.
.. [2] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
       optimization problems". International Journal of Mathematical Modelling and Numerical
       Optimisation, 4(2), 150-194.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("XinSheYang01")
@register("xin_she_yang01")
class XinSheYang01Function(OptimizationFunction):
    """
    Xin-She Yang's First function.

    f(x) = sum(epsilon_i * |x_i|^i)

    where epsilon_i are random numbers in [0, 1] drawn from a seeded RNG
    (seed=42) for reproducibility, and i is 1-indexed.

    Properties: Continuous, Non-Differentiable, Separable, Scalable, Multimodal
    Domain: [-5, 5]^D
    Global minimum: f(0, ..., 0) = 0
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        seed: int = 42,
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
        rng = np.random.default_rng(seed)
        self._epsilon = rng.random(dimension)
        # Exponents are 1-indexed: i = 1, 2, ..., D
        self._exponents = np.arange(1, dimension + 1, dtype=float)

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the Xin-She Yang 01 function value."""
        x = self._validate_input(x)
        return float(np.sum(self._epsilon * np.abs(x) ** self._exponents))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Xin-She Yang 01 function for a batch of points."""
        X = self._validate_batch_input(X)
        return np.sum(self._epsilon * np.abs(X) ** self._exponents, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Xin-She Yang 01 function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.zeros(self.dimension), 0.0
