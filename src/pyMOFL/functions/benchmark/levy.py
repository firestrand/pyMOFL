"""
Levy function implementation.

The Levy function uses a transformation w_i = 1 + (x_i - 1)/4 with sinusoidal
terms, creating a highly multimodal landscape with global minimum at (1,...,1).

References
----------
.. [1] Levy, A. & Montalvo, A. (1985). "The tunneling algorithm for the global
       minimization of functions." SIAM J. Sci. Stat. Comput., 6(1).
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Levy")
@register("levy")
class LevyFunction(OptimizationFunction):
    """
    Levy function.

    wᵢ = 1 + (xᵢ - 1)/4
    f(x) = sin²(πw₁) + Σᵢ₌₁ᴰ⁻¹ (wᵢ-1)²(1+10sin²(πwᵢ₊₁)) + (wD-1)²(1+sin²(2πwD))

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-10, 10]^D
    Global minimum: f(1, ..., 1) = 0
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -10.0),
            high=np.full(dimension, 10.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Levy function."""
        x = self._validate_input(x)
        w = 1.0 + (x - 1.0) / 4.0
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:]) ** 2))
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        return float(term1 + term2 + term3)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Levy for a batch of points."""
        X = self._validate_batch_input(X)
        W = 1.0 + (X - 1.0) / 4.0
        term1 = np.sin(np.pi * W[:, 0]) ** 2
        term2 = np.sum((W[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * W[:, 1:]) ** 2), axis=1)
        term3 = (W[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * W[:, -1]) ** 2)
        return term1 + term2 + term3

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Levy function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.ones(self.dimension), 0.0


@register("LevyCEC")
@register("levy_cec")
class LevyCECFunction(OptimizationFunction):
    """
    Levy function (CEC 2017 variant).

    Uses sin(πwᵢ + 1) in the summation term instead of sin(πwᵢ₊₁).
    Keeps the standard w computation: wᵢ = 1 + (xᵢ - 1)/4.
    Used in CEC 2013/2017 C reference implementations.

    f(x) = sin²(πw₁) + Σᵢ₌₁ᴰ⁻¹ (wᵢ-1)²(1+10sin²(πwᵢ+1)) + (wD-1)²(1+sin²(2πwD))

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-10, 10]^D
    Global minimum: f(1, ..., 1) = 0
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -10.0),
            high=np.full(dimension, 10.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Levy CEC variant."""
        x = self._validate_input(x)
        w = 1.0 + (x - 1.0) / 4.0
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        return float(term1 + term2 + term3)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Levy CEC variant for a batch of points."""
        X = self._validate_batch_input(X)
        W = 1.0 + (X - 1.0) / 4.0
        term1 = np.sin(np.pi * W[:, 0]) ** 2
        term2 = np.sum(
            (W[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * W[:, :-1] + 1) ** 2),
            axis=1,
        )
        term3 = (W[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * W[:, -1]) ** 2)
        return term1 + term2 + term3

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Levy CEC function."""
        return np.ones(self.dimension), 0.0


@register("LevyCEC2022")
@register("levy_cec2022")
class LevyCEC2022Function(OptimizationFunction):
    """
    Levy function (CEC 2022+ variant).

    Two differences from standard Levy:
    1. Uses wᵢ = 1 + xᵢ/4 (optimum at origin) instead of wᵢ = 1 + (xᵢ-1)/4
    2. Uses sin(πwᵢ + 1) in the summation instead of sin(πwᵢ₊₁)

    Used in CEC 2022 C reference implementations.

    f(x) = sin²(πw₁) + Σᵢ₌₁ᴰ⁻¹ (wᵢ-1)²(1+10sin²(πwᵢ+1)) + (wD-1)²(1+sin²(2πwD))

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-10, 10]^D
    Global minimum: f(0, ..., 0) = 0
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -10.0),
            high=np.full(dimension, 10.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Levy CEC 2022 variant."""
        x = self._validate_input(x)
        w = 1.0 + x / 4.0
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        return float(term1 + term2 + term3)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Levy CEC 2022 variant for a batch of points."""
        X = self._validate_batch_input(X)
        W = 1.0 + X / 4.0
        term1 = np.sin(np.pi * W[:, 0]) ** 2
        term2 = np.sum(
            (W[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * W[:, :-1] + 1) ** 2),
            axis=1,
        )
        term3 = (W[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * W[:, -1]) ** 2)
        return term1 + term2 + term3

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Levy CEC 2022 function."""
        return np.zeros(self.dimension), 0.0
