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

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Lunacek Bi-Rastrigin function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.full(self.dimension, 2.5), 0.0


@register("LunacekBiRastriginCEC")
@register("lunacek_bi_rastrigin_cec")
class LunacekBiRastriginCECFunction(OptimizationFunction):
    """
    Lunacek Bi-Rastrigin function (CEC variant).

    CEC reference implementations center the function at the origin with
    internal 2x scaling. The formula in terms of input x:

    z = 2·x
    f(x) = min(Σz², d·D + s·Σ(z+μ₀-μ₁)²) + 10·(D - Σcos(2πz))

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-100, 100]^D
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
        self._mu0 = 2.5
        self._d = 1.0
        self._s = 1.0 - 1.0 / (2.0 * np.sqrt(dimension + 20.0) - 8.2)
        self._mu1 = -np.sqrt((self._mu0**2 - self._d) / self._s)
        self._mu_diff = self._mu0 - self._mu1

    def evaluate(self, x: np.ndarray) -> float:
        """Compute CEC Lunacek Bi-Rastrigin."""
        x = self._validate_input(x)
        z = 2.0 * x
        sum1 = np.sum(z**2)
        sum2 = self._d * self.dimension + self._s * np.sum((z + self._mu_diff) ** 2)
        rastrigin = 10.0 * (self.dimension - np.sum(np.cos(2.0 * np.pi * z)))
        return float(min(sum1, sum2) + rastrigin)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute CEC Lunacek Bi-Rastrigin for a batch of points."""
        X = self._validate_batch_input(X)
        Z = 2.0 * X
        sum1 = np.sum(Z**2, axis=1)
        sum2 = self._d * self.dimension + self._s * np.sum((Z + self._mu_diff) ** 2, axis=1)
        rastrigin = 10.0 * (self.dimension - np.sum(np.cos(2.0 * np.pi * Z), axis=1))
        return np.minimum(sum1, sum2) + rastrigin

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the CEC Lunacek Bi-Rastrigin function."""
        return np.zeros(self.dimension), 0.0


@register("LunacekRotatedCosine")
@register("lunacek_rotated_cosine")
class LunacekRotatedCosineFunction(OptimizationFunction):
    """
    Lunacek Bi-Rastrigin with rotation applied only to the cosine term.

    CEC C reference implementations apply rotation asymmetrically: the
    quadratic basin terms use unrotated coordinates while the cosine
    modulation uses rotated coordinates. This cannot be expressed as an
    external ComposedFunction transform and must be handled internally.

    z = 2 · sign(shift) · x
    tmpx = z + μ₀
    f(x) = min(Σ(tmpxᵢ-μ₀)², d·D + s·Σ(tmpxᵢ-μ₁)²)
           + 10·(D - Σcos(2π·(R·z)ᵢ))

    When cosine_rotation is None, cosine uses unrotated z (equivalent to
    LunacekBiRastriginCECFunction with signs).

    Parameters
    ----------
    shift_signs : array-like, optional
        Array of +1/-1 values (typically np.sign(shift_vector)). When None,
        defaults to all +1 (no sign flip).
    cosine_rotation : array-like, optional
        Rotation matrix applied only to the cosine term. When None, cosine
        uses unrotated z.

    Domain: [-100, 100]^D
    Global minimum: f(0, ..., 0) = 0
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        shift_signs = kwargs.pop("shift_signs", None)
        cosine_rotation = kwargs.pop("cosine_rotation", None)
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
        self._mu0 = 2.5
        self._d = 1.0
        self._s = 1.0 - 1.0 / (2.0 * np.sqrt(dimension + 20.0) - 8.2)
        self._mu1 = -np.sqrt((self._mu0**2 - self._d) / self._s)
        self._mu_diff = self._mu0 - self._mu1

        if shift_signs is not None:
            self._signs = np.asarray(shift_signs, dtype=np.float64).ravel()
        else:
            self._signs = np.ones(dimension, dtype=np.float64)

        if cosine_rotation is not None:
            self._cosine_rotation = np.asarray(cosine_rotation, dtype=np.float64)
        else:
            self._cosine_rotation = None

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Lunacek Bi-Rastrigin with rotated cosine."""
        x = self._validate_input(x)
        z = 2.0 * self._signs * x
        tmpx = z + self._mu0

        sum1 = np.sum((tmpx - self._mu0) ** 2)
        sum2 = self._d * self.dimension + self._s * np.sum((tmpx - self._mu1) ** 2)

        if self._cosine_rotation is not None:
            z_rot = self._cosine_rotation @ z
        else:
            z_rot = z
        cosine = 10.0 * (self.dimension - np.sum(np.cos(2.0 * np.pi * z_rot)))
        return float(min(sum1, sum2) + cosine)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Lunacek Bi-Rastrigin with rotated cosine for a batch."""
        X = self._validate_batch_input(X)
        Z = 2.0 * self._signs[np.newaxis, :] * X
        tmpx = Z + self._mu0

        sum1 = np.sum((tmpx - self._mu0) ** 2, axis=1)
        sum2 = self._d * self.dimension + self._s * np.sum((tmpx - self._mu1) ** 2, axis=1)

        if self._cosine_rotation is not None:
            Z_rot = (self._cosine_rotation @ Z.T).T
        else:
            Z_rot = Z
        cosine = 10.0 * (self.dimension - np.sum(np.cos(2.0 * np.pi * Z_rot), axis=1))
        return np.minimum(sum1, sum2) + cosine

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Lunacek rotated cosine function."""
        return np.zeros(self.dimension), 0.0
