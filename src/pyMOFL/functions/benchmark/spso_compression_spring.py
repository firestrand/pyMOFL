"""Explicit native-coordinate SPSO spring variants; legacy Spring is unchanged."""

from typing import Literal, override

import numpy as np
from numpy.typing import NDArray

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("spso_compression_spring")
class SPSOCompressionSpringFunction(OptimizationFunction):
    """SPSO spring weight and multiplicative penalties in [N,D,d] coordinates.

    Parameters
    ----------
    penalty_version
        "2011" uses corrected free-length constraint g2. "2007" preserves the
        pinned distribution's documented g2-positive multiplier using g1.
    initialization_bounds, operational_bounds
        Metadata only; defaults [1,.6,.207] to [70,3,.5]. No quantization or
        clipping occurs in this base function. Compose the half-up transform.

    Notes
    -----
    Raw weight is pi**2/4 * D*d**2*(N+2). Constraints use source material
    constants; each positive constraint adds a multiplicative cubic factor.
    The best-known source target 2.6254214578 is suite metadata, not a certified
    optimum point. get_global_minimum therefore retains NotImplementedError.
    See docs/spso-reference-review.md for source/version/domain limits.
    """

    def __init__(
        self,
        penalty_version: Literal["2007", "2011"] = "2011",
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
    ) -> None:
        if penalty_version not in ("2007", "2011"):
            raise ValueError("penalty_version must be '2007' or '2011'")
        low, high = np.array([1.0, 0.6, 0.207]), np.array([70.0, 3.0, 0.5])
        qtype = np.array(
            [
                QuantizationTypeEnum.INTEGER,
                QuantizationTypeEnum.CONTINUOUS,
                QuantizationTypeEnum.STEP,
            ]
        )
        super().__init__(
            3,
            initialization_bounds
            or Bounds(low.copy(), high.copy(), BoundModeEnum.INITIALIZATION, qtype.copy(), 0.001),
            operational_bounds
            or Bounds(low.copy(), high.copy(), BoundModeEnum.OPERATIONAL, qtype.copy(), 0.001),
        )
        self.penalty_version = penalty_version

    @override
    def evaluate(self, x: NDArray[np.float64]) -> float:
        """Evaluate one native [N,D,d] vector of shape (3,)."""
        x = self._validate_input(x)
        return float(self.evaluate_batch(x[None, :])[0])

    @override
    def evaluate_batch(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate rows, preserving the explicitly selected source penalty."""
        X = self._validate_batch_input(X)
        N, D, d = X[:, 0], X[:, 1], X[:, 2]
        cf = 1 + 0.75 * d / (D - d) + 0.615 * d / D
        stiffness = 0.125 * 11500000 * d**4 / (N * D**3)
        g1 = 8 * cf * 1000 * D / (np.pi * d**3) - 189000
        g2 = 1000 / stiffness + 1.05 * (N + 2) * d - 14
        g3 = 300 / stiffness - 6
        g4 = 1.25 - (1000 - 300) / stiffness
        result = np.pi * np.pi * D * d * d * (N + 2) * 0.25
        # Preserve the pinned 2007 bug explicitly; never change generic Spring.
        g2_multiplier = g1 if self.penalty_version == "2007" else g2
        for condition, multiplier in ((g1, g1), (g2, g2_multiplier), (g3, g3), (g4, 1e10 * g4)):
            factor = np.where(condition > 0, 1 + multiplier, 1.0)
            result = result * factor * factor * factor
        return result
