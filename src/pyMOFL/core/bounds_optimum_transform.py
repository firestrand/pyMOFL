"""
Transform that constructs optimum points using bounds and shift patterns.

Used for benchmark functions where the optimum is placed at specific patterns
involving bounds and shift vector values.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

_OPTIMUM_ALIAS: dict[str, str] = {
    "bounds_shift": "bounds_shift",
    "cec2005_f05": "bounds_shift",
    "alternate_bounds": "alternate_bounds",
    "alternate": "alternate_bounds",
    "alternate_shift": "alternate_bounds",
    "alternate_bounds_opt": "alternate_bounds",
    "cec2005_f08": "alternate_bounds",
    "composition_bounds": "composition_bounds",
    "cec2005_f20": "composition_bounds",
    "alternate_odds": "alternate_odds",
    "f20_alternate_odds": "alternate_odds",
    "non_continuous": "non_continuous",
    "cec2005_f23": "non_continuous",
}


def normalize_optimum_pattern(pattern: Any) -> str:
    """Return canonical pattern key for suite configuration aliases."""

    pattern_type = str(pattern).lower()
    return _OPTIMUM_ALIAS.get(pattern_type, pattern_type)


class OptimumPattern(ABC):
    """Base class for optimum construction patterns."""

    @abstractmethod
    def construct_optimum(
        self, dimension: int, shift_vector: np.ndarray, lower_bound: float, upper_bound: float
    ) -> np.ndarray:
        """Construct the optimum point based on the pattern."""
        pass


class BoundsShiftOptimumPattern(OptimumPattern):
    """
    Bounds-shift optimum pattern.

    Creates optimum points using a combination of bounds and shift vector values.
    The pattern divides dimensions into three sections: lower bounds, shift values,
    and upper bounds.

    Pattern by dimension:
    - D=2: [upper, upper]
    - D=10: [lower]*3 + shift[3:6] + [upper]*4
    - D=30: [lower]*8 + shift[8:21] + [upper]*9
    - D=50: [lower]*13 + shift[13:36] + [upper]*14
    - Other D: approximation using thirds
    """

    def construct_optimum(
        self,
        dimension: int,
        shift_vector: np.ndarray,
        lower_bound: float = -100.0,
        upper_bound: float = 100.0,
    ) -> np.ndarray:
        """Construct optimum using bounds-shift pattern."""
        optimum = np.zeros(dimension)

        if dimension == 2:
            optimum[:] = upper_bound
        elif dimension == 10:
            optimum[:3] = lower_bound
            optimum[3:6] = shift_vector[3:6]
            optimum[6:] = upper_bound
        elif dimension == 30:
            optimum[:8] = lower_bound
            optimum[8:21] = shift_vector[8:21]
            optimum[21:] = upper_bound
        elif dimension == 50:
            optimum[:13] = lower_bound
            optimum[13:36] = shift_vector[13:36]
            optimum[36:] = upper_bound
        else:
            # General pattern for other dimensions
            third = dimension // 3
            optimum[:third] = lower_bound
            if 2 * third <= len(shift_vector):
                optimum[third : 2 * third] = shift_vector[third : 2 * third]
            else:
                # If shift vector is shorter, use what we have
                available = min(third, len(shift_vector) - third)
                optimum[third : third + available] = shift_vector[third : third + available]
                optimum[third + available : 2 * third] = upper_bound
            optimum[2 * third :] = upper_bound

        return optimum


class AlternateShiftOptimumPattern(OptimumPattern):
    """
    Alternate shift optimum pattern.

    Creates optimum points where even indices are set to a fixed value
    and odd indices use shift vector values.

    Pattern:
    - Even indices (0, 2, 4, ...): fixed_value (default -32.0)
    - Odd indices (1, 3, 5, ...): shift_vector values
    """

    def construct_optimum(
        self,
        dimension: int,
        shift_vector: np.ndarray,
        lower_bound: float = -32.0,
        upper_bound: float = 100.0,
    ) -> np.ndarray:
        """Construct optimum using alternate pattern."""
        optimum = shift_vector.copy()  # Start with shift vector

        # Set even indices to fixed value
        for i in range(0, dimension, 2):
            optimum[i] = lower_bound

        return optimum


class CompositionBoundsOptimumPattern(OptimumPattern):
    """
    Composition bounds optimum pattern.

    Creates optimum points where even indices use shift vector values
    and odd indices are set to the upper bound.

    Pattern:
    - Even indices (0, 2, 4, ...): shift_vector values
    - Odd indices (1, 3, 5, ...): upper_bound (default 5.0)
    """

    def construct_optimum(
        self,
        dimension: int,
        shift_vector: np.ndarray,
        lower_bound: float = -5.0,
        upper_bound: float = 5.0,
    ) -> np.ndarray:
        """Construct optimum using composition bounds pattern."""
        optimum = shift_vector.copy()  # Start with shift vector

        # Set odd indices to upper bound
        for i in range(1, dimension, 2):
            optimum[i] = upper_bound

        return optimum


class AlternateOddsOptimumPattern(OptimumPattern):
    """Alternate odds optimum pattern.

    Sets odd indices to a fixed value while keeping even indices from shift data.
    """

    def construct_optimum(
        self,
        dimension: int,
        shift_vector: np.ndarray,
        lower_bound: float = 5.0,
        upper_bound: float = 100.0,
    ) -> np.ndarray:
        optimum = shift_vector.copy()
        for i in range(1, dimension, 2):
            optimum[i] = lower_bound
        return optimum


class NonContinuousOptimumPattern(OptimumPattern):
    """
    Non-continuous optimum pattern.

    Transforms component optima by applying non-continuous rounding.
    This ensures that weight calculation uses the transformed optima that
    match the transformed input space.
    """

    def construct_optimum(
        self,
        dimension: int,
        shift_vector: np.ndarray,
        lower_bound: float = -100.0,
        upper_bound: float = 100.0,
    ) -> np.ndarray:
        """Construct optimum by applying non-continuous transform to shift vector."""
        y = shift_vector.copy()
        mask = np.abs(shift_vector) >= 0.5
        y[mask] = np.round(2 * shift_vector[mask]) / 2
        return y


class BoundsOptimumTransform:
    """
    Transform that helps construct optimum points for functions where
    the optimum is placed at bounds or follows specific patterns.

    This is used internally by functions like Schwefel 2.6 to compute
    their B vectors based on the constructed optimum.
    """

    def __init__(self, pattern: OptimumPattern):
        self.pattern = pattern

    def get_optimum(
        self,
        dimension: int,
        shift_vector: np.ndarray,
        lower_bound: float = -100.0,
        upper_bound: float = 100.0,
    ) -> np.ndarray:
        """Get the constructed optimum point."""
        return self.pattern.construct_optimum(dimension, shift_vector, lower_bound, upper_bound)

    @staticmethod
    def create_from_config(config: dict[str, Any]) -> "BoundsOptimumTransform":
        """Create transform from configuration.

        Supported pattern names:
        - ``"bounds_shift"`` (alias: ``"cec2005_f05"``)
        - ``"alternate_bounds"`` (alias: ``"alternate_shift"``, ``"alternate_bounds_opt"``, ``"cec2005_f08"``)
        - ``"composition_bounds"`` (alias: ``"cec2005_f20"``)
        - ``"alternate_odds"`` (alias: ``"f20_alternate_odds"``)
        - ``"non_continuous"`` (alias: ``"cec2005_f23"``)
        """
        pattern_type = normalize_optimum_pattern(config.get("pattern", "bounds_shift"))

        if pattern_type == "bounds_shift":
            pattern = BoundsShiftOptimumPattern()
        elif pattern_type == "alternate_bounds":
            pattern = AlternateShiftOptimumPattern()
        elif pattern_type == "composition_bounds":
            pattern = CompositionBoundsOptimumPattern()
        elif pattern_type == "alternate_odds":
            pattern = AlternateOddsOptimumPattern()
        elif pattern_type == "non_continuous":
            pattern = NonContinuousOptimumPattern()
        else:
            raise ValueError(f"Unknown optimum pattern: {pattern_type}")

        return BoundsOptimumTransform(pattern)
