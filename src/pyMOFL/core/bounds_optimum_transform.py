"""
Transform that constructs optimum points using bounds and shift patterns.

Used for benchmark functions where the optimum is placed at specific patterns
involving bounds and shift vector values.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

_OPTIMUM_ALIAS: dict[str, str] = {
    "alternate_bounds": "alternate_bounds",
    "alternate": "alternate_bounds",
    "composition_bounds": "composition_bounds",
    "alternate_odds": "alternate_odds",
    "non_continuous": "non_continuous",
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
        - ``"alternate_bounds"``
        - ``"composition_bounds"``
        - ``"alternate_odds"``
        - ``"non_continuous"``
        """
        pattern_type = normalize_optimum_pattern(config.get("pattern", "alternate_bounds"))

        if pattern_type == "alternate_bounds":
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
