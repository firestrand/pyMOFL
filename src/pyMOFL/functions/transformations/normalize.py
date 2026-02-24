"""
Composition normalize transformation.

Implements the normalization used in weighted composition functions.
This is an OUTPUT transformation that scales function values, not an input transformation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import ScalarTransform


class NormalizeTransform(ScalarTransform):
    """
    Composition normalize transformation.

    Scales function output by C/f_max factor where:
    - C is a fixed constant (default 2000.0)
    - f_max is computed by evaluating the function at reference point [5.0, 5.0, ...]

    Used in weighted composition functions to normalize component outputs
    so they contribute equally to the composition.
    """

    def __init__(
        self,
        C: float = 2000.0,
        f_max: float | str | None = None,
        reference_point: float = 5.0,
        component_function: Any = None,
    ):
        """
        Initialize normalize transformation.

        Args:
            C: Normalization constant (default 2000.0)
            f_max: Maximum value for normalization. If "computed" or None,
                   will be computed by evaluating component_function at reference_point
            reference_point: Point value used for computing f_max (default 5.0)
            component_function: Function to evaluate for computing f_max
        """
        self.C = C
        self.reference_point = reference_point
        self.component_function = component_function

        if f_max == "computed" or f_max is None:
            # Will compute f_max lazily when first needed
            self._f_max = None
            self._f_max_computed = False
        else:
            self._f_max = float(f_max)
            self._f_max_computed = True

    def _compute_f_max(self, dimension: int):
        """Compute f_max by evaluating component function at reference point."""
        if self._f_max_computed:
            return self._f_max

        if self.component_function is None:
            raise ValueError("Cannot compute f_max: no component function provided")

        # Create reference point vector [5.0, 5.0, ..., 5.0]
        ref_point = np.full(dimension, self.reference_point)

        # Evaluate function at reference point to get f_max
        self._f_max = abs(self.component_function.evaluate(ref_point))
        self._f_max_computed = True

        return self._f_max

    def set_component_function(self, component_function: Any) -> None:
        """Set the component function for computing f_max."""
        self.component_function = component_function
        self._f_max_computed = False  # Reset to recompute if needed

    def __call__(self, value: float, dimension: int | None = None) -> float:  # type: ignore[override]
        """
        Apply normalization to function output.

        Args:
            value: Function output value
            dimension: Needed for computing f_max if not already computed

        Returns:
            Normalized value: value * C/f_max
        """
        if not self._f_max_computed:
            if dimension is None and self.component_function is not None:
                dimension = self.component_function.dimension
            if dimension is None:
                raise ValueError("Dimension required for computing f_max")
            self._compute_f_max(dimension)

        # Apply normalization: output * C/f_max
        assert self._f_max is not None
        return float(value * self.C / self._f_max)

    def transform_batch(self, values: np.ndarray, dimension: int | None = None) -> np.ndarray:  # type: ignore[override]
        """
        Apply normalization to batch of values.

        Args:
            values: Batch of function output values
            dimension: Needed for computing f_max if not already computed

        Returns:
            Normalized values
        """
        if not self._f_max_computed:
            if dimension is None:
                raise ValueError("Dimension required for computing f_max")
            self._compute_f_max(dimension)

        assert self._f_max is not None
        return values * self.C / self._f_max
