"""
OptimizationFunction base class for all optimization functions in pyMOFL.
Handles bounds, quantization, and constraint logic.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .bounds import Bounds


class OptimizationFunction(ABC):
    """
    Abstract base class for optimization functions.
    Handles bounds enforcement, quantization, constraints, and input validation.

    Subclasses should use _validate_input(x) and _validate_batch_input(X) to check input shape/type.
    """

    initialization_bounds: Bounds
    operational_bounds: Bounds
    constraint_penalty: float = 1e8

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
    ):
        self.dimension = dimension
        base_low = np.full(dimension, -np.inf, dtype=np.float64)
        base_high = np.full(dimension, np.inf, dtype=np.float64)
        self.initialization_bounds = initialization_bounds or Bounds(low=base_low, high=base_high)
        self.operational_bounds = operational_bounds or Bounds(low=base_low, high=base_high)

    def __call__(self, x: NDArray[Any]) -> float:
        """
        Evaluate the function at x. Bounds are metadata only; no enforcement is performed.
        Returns np.nan if constraints are violated (as determined by evaluate or violations).
        """
        x = self._validate_input(x)
        result = self.evaluate(x)
        if np.any(np.isnan(result)):
            return np.nan
        return result

    @abstractmethod
    def evaluate(self, x: NDArray[Any]) -> float:
        """
        Evaluate the function at a point x.
        """
        ...

    def evaluate_batch(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Evaluate the function at multiple points.

        Default implementation loops over rows. Subclasses should override
        with a vectorized implementation for performance.
        """
        X = self._validate_batch_input(X)
        return np.array([self.evaluate(row) for row in X])

    def violations(self, x: NDArray[Any]) -> float:
        """
        Returns the total constraint violation magnitude for x.
        Override in subclasses for custom constraints.
        """
        return 0.0

    def _validate_input(self, x: NDArray[Any]) -> NDArray[Any]:
        """
        Validate that x is a 1D array of length self.dimension.
        Raises ValueError if not.
        """
        x = np.asarray(x)
        if x.shape != (self.dimension,):
            raise ValueError(f"Input must be of shape ({self.dimension},), got {x.shape}")
        return x

    def _validate_batch_input(self, X: NDArray[Any]) -> NDArray[Any]:
        """
        Validate that X is a 2D array with shape (n, self.dimension).
        Raises ValueError if not.
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.dimension:
            raise ValueError(f"Each input must have dimension {self.dimension}, got {X.shape}")
        return X
