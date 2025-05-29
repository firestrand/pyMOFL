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
    operational_bounds: Bounds | None
    constraint_penalty: float = 1e8

    def __call__(self, x: NDArray[Any]) -> float:
        """
        Evaluate the function at x, enforcing operational bounds and constraints.
        Returns np.nan if constraints are violated.
        """
        x_proj = self._enforce(x)
        if np.any(np.isnan(x_proj)):
            return np.nan
        return self.evaluate(x_proj)

    @abstractmethod
    def evaluate(self, z: NDArray[Any]) -> float:
        """
        Evaluate the function at a (possibly repaired) point z.
        """
        pass

    def _enforce(self, x: NDArray[Any]) -> NDArray[Any]:
        """
        Enforce operational bounds and constraints on x.
        Returns np.nan if constraints are violated.
        """
        if self.operational_bounds is not None:
            x = self.operational_bounds.project(x)
        if self.violations(x) > 0:
            return np.full_like(x, np.nan)
        return x

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