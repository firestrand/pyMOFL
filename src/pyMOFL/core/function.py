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
    Handles bounds enforcement, quantization, and constraints.
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