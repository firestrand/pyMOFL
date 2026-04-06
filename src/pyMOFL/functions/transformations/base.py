"""
Base classes for transformation functions.

Transformations are pure functions that transform inputs without metadata.
They can be vector-to-vector or scalar-to-scalar.
"""

from abc import ABC, abstractmethod

import numpy as np


class VectorTransform(ABC):
    """
    Base class for vector-to-vector transformations.

    These transform input vectors before they're passed to optimization functions.
    Examples: shift, rotate, scale
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Transform a vector."""
        pass

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """Transform a batch of vectors."""
        # Default implementation - can be overridden for efficiency
        return np.array([self(x) for x in X])


class PenaltyTransform(ABC):
    """
    Base class for vector-to-scalar penalty transformations.

    These take an input vector and return a scalar penalty value
    to be added to the function output. Used for boundary penalties
    in BBOB-style benchmarks where the penalty depends on the raw input
    vector, not the scalar function output.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        """Compute penalty for a single vector."""
        pass

    def compute_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute penalties for a batch of vectors."""
        # Default implementation - can be overridden for efficiency
        return np.array([self(x) for x in X])


class ScalarTransform(ABC):
    """
    Base class for scalar-to-scalar transformations.

    These transform the output of optimization functions.
    Examples: bias, noise
    """

    @abstractmethod
    def __call__(self, y: float) -> float:
        """Transform a scalar."""
        pass

    def transform_batch(self, Y: np.ndarray) -> np.ndarray:
        """Transform a batch of scalars."""
        # Default implementation - can be overridden for efficiency
        return np.array([self(y) for y in Y])
