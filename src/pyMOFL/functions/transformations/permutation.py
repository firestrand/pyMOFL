"""PermutationTransform — reorder input vector elements."""

from __future__ import annotations

import numpy as np

from .base import VectorTransform


class PermutationTransform(VectorTransform):
    """Reorder input vector elements by a fixed permutation.

    Applies: result[i] = x[perm[i]]

    Parameters
    ----------
    permutation : np.ndarray
        Integer array where perm[i] is the source index for position i.
        Must be a valid permutation of 0..len-1.
    """

    def __init__(self, permutation: np.ndarray):
        perm = np.asarray(permutation, dtype=int)
        if perm.ndim != 1:
            raise ValueError("Permutation must be a one-dimensional integer array.")
        if perm.size == 0:
            raise ValueError("Permutation must contain at least one element.")
        if not np.array_equal(np.sort(perm), np.arange(len(perm))):
            raise ValueError(f"Must be a valid permutation of 0..{len(perm) - 1}.")
        self.permutation = perm

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x[self.permutation]

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.permutation]

    def __repr__(self) -> str:
        return f"PermutationTransform(dim={len(self.permutation)})"
