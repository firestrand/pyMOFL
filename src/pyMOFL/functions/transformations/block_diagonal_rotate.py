"""BlockDiagonalRotateTransform — per-block rotation, O(D*s) complexity."""

from __future__ import annotations

import numpy as np

from .base import VectorTransform


class BlockDiagonalRotateTransform(VectorTransform):
    """Block-diagonal rotation transform.

    Instead of a full D×D rotation matrix, applies a sequence of
    smaller block rotations. This reduces O(D²) to O(D*s) where
    s is the block size.

    Parameters
    ----------
    blocks : list[np.ndarray]
        List of orthogonal rotation matrices. Their sizes must sum
        to the total dimension.
    """

    def __init__(self, blocks: list[np.ndarray]):
        if not blocks:
            raise ValueError("At least one block matrix is required.")

        validated_blocks: list[np.ndarray] = []
        for block in blocks:
            mat = np.asarray(block, dtype=float)
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError("All block matrices must be square (ndim=2, shape[0]==shape[1]).")
            validated_blocks.append(mat)

        self.blocks = validated_blocks
        self._sizes = [b.shape[0] for b in blocks]
        self._dim = sum(self._sizes)
        # Pre-compute cumulative offsets
        self._offsets = [0]
        for s in self._sizes:
            self._offsets.append(self._offsets[-1] + s)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        result = np.empty_like(x)
        for i, block in enumerate(self.blocks):
            start = self._offsets[i]
            end = self._offsets[i + 1]
            result[start:end] = block @ x[start:end]
        return result

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        result = np.empty_like(X)
        for i, block in enumerate(self.blocks):
            start = self._offsets[i]
            end = self._offsets[i + 1]
            # (N, s) @ (s, s).T = (N, s) ... but we need (s, s) @ (s,)
            result[:, start:end] = (block @ X[:, start:end].T).T
        return result

    def to_full_matrix(self) -> np.ndarray:
        """Construct the full D×D block-diagonal matrix."""
        full = np.zeros((self._dim, self._dim))
        for i, block in enumerate(self.blocks):
            start = self._offsets[i]
            end = self._offsets[i + 1]
            full[start:end, start:end] = block
        return full

    def __repr__(self) -> str:
        sizes = "x".join(str(s) for s in self._sizes)
        return f"BlockDiagonalRotateTransform(blocks={sizes}, dim={self._dim})"
