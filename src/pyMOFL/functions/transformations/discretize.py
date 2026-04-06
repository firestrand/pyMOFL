"""DiscretizeTransform — mixed-integer variable discretization (COCO bbob-mixint).

80% of variables are discretized with arities [2, 4, 8, 16] per group.
20% remain continuous. Dimension must be divisible by 5.
"""

from __future__ import annotations

import numpy as np

from .base import VectorTransform

# Group arities in COCO order
_GROUP_ARITIES = [2, 4, 8, 16, 0]  # 0 = continuous


class DiscretizeTransform(VectorTransform):
    """Per-variable discretization for bbob-mixint.

    Variables are split into 5 groups of D/5. Each group gets one arity
    from [2, 4, 8, 16, continuous]. Discrete variables are mapped from
    outer integer domain {0, ..., n-1} to inner continuous domain via
    affine mapping within [-4, 4].

    Parameters
    ----------
    dimension : int
        Must be divisible by 5.
    xopt : np.ndarray, optional
        Optimum in outer space. If provided, offset is computed to align
        xopt to nearest grid point.
    """

    def __init__(self, dimension: int, xopt: np.ndarray | None = None):
        if dimension % 5 != 0:
            raise ValueError(f"Dimension must be divisible by 5 for mixint, got {dimension}.")

        self.dimension = dimension
        group_size = dimension // 5

        # Assign arities per variable
        self.arities: list[int] = []
        for arity in _GROUP_ARITIES:
            self.arities.extend([arity] * group_size)

        # Pre-compute inner bounds per variable
        self._inner_l = np.zeros(dimension)
        self._inner_u = np.zeros(dimension)
        self._is_discrete = np.zeros(dimension, dtype=bool)

        for i, arity in enumerate(self.arities):
            if arity > 0:
                self._is_discrete[i] = True
                self._inner_l[i] = -4.0 + 8.0 / (arity + 1)
                self._inner_u[i] = 4.0 - 8.0 / (arity + 1)

        # Compute offset if xopt provided
        self._offset = np.zeros(dimension)
        if xopt is not None:
            for i, arity in enumerate(self.arities):
                if arity > 0:
                    # Snap xopt[i] to nearest grid point
                    rounded = np.clip(np.round(xopt[i]), 0, arity - 1)
                    self._offset[i] = xopt[i] - rounded

    def __call__(self, x: np.ndarray) -> np.ndarray:
        result = np.copy(x)

        for i, arity in enumerate(self.arities):
            if arity == 0:
                continue  # continuous passthrough
            # Apply offset, round, clamp
            adjusted = x[i] - self._offset[i]
            rounded = int(np.clip(np.round(adjusted), 0, arity - 1))
            # Affine map: outer integer -> inner continuous
            if arity == 1:
                result[i] = self._inner_l[i]
            else:
                result[i] = self._inner_l[i] + (self._inner_u[i] - self._inner_l[i]) * rounded / (
                    arity - 1
                )

        return result

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        result = np.copy(X)

        for i, arity in enumerate(self.arities):
            if arity == 0:
                continue
            adjusted = X[:, i] - self._offset[i]
            rounded = np.clip(np.round(adjusted), 0, arity - 1).astype(int)
            if arity == 1:
                result[:, i] = self._inner_l[i]
            else:
                result[:, i] = self._inner_l[i] + (
                    self._inner_u[i] - self._inner_l[i]
                ) * rounded / (arity - 1)

        return result

    def __repr__(self) -> str:
        return f"DiscretizeTransform(dimension={self.dimension})"
