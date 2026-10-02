"""
Decomposition and variable grouping transformations for large-scale optimization.

Supports splitting input vectors into separable, m-dimensional non-separable,
and overlapping sub-components with independent block rotation matrices,
as required for CEC 2008, CEC 2010, and CEC 2013 Large-Scale Global Optimization (LSGO).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .base import VectorTransform


@dataclass
class ComponentGroup:
    """Specification of a single variable group within a decomposed landscape.

    Parameters
    ----------
    indices : np.ndarray
        1D array of variable indices belonging to this group.
    group_type : str, default="non_separable"
        Category of the component: "non_separable", "separable", or "overlapping".
    rotation_matrix : np.ndarray | None, optional
        Independent orthogonal rotation matrix of shape (m, m) applied to this group.
    name : str, default=""
        Optional descriptive label for this group.
    """

    indices: np.ndarray
    group_type: str = "non_separable"
    rotation_matrix: np.ndarray | None = None
    shift_vector: np.ndarray | None = None
    name: str = ""

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int64)
        if self.indices.ndim != 1:
            raise ValueError(f"Group indices must be 1D, got shape {self.indices.shape}")
        if len(self.indices) == 0:
            raise ValueError("Group indices cannot be empty")

        valid_types = {"non_separable", "separable", "overlapping"}
        if self.group_type not in valid_types:
            raise ValueError(
                f"Invalid group_type '{self.group_type}', must be one of {valid_types}"
            )

        m = len(self.indices)
        if self.rotation_matrix is not None:
            mat = np.asarray(self.rotation_matrix, dtype=np.float64)
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError(f"rotation_matrix must be square 2D array, got shape {mat.shape}")
            if mat.shape[0] != m:
                raise ValueError(
                    f"rotation_matrix shape {mat.shape} must match number of indices ({m})"
                )
            self.rotation_matrix = mat

        if self.shift_vector is not None:
            svec = np.asarray(self.shift_vector, dtype=np.float64)
            if svec.shape != (m,):
                raise ValueError(
                    f"shift_vector shape {svec.shape} must match number of indices ({m})"
                )
            self.shift_vector = svec

    @property
    def dimension(self) -> int:
        """Dimensionality of this sub-component."""
        return len(self.indices)


@dataclass
class DecomposedResult:
    """Structured container holding split sub-components for a 1D input vector."""

    components: list[np.ndarray]
    non_separable: list[np.ndarray] = field(default_factory=list)
    separable: np.ndarray | None = None
    overlapping: list[np.ndarray] = field(default_factory=list)

    def to_concatenated(self) -> np.ndarray:
        """Concatenate all transformed sub-components into a single 1D array."""
        if not self.components:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(self.components)


@dataclass
class DecomposedBatchResult:
    """Structured container holding split sub-components for a 2D batch of vectors."""

    components: list[np.ndarray]
    non_separable: list[np.ndarray] = field(default_factory=list)
    separable: np.ndarray | None = None
    overlapping: list[np.ndarray] = field(default_factory=list)

    def to_concatenated(self) -> np.ndarray:
        """Concatenate all transformed batch components along axis 1."""
        if not self.components:
            return np.empty((0, 0), dtype=np.float64)
        return np.concatenate(self.components, axis=1)


class GroupingTransform(VectorTransform):
    """
    Decomposition and grouping transform for high-dimensional optimization problems.

    Splits input vectors into separable, m-dimensional non-separable, and
    overlapping sub-components with independent block rotation matrices.

    Parameters
    ----------
    dimension : int
        Total dimensionality of the input space.
    groups : list[ComponentGroup], optional
        List of pre-configured component groups.
    permutation : np.ndarray | None, optional
        Optional permutation array of shape (dimension,) applied before grouping.
    non_separable_groups : list[Sequence[int]] | None, optional
        List of index sequences defining disjoint or specific non-separable components.
    block_rotations : list[np.ndarray] | None, optional
        Rotation matrices matching non_separable_groups.
    separable_indices : Sequence[int] | None, optional
        Indices corresponding to unrotated separable components.
    overlapping_groups : list[Sequence[int]] | None, optional
        List of index sequences defining overlapping sub-components.
    overlapping_rotations : list[np.ndarray] | None, optional
        Rotation matrices matching overlapping_groups.
    """

    def __init__(
        self,
        dimension: int,
        groups: list[ComponentGroup] | None = None,
        *,
        permutation: np.ndarray | Sequence[int] | None = None,
        non_separable_groups: list[Sequence[int] | np.ndarray] | None = None,
        block_rotations: list[np.ndarray] | None = None,
        separable_indices: Sequence[int] | np.ndarray | None = None,
        overlapping_groups: list[Sequence[int] | np.ndarray] | None = None,
        overlapping_rotations: list[np.ndarray] | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension = dimension

        if permutation is not None:
            p = np.asarray(permutation, dtype=np.int64)
            if p.shape != (dimension,):
                raise ValueError(f"permutation must have shape ({dimension},), got {p.shape}")
            if len(np.unique(p)) != dimension or np.min(p) < 0 or np.max(p) >= dimension:
                raise ValueError("permutation must be a valid permutation of range(dimension)")
            self.permutation: np.ndarray | None = p
        else:
            self.permutation = None

        if groups is not None:
            self.groups = list(groups)
        else:
            self.groups = []

            # 1. Add non-separable groups
            if non_separable_groups is not None:
                n_groups = len(non_separable_groups)
                rotations = block_rotations or [None] * n_groups
                if len(rotations) != n_groups:
                    raise ValueError(
                        f"Number of block_rotations ({len(rotations)}) must match "
                        f"number of non_separable_groups ({n_groups})"
                    )
                for idx_seq, rot in zip(non_separable_groups, rotations, strict=False):
                    self.groups.append(
                        ComponentGroup(
                            indices=np.asarray(idx_seq, dtype=np.int64),
                            group_type="non_separable",
                            rotation_matrix=rot,
                        )
                    )
            elif block_rotations is not None:
                # Infer sequential groups from block rotation dimensions
                curr = 0
                for rot in block_rotations:
                    r = np.asarray(rot)
                    m = r.shape[0]
                    indices = np.arange(curr, curr + m, dtype=np.int64)
                    self.groups.append(
                        ComponentGroup(
                            indices=indices,
                            group_type="non_separable",
                            rotation_matrix=r,
                        )
                    )
                    curr += m

            # 2. Add overlapping groups
            if overlapping_groups is not None:
                n_ov = len(overlapping_groups)
                ov_rotations = overlapping_rotations or [None] * n_ov
                if len(ov_rotations) != n_ov:
                    raise ValueError(
                        f"Number of overlapping_rotations ({len(ov_rotations)}) must match "
                        f"number of overlapping_groups ({n_ov})"
                    )
                for idx_seq, rot in zip(overlapping_groups, ov_rotations, strict=False):
                    self.groups.append(
                        ComponentGroup(
                            indices=np.asarray(idx_seq, dtype=np.int64),
                            group_type="overlapping",
                            rotation_matrix=rot,
                        )
                    )

            # 3. Add separable indices
            if separable_indices is not None and len(separable_indices) > 0:
                self.groups.append(
                    ComponentGroup(
                        indices=np.asarray(separable_indices, dtype=np.int64),
                        group_type="separable",
                        rotation_matrix=None,
                    )
                )

        # Validate that all group indices fall in [0, dimension - 1]
        for g in self.groups:
            if np.any(g.indices < 0) or np.any(g.indices >= dimension):
                raise ValueError(f"Group indices out of bounds [0, {dimension - 1}]: {g.indices}")

        self._check_partition_properties()

    def _check_partition_properties(self) -> None:
        """Analyze whether the configured groups form a disjoint partition of the space."""
        counts = np.zeros(self.dimension, dtype=np.int32)
        has_overlap_type = any(g.group_type == "overlapping" for g in self.groups)

        for g in self.groups:
            counts[g.indices] += 1

        self.is_overlapping: bool = has_overlap_type or bool(np.any(counts > 1))
        self.is_partition: bool = not self.is_overlapping and bool(np.all(counts == 1))

    @classmethod
    def from_sizes(
        cls,
        dimension: int,
        block_sizes: Sequence[int],
        *,
        block_rotations: list[np.ndarray] | None = None,
        separable_size: int = 0,
        permutation: np.ndarray | Sequence[int] | None = None,
    ) -> GroupingTransform:
        """Create a GroupingTransform from contiguous block sizes and optional separable remainder."""
        total_blocks = sum(block_sizes) + separable_size
        if total_blocks > dimension:
            raise ValueError(
                f"Sum of block sizes ({sum(block_sizes)}) + separable ({separable_size}) "
                f"exceeds dimension ({dimension})"
            )

        non_sep_groups = []
        curr = 0
        for s in block_sizes:
            non_sep_groups.append(np.arange(curr, curr + s, dtype=np.int64))
            curr += s

        sep_indices = (
            np.arange(curr, curr + separable_size, dtype=np.int64) if separable_size > 0 else None
        )

        return cls(
            dimension=dimension,
            permutation=permutation,
            non_separable_groups=non_sep_groups,
            block_rotations=block_rotations,
            separable_indices=sep_indices,
        )

    @classmethod
    def from_overlapping_window(
        cls,
        dimension: int,
        group_size: int,
        overlap_size: int,
        num_groups: int | None = None,
        *,
        block_rotations: list[np.ndarray] | None = None,
        permutation: np.ndarray | Sequence[int] | None = None,
    ) -> GroupingTransform:
        """Create overlapping sub-components with a sliding window sharing overlap_size variables.

        For example, group_size=100 and overlap_size=20 gives:
        Group 0: [0, 100), Group 1: [80, 180), Group 2: [160, 260), etc.
        """
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        if overlap_size < 0 or overlap_size >= group_size:
            raise ValueError(
                f"overlap_size must satisfy 0 <= overlap_size < group_size ({group_size}), "
                f"got {overlap_size}"
            )

        step = group_size - overlap_size
        groups_list = []
        curr = 0

        while True:
            end = curr + group_size
            if end > dimension:
                break
            groups_list.append(np.arange(curr, end, dtype=np.int64))
            curr += step
            if num_groups is not None and len(groups_list) >= num_groups:
                break

        if not groups_list:
            raise ValueError(f"Window of size {group_size} cannot fit within dimension {dimension}")

        return cls(
            dimension=dimension,
            overlapping_groups=groups_list,
            overlapping_rotations=block_rotations,
            permutation=permutation,
        )

    @classmethod
    def from_overlapping_sizes(
        cls,
        dimension: int,
        block_sizes: Sequence[int],
        overlap_size: int,
        *,
        block_rotations: list[np.ndarray] | None = None,
        shift_vectors: list[np.ndarray] | None = None,
        permutation: np.ndarray | Sequence[int] | None = None,
    ) -> GroupingTransform:
        """Create overlapping sub-components from non-uniform block sizes sharing overlap_size variables.

        Parameters
        ----------
        dimension : int
            Total input dimension.
        block_sizes : Sequence[int]
            Sizes of individual sub-components.
        overlap_size : int
            Number of variables shared between adjacent sub-components.
        block_rotations : list[np.ndarray] | None, optional
            Rotation matrices matching each block in block_sizes.
        shift_vectors : list[np.ndarray] | None, optional
            Optional per-group shift vectors (e.g. for conflicting optima).
        permutation : np.ndarray | Sequence[int] | None, optional
            Optional permutation applied before grouping.
        """
        if overlap_size < 0:
            raise ValueError(f"overlap_size must be non-negative, got {overlap_size}")
        for s in block_sizes:
            if s <= overlap_size:
                raise ValueError(
                    f"Each block size ({s}) must be strictly greater than overlap_size ({overlap_size})"
                )

        groups_list = []
        curr = 0
        n_groups = len(block_sizes)
        rotations = block_rotations or [None] * n_groups
        shifts = shift_vectors or [None] * n_groups

        for i, s in enumerate(block_sizes):
            start = curr - i * overlap_size
            end = curr + s - i * overlap_size
            if end > dimension:
                raise ValueError(f"Group {i} range [{start}, {end}) exceeds dimension {dimension}")
            indices = np.arange(start, end, dtype=np.int64)
            groups_list.append(
                ComponentGroup(
                    indices=indices,
                    group_type="overlapping",
                    rotation_matrix=rotations[i],
                    shift_vector=shifts[i],
                )
            )
            curr += s

        return cls(
            dimension=dimension,
            groups=groups_list,
            permutation=permutation,
        )

    def split(self, x: np.ndarray) -> DecomposedResult:
        """Split a 1D vector into transformed sub-components.

        Parameters
        ----------
        x : np.ndarray
            1D array of shape (dimension,).

        Returns
        -------
        DecomposedResult
            Container with non_separable, separable, overlapping, and all components.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.shape != (self.dimension,):
            raise ValueError(f"Input must have shape ({self.dimension},), got {x_arr.shape}")

        if self.permutation is not None:
            x_view = x_arr[self.permutation]
        else:
            x_view = x_arr

        all_comps: list[np.ndarray] = []
        non_sep: list[np.ndarray] = []
        sep: np.ndarray | None = None
        overlap: list[np.ndarray] = []

        for g in self.groups:
            sub = x_view[g.indices]
            if g.shift_vector is not None:
                sub = sub - g.shift_vector
            if g.rotation_matrix is not None:
                sub = g.rotation_matrix @ sub
            all_comps.append(sub)

            if g.group_type == "non_separable":
                non_sep.append(sub)
            elif g.group_type == "separable":
                sep = sub if sep is None else np.concatenate([sep, sub])
            elif g.group_type == "overlapping":
                overlap.append(sub)

        return DecomposedResult(
            components=all_comps,
            non_separable=non_sep,
            separable=sep,
            overlapping=overlap,
        )

    def split_batch(self, X: np.ndarray) -> DecomposedBatchResult:
        """Split a 2D batch of vectors into transformed sub-components.

        Parameters
        ----------
        X : np.ndarray
            2D array of shape (n_samples, dimension).

        Returns
        -------
        DecomposedBatchResult
            Container with non_separable, separable, overlapping, and all batch components.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.dimension:
            raise ValueError(
                f"Batch input must have shape (n, {self.dimension}), got {X_arr.shape}"
            )

        if self.permutation is not None:
            X_view = X_arr[:, self.permutation]
        else:
            X_view = X_arr

        all_comps: list[np.ndarray] = []
        non_sep: list[np.ndarray] = []
        sep: np.ndarray | None = None
        overlap: list[np.ndarray] = []

        for g in self.groups:
            sub = X_view[:, g.indices]
            if g.shift_vector is not None:
                sub = sub - g.shift_vector
            if g.rotation_matrix is not None:
                sub = sub @ g.rotation_matrix.T
            all_comps.append(sub)

            if g.group_type == "non_separable":
                non_sep.append(sub)
            elif g.group_type == "separable":
                sep = sub if sep is None else np.concatenate([sep, sub], axis=1)
            elif g.group_type == "overlapping":
                overlap.append(sub)

        return DecomposedBatchResult(
            components=all_comps,
            non_separable=non_sep,
            separable=sep,
            overlapping=overlap,
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply transformation to input vector.

        If groups form a disjoint partition of the search space, reconstructs
        the transformed vector of shape (dimension,).
        If groups overlap or do not cover all variables, returns the concatenated
        sub-components.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        if x_arr.shape != (self.dimension,):
            raise ValueError(f"Input must have shape ({self.dimension},), got {x_arr.shape}")

        if self.is_partition:
            if self.permutation is not None:
                x_perm = x_arr[self.permutation]
            else:
                x_perm = x_arr

            out_perm = np.empty_like(x_perm)
            for g in self.groups:
                sub = x_perm[g.indices]
                if g.rotation_matrix is not None:
                    sub = g.rotation_matrix @ sub
                out_perm[g.indices] = sub

            if self.permutation is not None:
                # Invert permutation: out[P[i]] = out_perm[i]
                out = np.empty_like(x_arr)
                out[self.permutation] = out_perm
                return out
            return out_perm

        return self.split(x_arr).to_concatenated()

    def transform_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch transformation of (n_samples, dimension) array."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.dimension:
            raise ValueError(
                f"Batch input must have shape (n, {self.dimension}), got {X_arr.shape}"
            )

        if self.is_partition:
            if self.permutation is not None:
                X_perm = X_arr[:, self.permutation]
            else:
                X_perm = X_arr

            out_perm = np.empty_like(X_perm)
            for g in self.groups:
                sub = X_perm[:, g.indices]
                if g.rotation_matrix is not None:
                    sub = sub @ g.rotation_matrix.T
                out_perm[:, g.indices] = sub

            if self.permutation is not None:
                out = np.empty_like(X_arr)
                out[:, self.permutation] = out_perm
                return out
            return out_perm

        return self.split_batch(X_arr).to_concatenated()


# Alias DecomposedTransform to GroupingTransform
DecomposedTransform = GroupingTransform
