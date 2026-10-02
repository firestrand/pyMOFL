"""
Unit tests for GroupingTransform and DecomposedTransform pipeline.

Tests variable splitting into separable, m-dimensional non-separable, and
overlapping sub-components with independent block rotation matrices.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyMOFL.factories.data_loader import DataLoader
from pyMOFL.factories.transform_builder import TransformBuilder
from pyMOFL.functions.transformations.decomposition import (
    ComponentGroup,
    DecomposedBatchResult,
    DecomposedResult,
    DecomposedTransform,
    GroupingTransform,
)


class TestComponentGroup:
    """Tests for ComponentGroup specification."""

    def test_valid_creation(self):
        cg = ComponentGroup(indices=np.array([0, 1, 2]), group_type="non_separable")
        assert cg.dimension == 3
        assert cg.group_type == "non_separable"
        assert cg.rotation_matrix is None

    def test_empty_indices_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            ComponentGroup(indices=np.array([], dtype=int))

    def test_invalid_group_type(self):
        with pytest.raises(ValueError, match="Invalid group_type"):
            ComponentGroup(indices=np.array([0, 1]), group_type="invalid_type")

    def test_rotation_matrix_dimension_mismatch(self):
        R = np.eye(4)
        with pytest.raises(ValueError, match="must match number of indices"):
            ComponentGroup(indices=np.array([0, 1, 2]), rotation_matrix=R)

    def test_non_square_rotation_matrix(self):
        R = np.ones((3, 4))
        with pytest.raises(ValueError, match="must be square"):
            ComponentGroup(indices=np.array([0, 1, 2]), rotation_matrix=R)

    def test_shift_vector_support(self):
        cg = ComponentGroup(
            indices=np.array([0, 1, 2]),
            group_type="overlapping",
            shift_vector=np.array([1.0, 2.0, 3.0]),
        )
        assert cg.shift_vector is not None
        np.testing.assert_allclose(cg.shift_vector, [1.0, 2.0, 3.0])

    def test_shift_vector_dimension_mismatch(self):
        with pytest.raises(ValueError, match="must match number of indices"):
            ComponentGroup(indices=np.array([0, 1]), shift_vector=np.array([1.0, 2.0, 3.0]))


class TestGroupingTransform:
    """Tests for GroupingTransform / DecomposedTransform."""

    def test_alias_identity(self):
        assert DecomposedTransform is GroupingTransform

    def test_invalid_dimension(self):
        with pytest.raises(ValueError, match="positive"):
            GroupingTransform(dimension=0)

    def test_out_of_bounds_indices(self):
        with pytest.raises(ValueError, match="out of bounds"):
            GroupingTransform(
                dimension=5,
                groups=[ComponentGroup(indices=np.array([0, 5]))],
            )

    def test_from_sizes_separable_and_non_separable(self):
        """Test partitioning into 2 non-separable groups of size 3 and 1 separable of size 2."""
        R1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        R2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)

        gt = GroupingTransform.from_sizes(
            dimension=8,
            block_sizes=[3, 3],
            block_rotations=[R1, R2],
            separable_size=2,
        )

        assert gt.is_partition is True
        assert gt.is_overlapping is False
        assert len(gt.groups) == 3

        x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        res = gt.split(x)

        assert isinstance(res, DecomposedResult)
        assert len(res.non_separable) == 2
        np.testing.assert_allclose(res.non_separable[0], [2, 3, 1])
        np.testing.assert_allclose(res.non_separable[1], [-4, 5, -6])
        assert res.separable is not None
        np.testing.assert_allclose(res.separable, [7, 8])

        # Reconstructed __call__
        x_rot = gt(x)
        np.testing.assert_allclose(x_rot, [2, 3, 1, -4, 5, -6, 7, 8])

    def test_from_sizes_overflow_raises(self):
        with pytest.raises(ValueError, match="exceeds dimension"):
            GroupingTransform.from_sizes(dimension=5, block_sizes=[3, 3])

    def test_overlapping_window(self):
        """Test overlapping sub-components sharing variables."""
        gt = GroupingTransform.from_overlapping_window(
            dimension=10,
            group_size=4,
            overlap_size=2,
        )

        assert gt.is_overlapping is True
        assert gt.is_partition is False
        # Window positions: [0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]
        assert len(gt.groups) == 4

        x = np.arange(10, dtype=float)
        res = gt.split(x)
        assert len(res.overlapping) == 4
        np.testing.assert_allclose(res.overlapping[0], [0, 1, 2, 3])
        np.testing.assert_allclose(res.overlapping[1], [2, 3, 4, 5])
        np.testing.assert_allclose(res.overlapping[2], [4, 5, 6, 7])
        np.testing.assert_allclose(res.overlapping[3], [6, 7, 8, 9])

        # Overlapping transform returns concatenated vector of length 16
        out = gt(x)
        assert out.shape == (16,)
        np.testing.assert_allclose(out[:4], [0, 1, 2, 3])
        np.testing.assert_allclose(out[4:8], [2, 3, 4, 5])

    def test_overlapping_window_validation(self):
        with pytest.raises(ValueError, match="group_size must be positive"):
            GroupingTransform.from_overlapping_window(dimension=10, group_size=0, overlap_size=0)
        with pytest.raises(ValueError, match="overlap_size must satisfy"):
            GroupingTransform.from_overlapping_window(dimension=10, group_size=4, overlap_size=4)

    def test_permutation_application_and_reconstruction(self):
        """Test permutation reorders inputs and reconstructed vector inverts it."""
        P = np.array([3, 2, 1, 0])  # Reversed permutation
        R = np.array([[0, 1], [1, 0]], dtype=float)  # Swap

        gt = GroupingTransform(
            dimension=4,
            permutation=P,
            non_separable_groups=[[0, 1]],
            block_rotations=[R],
            separable_indices=[2, 3],
        )

        x = np.array([10, 20, 30, 40], dtype=float)
        # x[P] = [40, 30, 20, 10]
        res = gt.split(x)
        # non_separable: R @ [40, 30] = [30, 40]
        np.testing.assert_allclose(res.non_separable[0], [30, 40])
        # separable: [20, 10]
        assert res.separable is not None
        np.testing.assert_allclose(res.separable, [20, 10])

        # __call__ inverts permutation:
        # permuted output is [30, 40, 20, 10]
        # x_out[P[i]] = permuted[i] => x_out[3]=30, x_out[2]=40, x_out[1]=20, x_out[0]=10
        x_rot = gt(x)
        np.testing.assert_allclose(x_rot, [10, 20, 40, 30])

    def test_batch_splitting_and_transformation(self):
        """Test split_batch and transform_batch produce consistent results with single evaluation."""
        R1 = np.array([[0, -1], [1, 0]], dtype=float)
        R2 = np.array([[1, 0], [0, 1]], dtype=float)

        gt = GroupingTransform.from_sizes(
            dimension=6,
            block_sizes=[2, 2],
            block_rotations=[R1, R2],
            separable_size=2,
        )

        rng = np.random.default_rng(123)
        X = rng.uniform(-10.0, 10.0, size=(7, 6))

        batch_res = gt.split_batch(X)
        assert isinstance(batch_res, DecomposedBatchResult)
        assert len(batch_res.non_separable) == 2
        assert batch_res.non_separable[0].shape == (7, 2)
        assert batch_res.non_separable[1].shape == (7, 2)
        assert batch_res.separable is not None
        assert batch_res.separable.shape == (7, 2)

        # Compare each row against split(x)
        for i in range(7):
            single_res = gt.split(X[i])
            np.testing.assert_allclose(batch_res.non_separable[0][i], single_res.non_separable[0])
            np.testing.assert_allclose(batch_res.non_separable[1][i], single_res.non_separable[1])
            assert single_res.separable is not None
            np.testing.assert_allclose(batch_res.separable[i], single_res.separable)

        # Batch transform
        X_trans = gt.transform_batch(X)
        for i in range(7):
            np.testing.assert_allclose(X_trans[i], gt(X[i]))

    def test_transform_builder_construction(self):
        """Test constructing GroupingTransform via TransformBuilder."""
        builder = TransformBuilder(DataLoader())
        params = {
            "block_sizes": [2, 2],
            "separable_size": 1,
            "block_rotations": [np.eye(2), -np.eye(2)],
        }
        transform = builder.build("grouping", params, dimension=5)
        assert isinstance(transform, GroupingTransform)
        assert len(transform.groups) == 3
        assert transform.is_partition is True

    def test_from_overlapping_sizes(self):
        """Test from_overlapping_sizes with varying block sizes, shifts, and rotations."""
        # Dim 15: block sizes 6, 7, 6 with overlap 2:
        # Group 0: [0, 6)
        # Group 1: [4, 11)
        # Group 2: [9, 15)
        R0 = -np.eye(6)
        R1 = np.eye(7)
        R2 = -np.eye(6)
        shift0 = np.ones(6) * 1.0
        shift1 = np.ones(7) * 2.0
        shift2 = np.ones(6) * 3.0

        gt = GroupingTransform.from_overlapping_sizes(
            dimension=15,
            block_sizes=[6, 7, 6],
            overlap_size=2,
            block_rotations=[R0, R1, R2],
            shift_vectors=[shift0, shift1, shift2],
        )

        assert gt.is_overlapping is True
        assert gt.is_partition is False
        assert len(gt.groups) == 3

        x = np.arange(15, dtype=float)
        # Group 0 indices: 0..5, shifted by 1.0, rotated by -I: - (x[0..5] - 1.0)
        # Group 1 indices: 4..10, shifted by 2.0, rotated by I: x[4..10] - 2.0
        # Group 2 indices: 9..14, shifted by 3.0, rotated by -I: - (x[9..14] - 3.0)
        res = gt.split(x)
        assert len(res.overlapping) == 3
        np.testing.assert_allclose(res.overlapping[0], -(x[0:6] - 1.0))
        np.testing.assert_allclose(res.overlapping[1], x[4:11] - 2.0)
        np.testing.assert_allclose(res.overlapping[2], -(x[9:15] - 3.0))

        # Test batch splitting consistency
        X = np.stack([x, x + 5.0])
        batch_res = gt.split_batch(X)
        for i in range(2):
            s_res = gt.split(X[i])
            for g in range(3):
                np.testing.assert_allclose(batch_res.overlapping[g][i], s_res.overlapping[g])

    def test_from_overlapping_sizes_validation(self):
        """Test error handling in from_overlapping_sizes."""
        with pytest.raises(ValueError, match="non-negative"):
            GroupingTransform.from_overlapping_sizes(
                dimension=10, block_sizes=[4, 4], overlap_size=-1
            )
        with pytest.raises(ValueError, match="strictly greater than overlap_size"):
            GroupingTransform.from_overlapping_sizes(
                dimension=10, block_sizes=[4, 2], overlap_size=2
            )
        with pytest.raises(ValueError, match="exceeds dimension"):
            GroupingTransform.from_overlapping_sizes(
                dimension=10, block_sizes=[6, 6], overlap_size=1
            )

    def test_transform_builder_dimension_keyed_rotations(self):
        """Test TransformBuilder with dimension-keyed block rotation mappings."""
        builder = TransformBuilder(DataLoader())
        params = {
            "block_sizes": [2, 3, 2],
            "overlap_size": 1,
            "block_rotations": {
                "2": np.array([[0, 1], [1, 0]], dtype=float),
                "3": np.eye(3, dtype=float),
            },
        }
        # Dim = 2 + 3 + 2 - 2*1 = 5
        transform = builder.build("grouping", params, dimension=5)
        assert isinstance(transform, GroupingTransform)
        assert len(transform.groups) == 3
        np.testing.assert_allclose(transform.groups[0].rotation_matrix, [[0, 1], [1, 0]])
        np.testing.assert_allclose(transform.groups[1].rotation_matrix, np.eye(3))
        np.testing.assert_allclose(transform.groups[2].rotation_matrix, [[0, 1], [1, 0]])
