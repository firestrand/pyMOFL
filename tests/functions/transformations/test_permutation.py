"""Tests for PermutationTransform."""

import numpy as np
import pytest

from pyMOFL.functions.transformations.permutation import PermutationTransform


class TestPermutationTransform:
    """Tests for PermutationTransform: x -> x[perm]."""

    def test_identity_permutation(self):
        """Identity permutation should not change the vector."""
        perm = np.array([0, 1, 2, 3, 4])
        t = PermutationTransform(perm)
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        np.testing.assert_array_equal(t(x), x)

    def test_swap_permutation(self):
        """Simple swap should reorder elements."""
        perm = np.array([1, 0, 2])
        t = PermutationTransform(perm)
        x = np.array([10.0, 20.0, 30.0])
        expected = np.array([20.0, 10.0, 30.0])
        np.testing.assert_array_equal(t(x), expected)

    def test_reverse_permutation(self):
        """Reverse permutation should reverse the vector."""
        perm = np.array([4, 3, 2, 1, 0])
        t = PermutationTransform(perm)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        np.testing.assert_array_equal(t(x), expected)

    def test_preserves_values(self):
        """All values should be preserved (just reordered)."""
        perm = np.array([2, 0, 3, 1])
        t = PermutationTransform(perm)
        x = np.array([10.0, 20.0, 30.0, 40.0])
        result = t(x)
        assert sorted(result) == sorted(x)

    def test_batch_consistency(self):
        """Batch transform should match individual transforms."""
        perm = np.array([2, 0, 1])
        t = PermutationTransform(perm)
        X = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        batch_result = t.transform_batch(X)
        for i in range(X.shape[0]):
            single_result = t(X[i])
            np.testing.assert_array_equal(batch_result[i], single_result)

    def test_repr(self):
        perm = np.array([1, 0])
        t = PermutationTransform(perm)
        assert "PermutationTransform" in repr(t)

    def test_invalid_permutation_nd(self):
        """Permutation must be 1D."""
        with pytest.raises(ValueError, match="one-dimensional"):
            PermutationTransform(np.array([[0, 1], [1, 0]]))

    def test_invalid_permutation(self):
        """Non-permutation should raise ValueError."""
        with pytest.raises(ValueError, match="valid permutation"):
            PermutationTransform(np.array([0, 0, 1]))

    def test_large_permutation(self):
        """Should handle larger dimensions."""
        dim = 100
        rng = np.random.default_rng(42)
        perm = rng.permutation(dim)
        t = PermutationTransform(perm)
        x = np.arange(dim, dtype=float)
        result = t(x)
        np.testing.assert_array_equal(result, x[perm])
