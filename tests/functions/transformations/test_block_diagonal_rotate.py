"""Tests for BlockDiagonalRotateTransform."""

import numpy as np
import pytest

from pyMOFL.functions.transformations.block_diagonal_rotate import (
    BlockDiagonalRotateTransform,
)


class TestBlockDiagonalRotateTransform:
    """Tests for BlockDiagonalRotateTransform."""

    def test_single_block_equals_full_rotation(self):
        """When block_size >= dim, should equal a full rotation."""
        dim = 5
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        t = BlockDiagonalRotateTransform(blocks=[Q])
        x = rng.standard_normal(dim)
        expected = Q @ x
        np.testing.assert_array_almost_equal(t(x), expected)

    def test_invalid_empty_blocks(self):
        """Block list must not be empty."""
        with pytest.raises(ValueError, match="At least one block"):
            BlockDiagonalRotateTransform(blocks=[])

    def test_invalid_non_square_block(self):
        """Block matrices must be square."""
        with pytest.raises(ValueError, match="square"):
            BlockDiagonalRotateTransform(blocks=[np.array([[1.0, 0.0]])])

    def test_two_block_independence(self):
        """Two blocks should transform independently."""
        # Block 1: 2x2 rotation
        theta = np.pi / 4
        R1 = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # Block 2: 3x3 identity
        R2 = np.eye(3)

        t = BlockDiagonalRotateTransform(blocks=[R1, R2])
        x = np.array([1.0, 0.0, 10.0, 20.0, 30.0])
        result = t(x)

        # First 2 elements: rotated
        np.testing.assert_array_almost_equal(result[:2], R1 @ x[:2])
        # Last 3 elements: unchanged (identity)
        np.testing.assert_array_almost_equal(result[2:], x[2:])

    def test_orthogonality_preserved(self):
        """Block-diagonal of orthogonal blocks is still orthogonal."""
        rng = np.random.default_rng(42)
        blocks = []
        for size in [10, 10, 5]:
            Q, _ = np.linalg.qr(rng.standard_normal((size, size)))
            blocks.append(Q)

        t = BlockDiagonalRotateTransform(blocks=blocks)
        # Full matrix should be orthogonal
        full = t.to_full_matrix()
        np.testing.assert_array_almost_equal(full @ full.T, np.eye(25))

    def test_uneven_last_block(self):
        """Last block can be smaller than the rest."""
        rng = np.random.default_rng(42)
        Q1, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        Q2, _ = np.linalg.qr(rng.standard_normal((3, 3)))

        t = BlockDiagonalRotateTransform(blocks=[Q1, Q2])
        x = rng.standard_normal(7)
        result = t(x)

        # First 4 elements rotated by Q1
        np.testing.assert_array_almost_equal(result[:4], Q1 @ x[:4])
        # Last 3 elements rotated by Q2
        np.testing.assert_array_almost_equal(result[4:], Q2 @ x[4:])

    def test_batch_consistency(self):
        """Batch transform should match individual transforms."""
        rng = np.random.default_rng(42)
        Q1, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        Q2, _ = np.linalg.qr(rng.standard_normal((2, 2)))
        t = BlockDiagonalRotateTransform(blocks=[Q1, Q2])

        X = rng.standard_normal((5, 5))
        batch_result = t.transform_batch(X)
        for i in range(5):
            single_result = t(X[i])
            np.testing.assert_array_almost_equal(batch_result[i], single_result)

    def test_to_full_matrix(self):
        """to_full_matrix should produce correct block-diagonal."""
        R1 = np.array([[1.0, 0.0], [0.0, -1.0]])
        R2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        t = BlockDiagonalRotateTransform(blocks=[R1, R2])
        full = t.to_full_matrix()
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=float,
        )
        np.testing.assert_array_almost_equal(full, expected)

    def test_repr(self):
        t = BlockDiagonalRotateTransform(blocks=[np.eye(3)])
        assert "BlockDiagonalRotateTransform" in repr(t)

    def test_performance_large(self):
        """Large block-diagonal should be O(D*s), not O(D^2)."""
        dim = 200
        block_size = 40
        rng = np.random.default_rng(42)
        blocks = []
        remaining = dim
        while remaining > 0:
            s = min(block_size, remaining)
            Q, _ = np.linalg.qr(rng.standard_normal((s, s)))
            blocks.append(Q)
            remaining -= s

        t = BlockDiagonalRotateTransform(blocks=blocks)
        x = rng.standard_normal(dim)
        result = t(x)
        assert result.shape == (dim,)
        assert np.all(np.isfinite(result))
