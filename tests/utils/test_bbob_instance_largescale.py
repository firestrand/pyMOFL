"""Tests for large-scale BBOB instance generation utilities."""

import numpy as np
import pytest

from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator


class TestBBOBLargeScaleGeneration:
    """Tests for generate_permutation and generate_block_diagonal_rotation."""

    @pytest.fixture
    def gen(self):
        return BBOBInstanceGenerator()

    def test_generate_permutation_valid(self, gen):
        """Should return a valid permutation."""
        perm = gen.generate_permutation(dim=100, seed=42)
        assert perm.shape == (100,)
        assert sorted(perm) == list(range(100))

    def test_generate_permutation_deterministic(self, gen):
        """Same seed should give same permutation."""
        p1 = gen.generate_permutation(dim=50, seed=42)
        p2 = gen.generate_permutation(dim=50, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_generate_permutation_truncated_swap_range(self, gen):
        """Average displacement should be moderate (not fully shuffled)."""
        dim = 120
        perm = gen.generate_permutation(dim=dim, seed=42)
        displacements = np.abs(perm - np.arange(dim))
        avg_displacement = np.mean(displacements)
        # With truncated swaps (nb_swaps=D/3, swap_range=D/3),
        # average displacement should be well below D/2 (full shuffle)
        assert avg_displacement < dim / 2
        # Should actually move some elements (not identity)
        assert avg_displacement > 0

    def test_generate_permutation_identity_small_dim(self, gen):
        """For D <= 40, permutation should be identity."""
        perm = gen.generate_permutation(dim=40, seed=42)
        np.testing.assert_array_equal(perm, np.arange(40))

    def test_generate_permutation_invalid_dim_raises(self, gen):
        """Non-positive dimensions should be rejected."""
        with pytest.raises(ValueError, match="positive"):
            gen.generate_permutation(dim=0, seed=42)
        with pytest.raises(ValueError, match="positive"):
            gen.generate_permutation(dim=-5, seed=42)

    def test_generate_block_diagonal_valid(self, gen):
        """Should return list of orthogonal blocks."""
        blocks = gen.generate_block_diagonal_rotation(dim=100, seed=42, block_size=40)
        total = sum(b.shape[0] for b in blocks)
        assert total == 100
        for b in blocks:
            # Check orthogonality
            np.testing.assert_array_almost_equal(b @ b.T, np.eye(b.shape[0]), decimal=10)

    def test_generate_block_diagonal_deterministic(self, gen):
        """Same seed should give same blocks."""
        b1 = gen.generate_block_diagonal_rotation(dim=80, seed=42, block_size=40)
        b2 = gen.generate_block_diagonal_rotation(dim=80, seed=42, block_size=40)
        assert len(b1) == len(b2)
        for a, b in zip(b1, b2, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_generate_block_diagonal_shapes(self, gen):
        """Blocks should have correct sizes."""
        blocks = gen.generate_block_diagonal_rotation(dim=100, seed=42, block_size=40)
        sizes = [b.shape[0] for b in blocks]
        # 100 = 40 + 40 + 20
        assert sizes == [40, 40, 20]

    def test_generate_block_diagonal_small_dim(self, gen):
        """For D <= block_size, should return single full-size block."""
        blocks = gen.generate_block_diagonal_rotation(dim=30, seed=42, block_size=40)
        assert len(blocks) == 1
        assert blocks[0].shape == (30, 30)

    def test_generate_block_diagonal_invalid_inputs_raise(self, gen):
        """Invalid dim or block_size should raise ValueError."""
        with pytest.raises(ValueError, match="Dimension must"):
            gen.generate_block_diagonal_rotation(dim=0, seed=42, block_size=40)
        with pytest.raises(ValueError, match="block_size must"):
            gen.generate_block_diagonal_rotation(dim=10, seed=42, block_size=0)
        with pytest.raises(ValueError, match="block_size must"):
            gen.generate_block_diagonal_rotation(dim=10, seed=42, block_size=-1)

    def test_generate_block_diagonal_orthogonality(self, gen):
        """Each block should be orthogonal."""
        blocks = gen.generate_block_diagonal_rotation(dim=160, seed=42, block_size=40)
        for block in blocks:
            np.testing.assert_array_almost_equal(
                block @ block.T, np.eye(block.shape[0]), decimal=10
            )
