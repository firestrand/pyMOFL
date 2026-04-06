"""Tests for Gaussian noise transform (COCO log-normal multiplicative noise)."""

import numpy as np
import pytest

from pyMOFL.functions.transformations.gaussian_noise import GaussianNoiseTransform


class TestGaussianNoiseTransform:
    """Tests for GaussianNoiseTransform: f * exp(beta * N(0,1))."""

    def test_seed_determinism(self):
        """Same seed produces identical results."""
        t1 = GaussianNoiseTransform(beta=1.0, seed=42)
        t2 = GaussianNoiseTransform(beta=1.0, seed=42)
        assert t1(10.0) == t2(10.0)

    def test_different_seeds_differ(self):
        """Different seeds produce different results (with high probability)."""
        t1 = GaussianNoiseTransform(beta=1.0, seed=42)
        t2 = GaussianNoiseTransform(beta=1.0, seed=99)
        assert t1(10.0) != t2(10.0)

    def test_beta_zero_is_identity(self):
        """beta=0 means exp(0*N) = 1, so f*1 = f."""
        t = GaussianNoiseTransform(beta=0.0, seed=42)
        assert t(5.0) == pytest.approx(5.0)
        assert t(-3.0) == pytest.approx(-3.0)

    def test_zero_input_returns_zero(self):
        """f=0 → 0 * exp(...) = 0."""
        t = GaussianNoiseTransform(beta=1.0, seed=42)
        assert t(0.0) == pytest.approx(0.0)

    def test_result_is_float(self):
        """Output should be a plain float."""
        t = GaussianNoiseTransform(beta=0.5, seed=42)
        result = t(10.0)
        assert isinstance(result, float)

    def test_multiplicative_nature(self):
        """Result should equal f * exp(beta * N(0,1))."""
        beta = 0.5
        seed = 123
        # Generate the expected noise value
        rng = np.random.default_rng(seed)
        n = rng.standard_normal()
        expected = 10.0 * np.exp(beta * n)

        t = GaussianNoiseTransform(beta=beta, seed=seed)
        assert t(10.0) == pytest.approx(expected)

    def test_negative_input(self):
        """Noise should work for negative function values."""
        t = GaussianNoiseTransform(beta=0.01, seed=42)
        result = t(-100.0)
        # With small beta, result should be close to -100
        assert abs(result - (-100.0)) < 10.0

    def test_transform_batch(self):
        """Batch transform should apply noise independently to each element."""
        t = GaussianNoiseTransform(beta=0.5, seed=42)
        Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        results = t.transform_batch(Y)
        assert results.shape == Y.shape
        # Each element should be different from deterministic value
        # (with high probability for beta=0.5)
        assert not np.allclose(results, Y)

    def test_batch_seed_determinism(self):
        """Batch results are deterministic with same seed."""
        Y = np.array([1.0, 2.0, 3.0])
        t1 = GaussianNoiseTransform(beta=0.5, seed=42)
        t2 = GaussianNoiseTransform(beta=0.5, seed=42)
        np.testing.assert_array_equal(t1.transform_batch(Y), t2.transform_batch(Y))

    def test_repr(self):
        t = GaussianNoiseTransform(beta=0.5, seed=42)
        assert "GaussianNoiseTransform" in repr(t)
        assert "0.5" in repr(t)

    def test_large_beta_amplifies(self):
        """Large beta should produce highly variable results."""
        t = GaussianNoiseTransform(beta=10.0, seed=42)
        results = [t(1.0) for _ in range(100)]
        # With beta=10, variance should be enormous
        assert max(results) / min(abs(r) for r in results if r != 0) > 10

    def test_small_beta_near_identity(self):
        """Very small beta should barely change the value."""
        t = GaussianNoiseTransform(beta=1e-10, seed=42)
        assert t(100.0) == pytest.approx(100.0, rel=1e-6)
