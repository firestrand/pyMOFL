"""Tests for Uniform noise transform (COCO scale-dependent noise)."""

import numpy as np
import pytest

from pyMOFL.functions.transformations.uniform_noise import UniformNoiseTransform


class TestUniformNoiseTransform:
    """Tests for UniformNoiseTransform.

    COCO formula: f * U^beta * max(1, (1e9/(f+1e-99))^(alpha*U'))
    where U, U' ~ Uniform(0,1).
    """

    def test_seed_determinism(self):
        """Same seed produces identical results."""
        t1 = UniformNoiseTransform(alpha=0.01, beta=0.01, seed=42)
        t2 = UniformNoiseTransform(alpha=0.01, beta=0.01, seed=42)
        assert t1(10.0) == t2(10.0)

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        t1 = UniformNoiseTransform(alpha=0.5, beta=0.5, seed=42)
        t2 = UniformNoiseTransform(alpha=0.5, beta=0.5, seed=99)
        assert t1(10.0) != t2(10.0)

    def test_large_f_minimal_amplification(self):
        """For large f, the max(1, (1e9/f)^alpha) term ≈ 1.

        So result ≈ f * U^beta, which is ≤ f.
        """
        t = UniformNoiseTransform(alpha=0.5, beta=0.01, seed=42)
        result = t(1e12)
        # With beta=0.01, U^beta ≈ 1, so result ≈ 1e12
        assert result > 0  # multiplicative, stays positive for positive input
        assert result <= 1e12 * 1.01  # shouldn't exceed by much

    def test_small_f_large_amplification(self):
        """For small f, the (1e9/f)^alpha term can be >> 1, amplifying noise."""
        t = UniformNoiseTransform(alpha=1.0, beta=0.01, seed=42)
        result = t(1e-6)
        # (1e9 / 1e-6)^(1.0 * U') can be very large
        # Result can be much larger than input
        assert result >= 0  # should be non-negative for non-negative input

    def test_result_is_float(self):
        t = UniformNoiseTransform(alpha=0.5, beta=0.5, seed=42)
        result = t(10.0)
        assert isinstance(result, float)

    def test_zero_input(self):
        """f=0 should return 0 (multiplicative)."""
        t = UniformNoiseTransform(alpha=0.5, beta=0.5, seed=42)
        assert t(0.0) == pytest.approx(0.0)

    def test_transform_batch(self):
        """Batch transform should handle arrays."""
        t = UniformNoiseTransform(alpha=0.5, beta=0.5, seed=42)
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        results = t.transform_batch(Y)
        assert results.shape == Y.shape
        assert np.all(np.isfinite(results))

    def test_batch_seed_determinism(self):
        """Batch results are deterministic with same seed."""
        Y = np.array([1.0, 10.0, 100.0])
        t1 = UniformNoiseTransform(alpha=0.5, beta=0.5, seed=42)
        t2 = UniformNoiseTransform(alpha=0.5, beta=0.5, seed=42)
        np.testing.assert_array_equal(t1.transform_batch(Y), t2.transform_batch(Y))

    def test_repr(self):
        t = UniformNoiseTransform(alpha=0.5, beta=0.3, seed=42)
        assert "UniformNoiseTransform" in repr(t)

    def test_positive_input_positive_output(self):
        """Positive input should yield positive output (multiplicative model)."""
        t = UniformNoiseTransform(alpha=1.0, beta=1.0, seed=42)
        for _ in range(50):
            assert t(10.0) > 0
