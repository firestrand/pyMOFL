"""Tests for Cauchy noise transform (COCO heavy-tailed additive noise)."""

import numpy as np
import pytest

from pyMOFL.functions.transformations.cauchy_noise import CauchyNoiseTransform


class TestCauchyNoiseTransform:
    """Tests for CauchyNoiseTransform.

    COCO formula: f + alpha * max(0, 1000 + I_{U<p} * N1/(|N2|+1e-99))
    where U ~ Uniform(0,1), N1, N2 ~ Normal(0,1), I is indicator function.
    """

    def test_seed_determinism(self):
        """Same seed produces identical results."""
        t1 = CauchyNoiseTransform(alpha=1.0, p=0.2, seed=42)
        t2 = CauchyNoiseTransform(alpha=1.0, p=0.2, seed=42)
        assert t1(10.0) == t2(10.0)

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        t1 = CauchyNoiseTransform(alpha=1.0, p=0.2, seed=42)
        # Run several times to ensure at least one differs
        results1 = [t1(10.0) for _ in range(10)]
        t2 = CauchyNoiseTransform(alpha=1.0, p=0.2, seed=99)
        results2 = [t2(10.0) for _ in range(10)]
        assert results1 != results2

    def test_alpha_zero_is_identity(self):
        """alpha=0 → no noise added, f + 0*(...) = f."""
        t = CauchyNoiseTransform(alpha=0.0, p=0.5, seed=42)
        assert t(5.0) == pytest.approx(5.0)
        assert t(-3.0) == pytest.approx(-3.0)

    def test_additive_non_negative(self):
        """The noise term max(0, ...) is always >= 0, so result >= f."""
        t = CauchyNoiseTransform(alpha=1.0, p=0.2, seed=42)
        for _ in range(100):
            result = t(10.0)
            assert result >= 10.0 - 1e-10  # allow tiny float error

    def test_result_is_float(self):
        t = CauchyNoiseTransform(alpha=0.5, p=0.1, seed=42)
        result = t(10.0)
        assert isinstance(result, float)

    def test_p_zero_no_cauchy(self):
        """p=0 means indicator I_{U<0} = 0 always, so noise = alpha*max(0, 1000).

        Result = f + alpha * 1000.
        """
        t = CauchyNoiseTransform(alpha=0.01, p=0.0, seed=42)
        # With p=0, the Cauchy part never triggers: noise = max(0, 1000) = 1000
        result = t(5.0)
        assert result == pytest.approx(5.0 + 0.01 * 1000.0)

    def test_p_one_always_cauchy(self):
        """p=1 means indicator always triggers."""
        t = CauchyNoiseTransform(alpha=0.01, p=1.0, seed=42)
        result = t(5.0)
        # Should differ from p=0 case (usually)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_transform_batch(self):
        t = CauchyNoiseTransform(alpha=0.5, p=0.2, seed=42)
        Y = np.array([1.0, 2.0, 3.0, 4.0])
        results = t.transform_batch(Y)
        assert results.shape == Y.shape
        assert np.all(np.isfinite(results))

    def test_batch_seed_determinism(self):
        Y = np.array([1.0, 10.0, 100.0])
        t1 = CauchyNoiseTransform(alpha=0.5, p=0.2, seed=42)
        t2 = CauchyNoiseTransform(alpha=0.5, p=0.2, seed=42)
        np.testing.assert_array_equal(t1.transform_batch(Y), t2.transform_batch(Y))

    def test_repr(self):
        t = CauchyNoiseTransform(alpha=0.5, p=0.2, seed=42)
        assert "CauchyNoiseTransform" in repr(t)

    def test_negative_input_noise_still_additive(self):
        """Even for negative f, noise is additive and non-negative."""
        t = CauchyNoiseTransform(alpha=1.0, p=0.2, seed=42)
        for _ in range(50):
            result = t(-100.0)
            assert result >= -100.0 - 1e-10
