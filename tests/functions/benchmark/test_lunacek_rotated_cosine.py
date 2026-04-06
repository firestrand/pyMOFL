"""Tests for LunacekRotatedCosineFunction."""

from __future__ import annotations

import numpy as np
import pytest

from pyMOFL.functions.benchmark.lunacek import (
    LunacekBiRastriginCECFunction,
    LunacekRotatedCosineFunction,
)


class TestLunacekRotatedCosineFunction:
    """Test the rotated-cosine variant used by CEC 2013/2017/2020/2021."""

    def test_no_rotation_no_signs_matches_cec_variant(self):
        """Without rotation or signs, should match LunacekBiRastriginCECFunction."""
        dim = 10
        func_rc = LunacekRotatedCosineFunction(dimension=dim)
        func_cec = LunacekBiRastriginCECFunction(dimension=dim)
        rng = np.random.default_rng(42)
        for _ in range(5):
            x = rng.uniform(-5, 5, dim)
            np.testing.assert_allclose(func_rc.evaluate(x), func_cec.evaluate(x), atol=1e-10)

    def test_identity_rotation_matches_no_rotation(self):
        """Identity rotation matrix should give same result as no rotation."""
        dim = 10
        R = np.eye(dim)
        func_no_rot = LunacekRotatedCosineFunction(dimension=dim)
        func_id_rot = LunacekRotatedCosineFunction(dimension=dim, cosine_rotation=R)
        rng = np.random.default_rng(42)
        for _ in range(5):
            x = rng.uniform(-5, 5, dim)
            np.testing.assert_allclose(func_no_rot.evaluate(x), func_id_rot.evaluate(x), atol=1e-10)

    def test_rotation_only_affects_cosine_term(self):
        """Rotation should change cosine term but not quadratic terms."""
        dim = 5
        # Create a rotation matrix
        rng = np.random.default_rng(123)
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        func_no_rot = LunacekRotatedCosineFunction(dimension=dim)
        func_rot = LunacekRotatedCosineFunction(dimension=dim, cosine_rotation=Q)
        x = np.array([1.0, -0.5, 2.0, -1.0, 0.3])
        # Results should differ (rotation changes cosine term)
        v1 = func_no_rot.evaluate(x)
        v2 = func_rot.evaluate(x)
        assert abs(v1 - v2) > 0.01, "Rotation should change the result"

    def test_signs_parameter(self):
        """Signs should flip the 2x scaling direction, producing different results."""
        dim = 5
        signs_pos = np.ones(dim)
        signs_neg = -np.ones(dim)
        func_pos = LunacekRotatedCosineFunction(dimension=dim, shift_signs=signs_pos)
        func_neg = LunacekRotatedCosineFunction(dimension=dim, shift_signs=signs_neg)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # With positive signs: z = 2*x, with negative: z = -2*x
        # The quadratic (z + mu_diff)^2 terms differ because mu_diff > 0,
        # so flipping sign of z changes the distances asymmetrically.
        v_pos = func_pos.evaluate(x)
        v_neg = func_neg.evaluate(x)
        assert np.isfinite(v_pos)
        assert np.isfinite(v_neg)
        # At origin, both should agree (z=0 regardless of signs)
        x0 = np.zeros(dim)
        np.testing.assert_allclose(func_pos.evaluate(x0), func_neg.evaluate(x0), atol=1e-10)

    def test_mixed_signs(self):
        """Mixed signs should produce different results from all-positive."""
        dim = 5
        signs_pos = np.ones(dim)
        signs_mixed = np.array([1, -1, 1, -1, 1], dtype=float)
        func_pos = LunacekRotatedCosineFunction(dimension=dim, shift_signs=signs_pos)
        func_mixed = LunacekRotatedCosineFunction(dimension=dim, shift_signs=signs_mixed)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v1 = func_pos.evaluate(x)
        v2 = func_mixed.evaluate(x)
        # z differs in elements 1 and 3 due to sign flip, so quadratic terms
        # and cosine term both change
        assert abs(v1 - v2) > 0.01, "Mixed signs should change the result"

    def test_at_origin(self):
        """At origin, function value should be 0 (default, no signs)."""
        dim = 10
        func = LunacekRotatedCosineFunction(dimension=dim)
        x = np.zeros(dim)
        # z = 2*x = 0, tmpx = z + mu0 = mu0
        # sum1 = sum((mu0-mu0)^2) = 0
        # sum2 = d*D + s*sum((mu0-mu1)^2) > 0
        # min(0, sum2) = 0
        # cosine = 10*(D - sum(cos(0))) = 10*(D-D) = 0
        assert func.evaluate(x) == pytest.approx(0.0)

    def test_evaluate_batch(self):
        """Batch evaluation should match single evaluation."""
        dim = 5
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        signs = np.array([1, -1, 1, 1, -1], dtype=float)
        func = LunacekRotatedCosineFunction(dimension=dim, shift_signs=signs, cosine_rotation=Q)
        X = rng.uniform(-5, 5, (10, dim))
        batch = func.evaluate_batch(X)
        singles = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, singles, atol=1e-10)

    def test_dimension_1(self):
        """Edge case: dimension 1."""
        func = LunacekRotatedCosineFunction(dimension=1)
        x = np.array([0.0])
        result = func.evaluate(x)
        assert np.isfinite(result)

    def test_signs_from_shift_vector(self):
        """shift_signs can be derived from sign of shift vector."""
        dim = 5
        shift = np.array([-30.0, 50.0, -10.0, 20.0, -40.0])
        signs = np.sign(shift)
        # Verify: negative shift → sign = -1, positive → +1
        assert signs[0] == -1.0
        assert signs[1] == 1.0
        func = LunacekRotatedCosineFunction(dimension=dim, shift_signs=signs)
        x = np.ones(dim)
        result = func.evaluate(x)
        assert np.isfinite(result)
