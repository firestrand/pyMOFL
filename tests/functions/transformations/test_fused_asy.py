"""Tests for FusedBufferAliasAsymmetricTransform."""

from __future__ import annotations

import numpy as np
import pytest

from pyMOFL.functions.transformations.asymmetric import AsymmetricTransform
from pyMOFL.functions.transformations.fused_asy import FusedBufferAliasAsymmetricTransform
from pyMOFL.functions.transformations.oscillation import OscillationTransform
from pyMOFL.functions.transformations.rotate import RotateTransform


class TestFusedBufferAliasAsymmetricTransform:
    def test_all_positive_matches_standard_asy(self):
        """When inner output is all positive, result should match standard asy."""
        dim = 5
        inner = OscillationTransform(boundary_only=False)
        fused = FusedBufferAliasAsymmetricTransform(inner, beta=0.5, dimension=dim)
        asy = AsymmetricTransform(beta=0.5, dimension=dim)

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        inner_result = inner(x)
        # All positive since input is positive
        assert np.all(inner_result > 0)
        expected = asy(inner_result)
        actual = fused(x)
        np.testing.assert_allclose(actual, expected)

    def test_negative_elements_use_fallback(self):
        """Negative inner results should use pre-inner (fallback) values."""
        dim = 4
        # Use identity as inner so we can control exactly
        # Instead, use a rotation that makes some elements negative
        mat = np.eye(dim, dtype=np.float64)
        mat[0, 0] = -1.0  # Flip sign of first element
        inner = RotateTransform(mat)
        fused = FusedBufferAliasAsymmetricTransform(inner, beta=0.5, dimension=dim)

        x = np.array([3.0, 2.0, 1.0, 0.5])
        inner_result = inner(x)
        assert inner_result[0] < 0  # First element flipped negative

        result = fused(x)
        # Element 0: inner_result[0] = -3.0 <= 0, so result[0] = x[0] = 3.0 (fallback)
        assert result[0] == pytest.approx(3.0)
        # Other elements: inner_result[i] > 0, so result[i] = asy(inner_result[i])
        asy = AsymmetricTransform(beta=0.5, dimension=dim)
        asy_result = asy(inner_result)
        np.testing.assert_allclose(result[1:], asy_result[1:])

    def test_zero_elements_use_fallback(self):
        """Zero inner results should also use fallback (not positive)."""
        dim = 3
        mat = np.eye(dim, dtype=np.float64)
        mat[1, 1] = 0.0  # Zero out second element

        # Create a custom inner that zeros element 1
        class ZeroSecond:
            def __call__(self, x):
                r = x.copy()
                r[1] = 0.0
                return r

            def transform_batch(self, X):
                r = X.copy()
                r[:, 1] = 0.0
                return r

        fused = FusedBufferAliasAsymmetricTransform(ZeroSecond(), beta=0.5, dimension=dim)
        x = np.array([2.0, 5.0, 3.0])
        result = fused(x)
        # Element 1: inner gives 0, so fallback to x[1] = 5.0
        assert result[1] == pytest.approx(5.0)

    def test_batch(self):
        """Batch processing should work correctly."""
        dim = 3
        inner = OscillationTransform(boundary_only=True)
        fused = FusedBufferAliasAsymmetricTransform(inner, beta=0.2, dimension=dim)

        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = fused.transform_batch(X)
        assert result.shape == (2, 3)
        # Each row should match single-vector result
        for i in range(2):
            np.testing.assert_allclose(result[i], fused(X[i]))

    def test_dimension_1(self):
        """Dimension 1 should work (edge case for ratios)."""
        inner = OscillationTransform()
        fused = FusedBufferAliasAsymmetricTransform(inner, beta=0.5, dimension=1)
        x = np.array([2.0])
        result = fused(x)
        # ratio = 0, so exponent = 1, so result = inner(x)
        np.testing.assert_allclose(result, inner(x))
