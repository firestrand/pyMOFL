"""Tests for ConditioningTransform."""

from __future__ import annotations

import numpy as np
import pytest

from pyMOFL.functions.transformations.conditioning import ConditioningTransform


class TestConditioningTransform:
    def test_identity_for_alpha_1(self):
        t = ConditioningTransform(alpha=1.0, dimension=10)
        x = np.arange(10, dtype=np.float64)
        result = t(x)
        np.testing.assert_allclose(result, x)

    def test_first_element_unchanged(self):
        t = ConditioningTransform(alpha=100.0, dimension=10)
        x = np.ones(10, dtype=np.float64)
        result = t(x)
        assert result[0] == pytest.approx(1.0)

    def test_last_element_scaled(self):
        t = ConditioningTransform(alpha=100.0, dimension=10)
        x = np.ones(10, dtype=np.float64)
        result = t(x)
        # Last element: alpha^(9/(2*9)) = alpha^0.5 = 10.0
        assert result[9] == pytest.approx(10.0)

    def test_middle_element(self):
        t = ConditioningTransform(alpha=10.0, dimension=5)
        x = np.ones(5, dtype=np.float64)
        result = t(x)
        # i=2: alpha^(2/(2*4)) = 10^(2/8) = 10^0.25
        expected = 10.0**0.25
        assert result[2] == pytest.approx(expected)

    def test_batch(self):
        t = ConditioningTransform(alpha=10.0, dimension=3)
        X = np.ones((5, 3), dtype=np.float64)
        result = t.transform_batch(X)
        assert result.shape == (5, 3)
        np.testing.assert_allclose(result[0], result[1])

    def test_dimension_1(self):
        t = ConditioningTransform(alpha=100.0, dimension=1)
        x = np.array([5.0])
        result = t(x)
        assert result[0] == pytest.approx(5.0)
