"""Tests for StepHalfTransform."""

from __future__ import annotations

import numpy as np

from pyMOFL.functions.transformations.step_half import StepHalfTransform


class TestStepHalfTransform:
    def test_small_values_unchanged(self):
        t = StepHalfTransform()
        x = np.array([0.1, -0.3, 0.0, 0.49])
        result = t(x)
        np.testing.assert_allclose(result, x)

    def test_exact_half_unchanged(self):
        t = StepHalfTransform()
        x = np.array([0.5, -0.5])
        result = t(x)
        np.testing.assert_allclose(result, x)

    def test_rounds_to_nearest_half(self):
        t = StepHalfTransform()
        x = np.array([0.6, 0.7, 0.8, 1.0, 1.3])
        result = t(x)
        expected = np.array([0.5, 0.5, 1.0, 1.0, 1.5])
        np.testing.assert_allclose(result, expected)

    def test_negative_values(self):
        t = StepHalfTransform()
        x = np.array([-0.6, -0.7, -1.0, -1.3])
        result = t(x)
        expected = np.array([-0.5, -0.5, -1.0, -1.5])
        np.testing.assert_allclose(result, expected)

    def test_batch(self):
        t = StepHalfTransform()
        X = np.array([[0.1, 0.9], [-0.1, -0.9]])
        result = t.transform_batch(X)
        expected = np.array([[0.1, 1.0], [-0.1, -1.0]])
        np.testing.assert_allclose(result, expected)
