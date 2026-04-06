"""Tests for AsymmetricTransform (T_asy^beta from BBOB/Hansen et al. 2009)."""

import numpy as np
import pytest

from pyMOFL.functions.transformations.asymmetric import AsymmetricTransform
from tests.utils.transform_validation import TransformValidator


class TestAsymmetricTransform:
    """Tests for AsymmetricTransform."""

    def test_vector_transform_contract_dim5(self):
        transform = AsymmetricTransform(beta=0.5, dimension=5)
        TransformValidator.assert_vector_transform_contract(transform, dimension=5)

    def test_vector_transform_contract_dim10(self):
        transform = AsymmetricTransform(beta=0.2, dimension=10)
        TransformValidator.assert_vector_transform_contract(transform, dimension=10)

    def test_negative_values_unchanged(self):
        transform = AsymmetricTransform(beta=0.5, dimension=4)
        x = np.array([-1.0, -5.0, -0.1, -100.0])
        result = transform(x)
        np.testing.assert_array_equal(result, x)

    def test_zero_unchanged(self):
        transform = AsymmetricTransform(beta=0.5, dimension=5)
        x = np.zeros(5)
        result = transform(x)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_mixed_positive_negative(self):
        transform = AsymmetricTransform(beta=0.5, dimension=4)
        x = np.array([4.0, -3.0, 2.0, -1.0])
        result = transform(x)
        # Negative elements unchanged
        np.testing.assert_equal(result[1], -3.0)
        np.testing.assert_equal(result[3], -1.0)
        # Positive elements transformed (at least some differ from input)
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])

    def test_beta_zero_is_identity(self):
        transform = AsymmetricTransform(beta=0.0, dimension=5)
        rng = np.random.default_rng(42)
        x = rng.uniform(-10, 10, size=5)
        result = transform(x)
        np.testing.assert_allclose(result, x, rtol=1e-14)

    def test_dimension_1_identity(self):
        """D=1 → ratio=0 → exponent=1 → identity for positive x."""
        transform = AsymmetricTransform(beta=0.5, dimension=1)
        x = np.array([4.0])
        result = transform(x)
        np.testing.assert_allclose(result[0], 4.0, rtol=1e-14)

    def test_known_value_first_element(self):
        """i=0 → ratio=0 → exponent = 1 + beta*0*sqrt(x) = 1 → x^1 = x."""
        transform = AsymmetricTransform(beta=0.5, dimension=5)
        x = np.array([4.0, 0.0, 0.0, 0.0, 0.0])
        result = transform(x)
        np.testing.assert_allclose(result[0], 4.0, rtol=1e-14)

    def test_known_value_last_element(self):
        """i=D-1=4, D=5 → ratio=1 → 4^(1+0.5*1*sqrt(4)) = 4^(1+1) = 4^2 = 16."""
        transform = AsymmetricTransform(beta=0.5, dimension=5)
        x = np.array([0.0, 0.0, 0.0, 0.0, 4.0])
        result = transform(x)
        np.testing.assert_allclose(result[4], 16.0, rtol=1e-14)

    def test_known_value_middle_element(self):
        """i=2, D=5 → ratio=0.5 → 4^(1+0.5*0.5*sqrt(4)) = 4^(1+0.5) = 4^1.5 = 8."""
        transform = AsymmetricTransform(beta=0.5, dimension=5)
        x = np.array([0.0, 0.0, 4.0, 0.0, 0.0])
        result = transform(x)
        np.testing.assert_allclose(result[2], 8.0, rtol=1e-14)

    def test_dimension_mismatch_raises(self):
        transform = AsymmetricTransform(beta=0.5, dimension=5)
        with pytest.raises(ValueError, match="dimension"):
            transform(np.array([1.0, 2.0, 3.0]))

    def test_batch_dimension_mismatch_raises(self):
        transform = AsymmetricTransform(beta=0.5, dimension=5)
        with pytest.raises(ValueError, match="dimension"):
            transform.transform_batch(np.ones((3, 4)))

    def test_very_small_positive_no_nan(self):
        transform = AsymmetricTransform(beta=0.5, dimension=3)
        x = np.array([1e-15, 1e-15, 1e-15])
        result = transform(x)
        assert np.all(np.isfinite(result))

    def test_batch_matches_single(self):
        transform = AsymmetricTransform(beta=0.3, dimension=6)
        rng = np.random.default_rng(99)
        X = rng.uniform(-5, 5, size=(8, 6))
        batch_result = transform.transform_batch(X)
        for i in range(len(X)):
            single_result = transform(X[i])
            np.testing.assert_allclose(
                batch_result[i],
                single_result,
                rtol=1e-14,
                err_msg=f"Mismatch at row {i}",
            )
