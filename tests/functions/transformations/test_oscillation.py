"""Tests for OscillationTransform (T_osz from BBOB/Hansen et al. 2009)."""

import numpy as np
import pytest

from pyMOFL.functions.transformations.oscillation import OscillationTransform
from tests.utils.transform_validation import TransformValidator


class TestOscillationTransform:
    """Tests for OscillationTransform."""

    def test_vector_transform_contract_dim2(self):
        transform = OscillationTransform()
        TransformValidator.assert_vector_transform_contract(transform, dimension=2)

    def test_vector_transform_contract_dim10(self):
        transform = OscillationTransform()
        TransformValidator.assert_vector_transform_contract(transform, dimension=10)

    def test_zero_maps_to_zero(self):
        transform = OscillationTransform()
        result = transform(np.zeros(5))
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_zero_in_mixed_vector(self):
        transform = OscillationTransform()
        x = np.array([1.0, 0.0, -2.0, 0.0, 3.0])
        result = transform(x)
        # Zero elements must remain zero
        assert result[1] == 0.0
        assert result[3] == 0.0
        # Non-zero elements must be transformed (not necessarily different, but finite)
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])
        assert np.isfinite(result[4])

    def test_preserves_sign(self):
        transform = OscillationTransform()
        x = np.array([1.0, 2.0, 5.0, 0.1, 100.0])
        result = transform(x)
        assert np.all(result > 0), "Positive inputs should yield positive outputs"

        x_neg = np.array([-1.0, -2.0, -5.0, -0.1, -100.0])
        result_neg = transform(x_neg)
        assert np.all(result_neg < 0), "Negative inputs should yield negative outputs"

    def test_known_value_x_equals_one(self):
        """x=1: x_hat=log(1)=0, sin(0)=0, so T_osz(1) = sign(1)*exp(0+0) = 1.0."""
        transform = OscillationTransform()
        result = transform(np.array([1.0]))
        np.testing.assert_allclose(result[0], 1.0, rtol=1e-14)

    def test_known_value_x_equals_neg_one(self):
        """x=-1: x_hat=log(1)=0, sin(0)=0, so T_osz(-1) = -exp(0) = -1.0."""
        transform = OscillationTransform()
        result = transform(np.array([-1.0]))
        np.testing.assert_allclose(result[0], -1.0, rtol=1e-14)

    def test_known_value_x_equals_e(self):
        """x=e: x_hat=1, c1=10, c2=7.9 → exp(1 + 0.049*(sin(10)+sin(7.9)))."""
        transform = OscillationTransform()
        x = np.array([np.e])
        result = transform(x)
        x_hat = 1.0  # log(e) = 1
        expected = np.exp(x_hat + 0.049 * (np.sin(10.0 * x_hat) + np.sin(7.9 * x_hat)))
        np.testing.assert_allclose(result[0], expected, rtol=1e-14)

    def test_approximately_identity_for_moderate_values(self):
        """The 0.049 coefficient means oscillation amplitude is small."""
        transform = OscillationTransform()
        x = np.array([2.0, 5.0, 10.0])
        result = transform(x)
        # Result should be within ~10% of original for moderate values
        np.testing.assert_allclose(result, x, rtol=0.15)

    def test_very_small_positive_no_nan(self):
        transform = OscillationTransform()
        x = np.array([1e-300])
        result = transform(x)
        assert np.all(np.isfinite(result))

    def test_very_large_values_finite(self):
        transform = OscillationTransform()
        x = np.array([1e50])
        result = transform(x)
        assert np.all(np.isfinite(result))

    def test_batch_matches_single(self):
        transform = OscillationTransform()
        rng = np.random.default_rng(123)
        X = rng.uniform(-10, 10, size=(8, 4))
        batch_result = transform.transform_batch(X)
        for i in range(len(X)):
            single_result = transform(X[i])
            np.testing.assert_allclose(
                batch_result[i],
                single_result,
                rtol=1e-14,
                err_msg=f"Mismatch at row {i}",
            )

    def test_call_rejects_scalar(self):
        transform = OscillationTransform()
        with pytest.raises(ValueError, match="1D"):
            transform(np.asarray(np.float64(3.0)))

    def test_call_rejects_2d(self):
        transform = OscillationTransform()
        with pytest.raises(ValueError, match="1D"):
            transform(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_call_rejects_3d(self):
        transform = OscillationTransform()
        with pytest.raises(ValueError, match="1D"):
            transform(np.ones((2, 3, 4)))

    def test_call_accepts_empty_1d(self):
        transform = OscillationTransform()
        result = transform(np.array([]))
        assert result.shape == (0,)

    def test_batch_rejects_1d(self):
        transform = OscillationTransform()
        with pytest.raises(ValueError, match="2D"):
            transform.transform_batch(np.array([1.0, 2.0, 3.0]))

    def test_batch_rejects_3d(self):
        transform = OscillationTransform()
        with pytest.raises(ValueError, match="2D"):
            transform.transform_batch(np.ones((2, 3, 4)))
