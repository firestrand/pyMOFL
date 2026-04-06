"""
Smoke tests for the TransformValidator utility.

Exercises the validator against existing transform classes to verify
the utility itself works correctly.
"""

import numpy as np

from pyMOFL.functions.transformations.bias import BiasTransform
from pyMOFL.functions.transformations.scale import ScaleTransform
from pyMOFL.functions.transformations.shift import ShiftTransform
from tests.utils.transform_validation import TransformValidator


class TestTransformValidatorWithShift:
    """Verify TransformValidator works against ShiftTransform (VectorTransform)."""

    def test_full_contract_dim3(self):
        shift = np.array([1.0, 2.0, 3.0])
        transform = ShiftTransform(shift=shift)
        TransformValidator.assert_vector_transform_contract(transform, dimension=3)

    def test_full_contract_dim10(self):
        shift = np.zeros(10)
        transform = ShiftTransform(shift=shift)
        TransformValidator.assert_vector_transform_contract(transform, dimension=10)


class TestTransformValidatorWithScale:
    """Verify TransformValidator works against ScaleTransform (VectorTransform)."""

    def test_full_contract(self):
        transform = ScaleTransform(factor=2.0)
        TransformValidator.assert_vector_transform_contract(transform, dimension=5)


class TestTransformValidatorWithBias:
    """Verify TransformValidator works against BiasTransform (ScalarTransform)."""

    def test_full_contract(self):
        transform = BiasTransform(bias=100.0)
        TransformValidator.assert_scalar_transform_contract(transform)


class TestTransformValidatorIndividualAssertions:
    """Test individual assertion methods for targeted usage."""

    def test_vector_call_returns_ndarray(self):
        shift = np.array([1.0, 2.0])
        transform = ShiftTransform(shift=shift)
        TransformValidator.assert_vector_call_returns_ndarray(transform, dimension=2)

    def test_vector_batch_shape(self):
        shift = np.array([1.0, 2.0])
        transform = ShiftTransform(shift=shift)
        TransformValidator.assert_vector_batch_shape(transform, dimension=2, batch_size=7)

    def test_vector_batch_consistent(self):
        shift = np.array([1.0, 2.0])
        transform = ShiftTransform(shift=shift)
        TransformValidator.assert_vector_batch_consistent(transform, dimension=2, batch_size=7)

    def test_scalar_call_returns_float(self):
        transform = BiasTransform(bias=5.0)
        TransformValidator.assert_scalar_call_returns_float(transform)

    def test_scalar_batch_shape(self):
        transform = BiasTransform(bias=5.0)
        TransformValidator.assert_scalar_batch_shape(transform, batch_size=7)

    def test_scalar_batch_consistent(self):
        transform = BiasTransform(bias=5.0)
        TransformValidator.assert_scalar_batch_consistent(transform, batch_size=7)
