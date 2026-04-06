"""
Shared test utilities for validating VectorTransform and ScalarTransform subclasses.

Provides reusable assertion helpers that verify any transform satisfies
its ABC contract: __call__ returns correct type/shape, transform_batch
is consistent with element-wise calls.

Usage in test files::

    from tests.utils.transform_validation import TransformValidator

    class TestMyTransform:
        def test_contract(self):
            transform = MyVectorTransform(shift=np.zeros(5))
            TransformValidator.assert_vector_transform_contract(transform, dimension=5)
"""

import numpy as np

from pyMOFL.functions.transformations.base import ScalarTransform, VectorTransform


class TransformValidator:
    """Reusable assertions for VectorTransform and ScalarTransform subclass contracts."""

    @staticmethod
    def assert_vector_call_returns_ndarray(transform: VectorTransform, dimension: int) -> None:
        """Assert that __call__ returns an ndarray with the correct shape."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-1.0, 1.0, size=dimension)
        result = transform(x)
        assert isinstance(result, np.ndarray), (
            f"__call__ returned {type(result).__name__}, expected ndarray"
        )
        assert result.shape == (dimension,), (
            f"__call__ returned shape {result.shape}, expected ({dimension},)"
        )

    @staticmethod
    def assert_vector_batch_shape(
        transform: VectorTransform, dimension: int, batch_size: int = 5
    ) -> None:
        """Assert that transform_batch returns an array with shape (batch_size, dimension)."""
        rng = np.random.default_rng(42)
        X = rng.uniform(-1.0, 1.0, size=(batch_size, dimension))
        result = transform.transform_batch(X)
        assert isinstance(result, np.ndarray), (
            f"transform_batch returned {type(result).__name__}, expected ndarray"
        )
        assert result.shape == (batch_size, dimension), (
            f"transform_batch returned shape {result.shape}, expected ({batch_size}, {dimension})"
        )

    @staticmethod
    def assert_vector_batch_consistent(
        transform: VectorTransform, dimension: int, batch_size: int = 5
    ) -> None:
        """Assert that transform_batch is consistent with element-wise __call__."""
        rng = np.random.default_rng(42)
        X = rng.uniform(-1.0, 1.0, size=(batch_size, dimension))
        batch_result = transform.transform_batch(X)
        for i in range(batch_size):
            single_result = transform(X[i])
            np.testing.assert_allclose(
                batch_result[i],
                single_result,
                rtol=1e-12,
                err_msg=f"transform_batch[{i}] != __call__(X[{i}])",
            )

    @staticmethod
    def assert_vector_transform_contract(
        transform: VectorTransform,
        dimension: int,
        *,
        batch_size: int = 5,
    ) -> None:
        """Run the full contract validation suite on a VectorTransform instance.

        Parameters
        ----------
        transform : VectorTransform
            The transform instance to validate.
        dimension : int
            Expected input/output vector dimension.
        batch_size : int
            Number of vectors for batch tests.
        """
        TransformValidator.assert_vector_call_returns_ndarray(transform, dimension)
        TransformValidator.assert_vector_batch_shape(transform, dimension, batch_size)
        TransformValidator.assert_vector_batch_consistent(transform, dimension, batch_size)

    @staticmethod
    def assert_scalar_call_returns_float(transform: ScalarTransform) -> None:
        """Assert that __call__ returns a float."""
        result = transform(3.14)
        assert isinstance(result, (float, np.floating)), (
            f"__call__ returned {type(result).__name__}, expected float"
        )

    @staticmethod
    def assert_scalar_batch_shape(transform: ScalarTransform, batch_size: int = 5) -> None:
        """Assert that transform_batch returns an array with shape (batch_size,)."""
        rng = np.random.default_rng(42)
        Y = rng.uniform(-10.0, 10.0, size=batch_size)
        result = transform.transform_batch(Y)
        assert isinstance(result, np.ndarray), (
            f"transform_batch returned {type(result).__name__}, expected ndarray"
        )
        assert result.shape == (batch_size,), (
            f"transform_batch returned shape {result.shape}, expected ({batch_size},)"
        )

    @staticmethod
    def assert_scalar_batch_consistent(transform: ScalarTransform, batch_size: int = 5) -> None:
        """Assert that transform_batch is consistent with element-wise __call__."""
        rng = np.random.default_rng(42)
        Y = rng.uniform(-10.0, 10.0, size=batch_size)
        batch_result = transform.transform_batch(Y)
        for i in range(batch_size):
            single_result = transform(Y[i])
            np.testing.assert_allclose(
                batch_result[i],
                single_result,
                rtol=1e-12,
                err_msg=f"transform_batch[{i}] != __call__(Y[{i}])",
            )

    @staticmethod
    def assert_scalar_transform_contract(
        transform: ScalarTransform,
        *,
        batch_size: int = 5,
    ) -> None:
        """Run the full contract validation suite on a ScalarTransform instance.

        Parameters
        ----------
        transform : ScalarTransform
            The transform instance to validate.
        batch_size : int
            Number of scalars for batch tests.
        """
        TransformValidator.assert_scalar_call_returns_float(transform)
        TransformValidator.assert_scalar_batch_shape(transform, batch_size)
        TransformValidator.assert_scalar_batch_consistent(transform, batch_size)
