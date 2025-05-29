"""
Tests for the MatrixTransformFunction (refactored for new bounds logic).
"""

import pytest
import numpy as np
from pyMOFL.functions.unimodal.sphere import SphereFunction
from pyMOFL.decorators.matrix_transform import MatrixTransformFunction
from pyMOFL.decorators.biased import BiasedFunction

class TestMatrixTransformFunction:
    def test_initialization_and_delegation(self):
        func = SphereFunction(dimension=2)
        matrix = np.eye(2)
        mt_func = MatrixTransformFunction(func, matrix)
        assert mt_func.dimension == 2
        assert mt_func.initialization_bounds == func.initialization_bounds
        assert mt_func.operational_bounds == func.operational_bounds

    def test_matrix_application_identity(self):
        func = SphereFunction(dimension=2)
        matrix = np.eye(2)
        mt_func = MatrixTransformFunction(func, matrix)
        x = np.array([1.0, 2.0])
        # Should be identical to base function
        assert np.isclose(mt_func.evaluate(x), func.evaluate(x))

    def test_matrix_application_nontrivial(self):
        func = SphereFunction(dimension=2)
        matrix = np.array([[2.0, 0.0], [0.0, 0.5]])
        mt_func = MatrixTransformFunction(func, matrix)
        # The decorator does not apply the matrix, but sets it in the base if supported
        # For SphereFunction, this is a no-op, so output should match base
        x = np.array([1.0, 2.0])
        assert np.isclose(mt_func.evaluate(x), func.evaluate(x))

    def test_evaluate_batch(self):
        func = SphereFunction(dimension=2)
        matrix = np.eye(2)
        mt_func = MatrixTransformFunction(func, matrix)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = func.evaluate_batch(X)
        result = mt_func.evaluate_batch(X)
        np.testing.assert_allclose(result, expected)

    def test_composability(self):
        func = SphereFunction(dimension=2)
        matrix = np.eye(2)
        biased = BiasedFunction(func, bias=5.0)
        mt_func = MatrixTransformFunction(biased, matrix)
        x = np.array([1.0, 2.0])
        assert np.isclose(mt_func.evaluate(x), biased.evaluate(x))

    def test_error_on_dimension_mismatch(self):
        func = SphereFunction(dimension=2)
        matrix = np.eye(3)
        with pytest.raises(ValueError):
            MatrixTransformFunction(func, matrix) 