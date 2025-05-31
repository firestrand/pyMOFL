import numpy as np
import pytest
from pyMOFL.functions.unimodal.sphere import SphereFunction
from pyMOFL.decorators.matrix_transform import MatrixTransform

class TestMatrixTransform:
    def test_as_decorator(self):
        base = SphereFunction(dimension=2)
        matrix = np.eye(2)
        mt = MatrixTransform(base_function=base, matrix_data=matrix)
        x = np.array([1.0, 2.0])
        # Should be identical to base function
        assert mt(x) == base(x)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(mt.evaluate_batch(X), base.evaluate_batch(X))

    def test_property_delegation(self):
        base = SphereFunction(dimension=2)
        matrix = np.eye(2)
        mt = MatrixTransform(base_function=base, matrix_data=matrix)
        assert mt.dimension == base.dimension
        assert mt.initialization_bounds == base.initialization_bounds
        assert mt.operational_bounds == base.operational_bounds

    def test_error_on_missing_matrix(self):
        base = SphereFunction(dimension=2)
        with pytest.raises(ValueError):
            MatrixTransform(base_function=base, matrix_data=None)

    def test_error_on_invalid_matrix_shape(self):
        base = SphereFunction(dimension=2)
        matrix = np.eye(3)
        with pytest.raises(ValueError):
            MatrixTransform(base_function=base, matrix_data=matrix) 