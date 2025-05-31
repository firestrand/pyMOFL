import numpy as np
import pytest
from pyMOFL.functions.unimodal.sphere import SphereFunction
from pyMOFL.decorators.rotated import Rotated

class TestRotated:
    def test_as_decorator(self):
        base = SphereFunction(dimension=2)
        rotation = np.array([[0, 1], [1, 0]])
        rotated = Rotated(base_function=base, rotation_matrix=rotation)
        x = np.array([1.0, 2.0])
        assert np.allclose(rotated(x), base(np.dot(rotation, x)))
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(rotated.evaluate_batch(X), base.evaluate_batch(np.dot(X, rotation.T)))

    def test_as_base(self):
        rotation = np.eye(2)
        rotated = Rotated(dimension=2, rotation_matrix=rotation)
        x = np.array([1.0, 2.0])
        np.testing.assert_allclose(rotated._apply(x), np.dot(rotation, x))
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(rotated._apply_batch(X), np.dot(X, rotation.T))

    def test_property_delegation(self):
        base = SphereFunction(dimension=2)
        rotation = np.eye(2)
        rotated = Rotated(base_function=base, rotation_matrix=rotation)
        assert rotated.dimension == base.dimension
        assert rotated.initialization_bounds == base.initialization_bounds
        assert rotated.operational_bounds == base.operational_bounds

    def test_composability(self):
        base = SphereFunction(dimension=2)
        rotation1 = np.array([[0, 1], [1, 0]])
        rotation2 = np.eye(2)
        rotated1 = Rotated(base_function=base, rotation_matrix=rotation1)
        rotated2 = Rotated(base_function=rotated1, rotation_matrix=rotation2)
        x = np.array([1.0, 2.0])
        expected = base(np.dot(rotation1, np.dot(rotation2, x)))
        assert np.allclose(rotated2(x), expected)

    def test_error_on_missing_rotation(self):
        base = SphereFunction(dimension=2)
        with pytest.raises(ValueError):
            Rotated(base_function=base, rotation_matrix=None)

    def test_error_on_invalid_rotation_shape(self):
        base = SphereFunction(dimension=2)
        rotation = np.eye(3)
        with pytest.raises(ValueError):
            Rotated(base_function=base, rotation_matrix=rotation) 