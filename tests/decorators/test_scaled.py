import numpy as np
import pytest
from pyMOFL.functions.unimodal.sphere import SphereFunction
from pyMOFL.decorators.scaled import Scaled

class TestScaled:
    def test_as_decorator(self):
        base = SphereFunction(dimension=2)
        scaled = Scaled(base_function=base, lambda_coef=2.0)
        x = np.array([2.0, 4.0])
        assert np.allclose(scaled(x), base(x / 2.0))
        X = np.array([[2.0, 4.0], [4.0, 8.0]])
        np.testing.assert_allclose(scaled.evaluate_batch(X), base.evaluate_batch(X / 2.0))

    def test_as_base(self):
        scaled = Scaled(dimension=2, lambda_coef=2.0)
        x = np.array([2.0, 4.0])
        np.testing.assert_allclose(scaled._apply(x), x / 2.0)
        X = np.array([[2.0, 4.0], [4.0, 8.0]])
        np.testing.assert_allclose(scaled._apply_batch(X), X / 2.0)

    def test_vector_lambda(self):
        base = SphereFunction(dimension=2)
        lambda_vec = np.array([2.0, 4.0])
        scaled = Scaled(base_function=base, lambda_coef=lambda_vec)
        x = np.array([2.0, 4.0])
        assert np.allclose(scaled(x), base(x / lambda_vec))
        X = np.array([[2.0, 4.0], [4.0, 8.0]])
        np.testing.assert_allclose(scaled.evaluate_batch(X), base.evaluate_batch(X / lambda_vec))

    def test_property_delegation(self):
        base = SphereFunction(dimension=2)
        scaled = Scaled(base_function=base, lambda_coef=2.0)
        assert scaled.dimension == base.dimension
        assert scaled.initialization_bounds == base.initialization_bounds
        assert scaled.operational_bounds == base.operational_bounds

    def test_composability(self):
        base = SphereFunction(dimension=2)
        scaled1 = Scaled(base_function=base, lambda_coef=2.0)
        scaled2 = Scaled(base_function=scaled1, lambda_coef=2.0)
        x = np.array([8.0, 8.0])
        expected = base(x / 2.0 / 2.0)
        assert np.allclose(scaled2(x), expected)

    def test_error_on_invalid_lambda_dim(self):
        base = SphereFunction(dimension=2)
        lambda_vec = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            Scaled(base_function=base, lambda_coef=lambda_vec) 