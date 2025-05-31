import numpy as np
import pytest
from pyMOFL.functions.unimodal.sphere import SphereFunction
from pyMOFL.decorators.biased import Biased

class TestBiased:
    def test_as_decorator(self):
        base = SphereFunction(dimension=2)
        biased = Biased(base_function=base, bias=5.0)
        x = np.array([1.0, 2.0])
        assert biased(x) == base(x) + 5.0
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(biased.evaluate_batch(X), base.evaluate_batch(X) + 5.0)

    def test_as_base(self):
        biased = Biased(dimension=2, bias=7.0)
        x = np.array([1.0, 2.0])
        np.testing.assert_allclose(biased(x), x + 7.0)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(biased.evaluate_batch(X), X + 7.0)

    def test_property_delegation(self):
        base = SphereFunction(dimension=2)
        biased = Biased(base_function=base, bias=2.0)
        assert biased.dimension == base.dimension
        assert biased.initialization_bounds == base.initialization_bounds
        assert biased.operational_bounds == base.operational_bounds

    def test_composability(self):
        base = SphereFunction(dimension=2)
        biased1 = Biased(base_function=base, bias=2.0)
        biased2 = Biased(base_function=biased1, bias=3.0)
        x = np.array([1.0, 2.0])
        assert biased2(x) == base(x) + 2.0 + 3.0

    def test_batch_composability(self):
        base = SphereFunction(dimension=2)
        biased1 = Biased(base_function=base, bias=2.0)
        biased2 = Biased(base_function=biased1, bias=3.0)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(biased2.evaluate_batch(X), base.evaluate_batch(X) + 2.0 + 3.0) 