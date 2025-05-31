import numpy as np
import pytest
from pyMOFL.functions.unimodal.sphere import SphereFunction
from pyMOFL.decorators.shifted import Shifted

class TestShifted:
    def test_as_decorator(self):
        base = SphereFunction(dimension=2)
        shift = np.array([1.0, 2.0])
        shifted = Shifted(base_function=base, shift=shift)
        x = np.array([2.0, 4.0])
        assert np.allclose(shifted(x), base(x - shift))
        X = np.array([[2.0, 4.0], [4.0, 8.0]])
        np.testing.assert_allclose(shifted.evaluate_batch(X), base.evaluate_batch(X - shift))

    def test_as_base(self):
        shift = np.array([1.0, 2.0])
        shifted = Shifted(dimension=2, shift=shift)
        x = np.array([2.0, 4.0])
        # As base, just returns x - shift
        np.testing.assert_allclose(shifted._apply(x), x - shift)
        X = np.array([[2.0, 4.0], [4.0, 8.0]])
        np.testing.assert_allclose(shifted._apply_batch(X), X - shift)

    def test_property_delegation(self):
        base = SphereFunction(dimension=2)
        shift = np.array([1.0, 2.0])
        shifted = Shifted(base_function=base, shift=shift)
        assert shifted.dimension == base.dimension
        assert shifted.initialization_bounds == base.initialization_bounds
        assert shifted.operational_bounds == base.operational_bounds

    def test_composability(self):
        base = SphereFunction(dimension=2)
        shift1 = np.array([1.0, 2.0])
        shift2 = np.array([0.5, 0.5])
        shifted1 = Shifted(base_function=base, shift=shift1)
        shifted2 = Shifted(base_function=shifted1, shift=shift2)
        x = np.array([2.0, 4.0])
        expected = base(x - shift2 - shift1)
        assert np.allclose(shifted2(x), expected)

    def test_error_on_missing_shift(self):
        base = SphereFunction(dimension=2)
        with pytest.raises(ValueError):
            Shifted(base_function=base, shift=None)

    def test_error_on_invalid_shift_dim(self):
        base = SphereFunction(dimension=2)
        shift = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            Shifted(base_function=base, shift=shift) 