import numpy as np
import pytest
from pyMOFL.core.bounds import Bounds
from pyMOFL.functions.unimodal.sphere import SphereFunction
from pyMOFL.decorators.boundary_adjusted_shift import BoundaryAdjustedShift
from pyMOFL.decorators.biased import Biased

class TestBoundaryAdjustedShift:
    def test_error_on_missing_shift_vector(self):
        with pytest.raises(ValueError):
            BoundaryAdjustedShift(base_function=SphereFunction(2), shift_vector=None)

    def test_boundary_adjustment(self):
        # Test for dimension 8 (divisible by 4)
        shift = np.arange(8)
        bas = BoundaryAdjustedShift(base_function=None, dimension=8, shift_vector=shift)
        # First 2 indices should be -100, last 2 should be 100
        assert np.all(bas.adjusted_shift[:2] == -100.0)
        assert np.all(bas.adjusted_shift[6:] == 100.0)
        # Middle should be unchanged except for the last index in the middle slice, which is set to 100.0
        np.testing.assert_allclose(bas.adjusted_shift[2:5], shift[2:5])
        assert bas.adjusted_shift[5] == 100.0

        # Test for dimension 7 (not divisible by 4)
        shift = np.arange(7)
        bas = BoundaryAdjustedShift(base_function=None, dimension=7, shift_vector=shift)
        # First 2 indices should be -100, last 2 should be 100
        assert np.all(bas.adjusted_shift[:2] == -100.0)
        assert np.all(bas.adjusted_shift[5:] == 100.0)
        # Middle should be unchanged except for the last index in the middle slice, which is set to 100.0
        np.testing.assert_allclose(bas.adjusted_shift[2:4], shift[2:4])
        assert bas.adjusted_shift[4] == 100.0

    def test_apply_and_apply_batch(self):
        shift = np.arange(4)
        bas = BoundaryAdjustedShift(base_function=None, dimension=4, shift_vector=shift)
        x = np.array([10.0, 20.0, 30.0, 40.0])
        expected = x - bas.adjusted_shift
        np.testing.assert_allclose(bas._apply(x), expected)
        X = np.stack([x, x+1])
        np.testing.assert_allclose(bas._apply_batch(X), X - bas.adjusted_shift)

    def test_decorator_on_function(self):
        shift = np.arange(4)
        base = SphereFunction(dimension=4)
        bas = BoundaryAdjustedShift(base_function=base, shift_vector=shift)
        x = np.array([10.0, 20.0, 30.0, 40.0])
        # Should evaluate at (x - adjusted_shift)
        expected = base(x - bas.adjusted_shift)
        assert np.allclose(bas(x), expected)
        # Batch
        X = np.stack([x, x+1])
        np.testing.assert_allclose(bas.evaluate_batch(X), base.evaluate_batch(X - bas.adjusted_shift))

    def test_property_delegation(self):
        shift = np.arange(3)
        base = SphereFunction(dimension=3)
        bas = BoundaryAdjustedShift(base_function=base, shift_vector=shift)
        assert bas.dimension == base.dimension
        assert bas.initialization_bounds == base.initialization_bounds
        assert bas.operational_bounds == base.operational_bounds

    def test_composability(self):
        shift = np.arange(2)
        base = SphereFunction(dimension=2)
        bas = BoundaryAdjustedShift(base_function=base, shift_vector=shift)
        biased = Biased(bas, bias=5.0)
        x = np.array([1.0, 2.0])
        # Should apply shift then bias
        expected = base(x - bas.adjusted_shift) + 5.0
        assert np.allclose(biased(x), expected)
        # Nesting
        bas2 = BoundaryAdjustedShift(base_function=biased, shift_vector=shift)
        expected2 = base(x - bas.adjusted_shift - bas2.adjusted_shift) + 5.0
        assert np.allclose(bas2(x), expected2) 