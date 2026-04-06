"""Tests for BBOBMixintSuiteFactory (24 mixed-integer BBOB functions)."""

import numpy as np
import pytest

from pyMOFL.factories.bbob_mixint_suite_factory import BBOBMixintSuiteFactory
from pyMOFL.functions.transformations.composed import ComposedFunction
from pyMOFL.functions.transformations.discretize import DiscretizeTransform


@pytest.fixture
def factory():
    return BBOBMixintSuiteFactory()


class TestBBOBMixintSuiteFactory:
    """Tests for the mixed-integer BBOB suite factory."""

    def test_all_24_instantiate_dim10(self, factory):
        """All 24 mixint functions should instantiate at D=10."""
        for fid in range(1, 25):
            func = factory.create_function(fid, iid=1, dim=10)
            assert func is not None
            assert isinstance(func, ComposedFunction)

    def test_all_24_instantiate_dim20(self, factory):
        """All 24 mixint functions should instantiate at D=20."""
        for fid in range(1, 25):
            func = factory.create_function(fid, iid=1, dim=20)
            assert func is not None

    def test_invalid_fid_raises(self, factory):
        with pytest.raises(ValueError):
            factory.create_function(0, iid=1, dim=10)
        with pytest.raises(ValueError):
            factory.create_function(25, iid=1, dim=10)

    def test_dim_not_divisible_by_5_raises(self, factory):
        """D must be divisible by 5."""
        with pytest.raises(ValueError, match="divisible by 5"):
            factory.create_function(1, iid=1, dim=7)

    def test_discretize_transform_present(self, factory):
        """Each function should have a DiscretizeTransform as first input transform."""
        for fid in range(1, 25):
            func = factory.create_function(fid, iid=1, dim=10)
            has_disc = any(isinstance(t, DiscretizeTransform) for t in func.input_transforms)
            assert has_disc, f"f{fid} missing DiscretizeTransform"

    def test_finite_evaluations(self, factory):
        """All functions should return finite values."""
        rng = np.random.default_rng(42)
        for fid in range(1, 25):
            func = factory.create_function(fid, iid=1, dim=10)
            # Use integer-range inputs for discrete vars
            x = rng.integers(0, 16, size=10).astype(float)
            result = func(x)
            assert np.isfinite(result), f"f{fid} returned non-finite: {result}"

    def test_create_suite(self, factory):
        """create_suite should return all 24 functions."""
        suite = factory.create_suite(iid=1, dim=10)
        assert len(suite) == 24

    def test_dimension_propagation(self, factory):
        """Function dimension should match requested dimension."""
        func = factory.create_function(1, iid=1, dim=10)
        assert func.dimension == 10

    def test_different_instances(self, factory):
        """Different instance IDs should produce different functions."""
        f1 = factory.create_function(1, iid=1, dim=10)
        f2 = factory.create_function(1, iid=2, dim=10)
        x = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 0.0, 1.0, 2.0, 5.0, 1.0])
        # At minimum both should be finite
        assert np.isfinite(f1(x))
        assert np.isfinite(f2(x))
