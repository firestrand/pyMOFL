"""Tests for BBOBLargeScaleSuiteFactory (24 large-scale BBOB functions)."""

import numpy as np
import pytest

from pyMOFL.factories.bbob_largescale_suite_factory import BBOBLargeScaleSuiteFactory
from pyMOFL.functions.transformations.block_diagonal_rotate import (
    BlockDiagonalRotateTransform,
)
from pyMOFL.functions.transformations.composed import ComposedFunction
from pyMOFL.functions.transformations.permutation import PermutationTransform


@pytest.fixture
def factory():
    return BBOBLargeScaleSuiteFactory()


class TestBBOBLargeScaleSuiteFactory:
    """Tests for the large-scale BBOB suite factory."""

    def test_dim40_matches_standard_bbob(self, factory):
        """At D=40 (= block_size), should behave identically to standard BBOB.

        For D <= 40, permutations are identity and blocks are full-size,
        so the result should be the same as standard BBOB.
        """
        from pyMOFL.factories.bbob_suite_factory import BBOBSuiteFactory

        std = BBOBSuiteFactory()
        rng = np.random.default_rng(42)
        x = rng.standard_normal(40)

        for fid in range(1, 25):
            std_func = std.create_function(fid, iid=1, dim=40)
            ls_func = factory.create_function(fid, iid=1, dim=40)
            std_val = std_func(x)
            ls_val = ls_func(x)
            assert std_val == pytest.approx(ls_val, abs=1e-8), (
                f"f{fid}: standard={std_val}, largescale={ls_val}"
            )

    def test_all_24_at_dim160(self, factory):
        """All 24 functions should instantiate at D=160."""
        rng = np.random.default_rng(42)
        for fid in range(1, 25):
            func = factory.create_function(fid, iid=1, dim=160)
            assert func is not None
            assert isinstance(func, ComposedFunction)
            x = rng.standard_normal(160)
            result = func(x)
            assert np.isfinite(result), f"f{fid} returned non-finite"

    def test_finite_evaluations(self, factory):
        """Functions should return finite values at various dimensions."""
        rng = np.random.default_rng(42)
        for dim in [80, 160]:
            func = factory.create_function(1, iid=1, dim=dim)
            x = rng.standard_normal(dim)
            assert np.isfinite(func(x))

    def test_invalid_fid_raises(self, factory):
        with pytest.raises(ValueError):
            factory.create_function(0, iid=1, dim=80)
        with pytest.raises(ValueError):
            factory.create_function(25, iid=1, dim=80)

    def test_create_suite(self, factory):
        """create_suite should return all 24 functions."""
        suite = factory.create_suite(iid=1, dim=80)
        assert len(suite) == 24

    def test_has_permutation_transforms_large_dim(self, factory):
        """For D > 40, should have permutation transforms."""
        func = factory.create_function(10, iid=1, dim=160)
        has_perm = any(isinstance(t, PermutationTransform) for t in func.input_transforms)
        assert has_perm, "Large-scale function should have PermutationTransform"

    def test_has_block_diagonal_rotate_large_dim(self, factory):
        """For D > 40, should have block-diagonal rotation."""
        func = factory.create_function(10, iid=1, dim=160)
        has_block = any(isinstance(t, BlockDiagonalRotateTransform) for t in func.input_transforms)
        assert has_block, "Large-scale function should have BlockDiagonalRotateTransform"

    def test_different_instances(self, factory):
        """Different instance IDs should produce different functions."""
        f1 = factory.create_function(1, iid=1, dim=80)
        f2 = factory.create_function(1, iid=2, dim=80)
        rng = np.random.default_rng(42)
        x = rng.standard_normal(80)
        assert np.isfinite(f1(x))
        assert np.isfinite(f2(x))
