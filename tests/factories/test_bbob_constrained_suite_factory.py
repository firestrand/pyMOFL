"""Tests for BBOBConstrainedSuiteFactory (54 constrained BBOB functions)."""

import numpy as np
import pytest

from pyMOFL.core.constrained_function import ConstrainedFunction
from pyMOFL.factories.bbob_constrained_suite_factory import (
    BBOBConstrainedSuiteFactory,
)


@pytest.fixture
def factory():
    return BBOBConstrainedSuiteFactory()


class TestBBOBConstrainedSuiteFactory:
    """Tests for the constrained BBOB suite factory."""

    def test_all_54_instantiate(self, factory):
        """All 54 constrained functions should instantiate."""
        funcs = factory.create_suite(iid=1, dim=10)
        assert len(funcs) == 54
        for f in funcs:
            assert isinstance(f, ConstrainedFunction)

    def test_finite_evaluations(self, factory):
        """All functions should return finite objective values."""
        rng = np.random.default_rng(42)
        funcs = factory.create_suite(iid=1, dim=5)
        for i, f in enumerate(funcs):
            x = rng.standard_normal(5)
            result = f(x)
            assert np.isfinite(result), f"Function {i} returned non-finite: {result}"

    def test_constraints_evaluable(self, factory):
        """All functions should have evaluable constraints."""
        rng = np.random.default_rng(42)
        funcs = factory.create_suite(iid=1, dim=5)
        for i, f in enumerate(funcs):
            x = rng.standard_normal(5)
            g = f.evaluate_constraints(x)
            assert np.all(np.isfinite(g)), f"Function {i} has non-finite constraints"
            assert g.shape[0] == f.num_constraints

    def test_feasible_at_optimum(self, factory):
        """All functions should be feasible at the optimum."""
        funcs = factory.create_suite(iid=1, dim=5)
        for i, f in enumerate(funcs):
            # The optimum is at xopt (which is the base BBOB function's optimum)
            # Active constraints should be binding, inactive should be feasible
            xopt = f.xopt
            assert f.is_feasible(xopt), (
                f"Function {i} not feasible at optimum. Violations: {f.violations(xopt)}"
            )

    def test_correct_base_types(self, factory):
        """Should use correct objective function types."""
        info = factory.get_function_info()
        # 9 objectives × 6 configs = 54
        assert len(info) == 54
        # Check objective types are as expected
        obj_fids = {item["obj_fid"] for item in info}
        assert obj_fids == {1, 2, 3, 5, 10, 11, 12, 14, 15}

    def test_create_function_by_index(self, factory):
        """Should be able to create individual functions by index."""
        f = factory.create_function(obj_idx=0, config=1, iid=1, dim=5)
        assert isinstance(f, ConstrainedFunction)

    def test_invalid_config_raises(self, factory):
        """Invalid config should raise."""
        with pytest.raises(ValueError):
            factory.create_function(obj_idx=0, config=0, iid=1, dim=5)

    def test_invalid_obj_idx_raises(self, factory):
        """Invalid objective index should raise."""
        with pytest.raises(ValueError):
            factory.create_function(obj_idx=9, config=1, iid=1, dim=5)

    def test_different_configs_different_constraints(self, factory):
        """Different configs should produce different numbers of constraints."""
        f1 = factory.create_function(obj_idx=0, config=1, iid=1, dim=10)
        f6 = factory.create_function(obj_idx=0, config=6, iid=1, dim=10)
        assert f1.num_constraints < f6.num_constraints

    def test_different_dimensions(self, factory):
        """Should work with different dimensions."""
        for dim in [2, 5, 10, 20]:
            f = factory.create_function(obj_idx=0, config=1, iid=1, dim=dim)
            assert f.dimension == dim
            x = np.zeros(dim)
            assert np.isfinite(f(x))
