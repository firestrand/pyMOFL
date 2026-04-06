"""Tests for BBOBConstraintGenerator."""

import numpy as np
import pytest

from pyMOFL.core.linear_constraint import LinearConstraint
from pyMOFL.utils.bbob_constraint_generator import BBOBConstraintGenerator


@pytest.fixture
def gen():
    return BBOBConstraintGenerator()


class TestBBOBConstraintGenerator:
    """Tests for BBOB constraint generation."""

    @pytest.mark.parametrize(
        "config,expected_active,dim",
        [
            (1, 1, 10),
            (2, 2, 10),
            (3, 6, 10),
            (4, 11, 10),  # 6 + D/2 = 6 + 5 = 11
            (5, 16, 10),  # 6 + D = 6 + 10 = 16
            (6, 36, 10),  # 6 + 3D = 6 + 30 = 36
        ],
    )
    def test_constraint_counts(self, gen, config, expected_active, dim):
        """Each config should produce correct number of active constraints."""
        constraints = gen.generate(fid=1, iid=1, dim=dim, config=config, xopt=np.zeros(dim))
        active = [c for c in constraints if c.is_active]
        inactive = [c for c in constraints if not c.is_active]
        assert len(active) == expected_active
        expected_inactive = expected_active // 2
        assert len(inactive) == expected_inactive

    def test_active_binding_at_optimum(self, gen):
        """Active constraints should be binding (g ≈ 0) at optimum."""
        dim = 5
        xopt = np.zeros(dim)
        constraints = gen.generate(fid=1, iid=1, dim=dim, config=1, xopt=xopt)
        for c in constraints:
            if c.is_active:
                val = c.evaluate(xopt)
                assert val == pytest.approx(0.0, abs=1e-10), (
                    f"Active constraint not binding at optimum: g={val}"
                )

    def test_inactive_feasible_at_optimum(self, gen):
        """Inactive constraints should be feasible (g < 0) at optimum."""
        dim = 10
        xopt = np.zeros(dim)
        constraints = gen.generate(fid=1, iid=1, dim=dim, config=2, xopt=xopt)
        for c in constraints:
            if not c.is_active:
                val = c.evaluate(xopt)
                assert val < 0, f"Inactive constraint not feasible at optimum: g={val}"

    def test_deterministic(self, gen):
        """Same inputs should produce same constraints."""
        xopt = np.zeros(5)
        c1 = gen.generate(fid=1, iid=1, dim=5, config=1, xopt=xopt)
        c2 = gen.generate(fid=1, iid=1, dim=5, config=1, xopt=xopt)
        for a, b in zip(c1, c2, strict=True):
            np.testing.assert_array_equal(a.normal, b.normal)
            assert a.shift == b.shift

    def test_different_fids_differ(self, gen):
        """Different function IDs should produce different constraints."""
        xopt = np.zeros(5)
        c1 = gen.generate(fid=1, iid=1, dim=5, config=1, xopt=xopt)
        c2 = gen.generate(fid=2, iid=1, dim=5, config=1, xopt=xopt)
        # At least normals should differ
        assert not np.allclose(c1[0].normal, c2[0].normal)

    def test_invalid_config_raises(self, gen):
        """Config outside 1-6 should raise."""
        with pytest.raises(ValueError, match="1-6"):
            gen.generate(fid=1, iid=1, dim=5, config=0, xopt=np.zeros(5))
        with pytest.raises(ValueError, match="1-6"):
            gen.generate(fid=1, iid=1, dim=5, config=7, xopt=np.zeros(5))

    def test_invalid_dimension_raises(self, gen):
        """Non-positive dimensions should be rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            gen.generate(fid=1, iid=1, dim=0, config=1, xopt=np.zeros(0))
        with pytest.raises(ValueError, match="must be positive"):
            gen.generate(fid=1, iid=1, dim=-3, config=1, xopt=np.zeros(1))

    def test_invalid_xopt_shape_raises(self, gen):
        """xopt shape must match dimension."""
        with pytest.raises(ValueError, match="xopt must have shape"):
            gen.generate(fid=1, iid=1, dim=5, config=1, xopt=np.zeros(4))

    def test_all_constraints_are_linear(self, gen):
        """All returned constraints should be LinearConstraint instances."""
        constraints = gen.generate(fid=1, iid=1, dim=5, config=3, xopt=np.zeros(5))
        for c in constraints:
            assert isinstance(c, LinearConstraint)

    def test_constraint_normals_finite(self, gen):
        """All constraint normals should be finite."""
        constraints = gen.generate(fid=1, iid=1, dim=10, config=6, xopt=np.zeros(10))
        for c in constraints:
            assert np.all(np.isfinite(c.normal))
