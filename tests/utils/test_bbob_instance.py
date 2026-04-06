"""
Tests for BBOB instance parameter generation.

Validates COCO-compatible seeded, deterministic generation of:
- x_opt (optimal point)
- f_opt (optimal value)
- Rotation matrices
- Lambda conditioning vectors
"""

import numpy as np


class TestBBOBInstanceGenerator:
    """Tests for BBOBInstanceGenerator (COCO-compatible)."""

    def test_generate_xopt_shape_and_bounds(self):
        """x_opt should be in [-4, 4]^D for standard functions."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        for dim in [2, 10, 20]:
            xopt = gen.generate_xopt(fid=1, iid=1, dim=dim)
            assert xopt.shape == (dim,)
            assert np.all(xopt >= -4.0)
            assert np.all(xopt <= 4.0)

    def test_generate_xopt_reproducible(self):
        """Same (fid, iid, dim) should always produce the same x_opt."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        x1 = gen.generate_xopt(fid=3, iid=5, dim=10)
        x2 = gen.generate_xopt(fid=3, iid=5, dim=10)
        np.testing.assert_array_equal(x1, x2)

    def test_generate_xopt_distinct_instances(self):
        """Different instances should produce different x_opt."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        x1 = gen.generate_xopt(fid=1, iid=1, dim=10)
        x2 = gen.generate_xopt(fid=1, iid=2, dim=10)
        assert not np.array_equal(x1, x2)

    def test_generate_xopt_distinct_functions(self):
        """Different functions should produce different x_opt."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        x1 = gen.generate_xopt(fid=1, iid=1, dim=10)
        x2 = gen.generate_xopt(fid=2, iid=1, dim=10)
        assert not np.array_equal(x1, x2)

    def test_generate_xopt_f5_boundary(self):
        """f5 (linear slope) has x_opt at boundary ±5."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        xopt = gen.generate_xopt(fid=5, iid=1, dim=10)
        assert np.all(np.abs(xopt) == 5.0)

    def test_generate_xopt_f8_scaled(self):
        """f8 (Rosenbrock) has x_opt scaled by 0.75."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        xopt = gen.generate_xopt(fid=8, iid=1, dim=10)
        # All values should be within [-3, 3] (0.75 * [-4, 4])
        assert np.all(np.abs(xopt) <= 3.01)

    def test_generate_xopt_no_exact_zeros(self):
        """COCO replaces exact zeros with -1e-5."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        # Check many instances — zeros should be replaced
        for fid in [1, 2, 3]:
            for iid in range(1, 10):
                xopt = gen.generate_xopt(fid=fid, iid=iid, dim=20)
                assert not np.any(xopt == 0.0), f"fid={fid}, iid={iid}: xopt contains exact zero"

    def test_generate_fopt(self):
        """f_opt should be finite and deterministic."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        f1 = gen.generate_fopt(fid=1, iid=1)
        f2 = gen.generate_fopt(fid=1, iid=1)
        assert np.isfinite(f1)
        assert f1 == f2

    def test_generate_fopt_distinct(self):
        """Different (fid, iid) should produce different f_opt."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        f1 = gen.generate_fopt(fid=1, iid=1)
        f2 = gen.generate_fopt(fid=1, iid=2)
        assert f1 != f2

    def test_generate_fopt_bounded(self):
        """f_opt should be bounded to [-1000, 1000]."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        for fid in range(1, 25):
            for iid in range(1, 6):
                f = gen.generate_fopt(fid=fid, iid=iid)
                assert -1000 <= f <= 1000, f"f_opt out of range for fid={fid}, iid={iid}: {f}"

    def test_generate_fopt_f4_uses_base_3(self):
        """f4 should use base seed 3 (matching COCO)."""
        from pyMOFL.utils.bbob_instance import _compute_rseed

        assert _compute_rseed(4, 1) == 3 + 10000

    def test_generate_fopt_f18_uses_base_17(self):
        """f18 should use base seed 17 (matching COCO)."""
        from pyMOFL.utils.bbob_instance import _compute_rseed

        assert _compute_rseed(18, 1) == 17 + 10000

    def test_generate_rotation_orthogonal(self):
        """Rotation matrix should be orthogonal: R @ R.T = I."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        for dim in [2, 5, 10]:
            R = gen.generate_rotation(dim=dim, seed=42)
            assert R.shape == (dim, dim)
            np.testing.assert_allclose(R @ R.T, np.eye(dim), atol=1e-12)

    def test_generate_rotation_reproducible(self):
        """Same seed should produce same rotation."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        R1 = gen.generate_rotation(dim=5, seed=42)
        R2 = gen.generate_rotation(dim=5, seed=42)
        np.testing.assert_array_equal(R1, R2)

    def test_generate_rotation_distinct_seeds(self):
        """Different seeds should produce different rotations."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        R1 = gen.generate_rotation(dim=5, seed=42)
        R2 = gen.generate_rotation(dim=5, seed=99)
        assert not np.array_equal(R1, R2)

    def test_generate_rotation_dim1(self):
        """Dimension 1 rotation should be identity."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        R = gen.generate_rotation(dim=1, seed=42)
        np.testing.assert_array_equal(R, np.array([[1.0]]))

    def test_generate_lambda(self):
        """Lambda vector should have correct conditioning."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        lam = gen.generate_lambda(alpha=10.0, dim=5)
        assert lam.shape == (5,)
        # First element should be 1.0 (10^0)
        np.testing.assert_allclose(lam[0], 1.0, atol=1e-12)
        # Last element should be 10^(alpha/2) = 10^5 = 100000
        np.testing.assert_allclose(lam[-1], 10.0 ** (10.0 / 2), rtol=1e-10)
        # Should be monotonically non-decreasing
        assert np.all(np.diff(lam) >= 0)

    def test_generate_lambda_alpha_zero(self):
        """Alpha=0 should give all ones."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        lam = gen.generate_lambda(alpha=0.0, dim=5)
        np.testing.assert_allclose(lam, np.ones(5), atol=1e-12)

    def test_generate_lambda_dim1(self):
        """Dimension 1 should give [1.0]."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        lam = gen.generate_lambda(alpha=10.0, dim=1)
        np.testing.assert_allclose(lam, np.array([1.0]))

    def test_generate_instance(self):
        """generate_instance should return all required params."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        params = gen.generate_instance(fid=1, iid=1, dim=5)
        assert "xopt" in params
        assert "fopt" in params
        assert "R" in params
        assert "Q" in params
        assert params["xopt"].shape == (5,)
        assert isinstance(params["fopt"], float)
        assert params["R"].shape == (5, 5)
        assert params["Q"].shape == (5, 5)

    def test_generate_instance_reproducible(self):
        """Same inputs should produce identical params."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        p1 = gen.generate_instance(fid=10, iid=3, dim=10)
        p2 = gen.generate_instance(fid=10, iid=3, dim=10)
        np.testing.assert_array_equal(p1["xopt"], p2["xopt"])
        assert p1["fopt"] == p2["fopt"]
        np.testing.assert_array_equal(p1["R"], p2["R"])
        np.testing.assert_array_equal(p1["Q"], p2["Q"])

    def test_rotation_seeds_match_coco_convention(self):
        """R should use rseed+1000000, Q should use rseed."""
        from pyMOFL.utils.bbob_instance import (
            BBOBInstanceGenerator,
            _compute_rotation,
            _compute_rseed,
        )

        gen = BBOBInstanceGenerator()
        params = gen.generate_instance(fid=1, iid=1, dim=5)
        rseed = _compute_rseed(1, 1)
        R_expected = _compute_rotation(5, rseed + 1000000)
        Q_expected = _compute_rotation(5, rseed)
        np.testing.assert_array_equal(params["R"], R_expected)
        np.testing.assert_array_equal(params["Q"], Q_expected)


class TestCOCOPRNG:
    """Tests for the COCO-compatible PRNG."""

    def test_uniform_range(self):
        """All values should be in (0, 1)."""
        from pyMOFL.utils.bbob_instance import _bbob2009_unif

        values = _bbob2009_unif(1000, 42)
        assert all(0 < v <= 1 for v in values)

    def test_uniform_deterministic(self):
        """Same seed should produce same sequence."""
        from pyMOFL.utils.bbob_instance import _bbob2009_unif

        v1 = _bbob2009_unif(100, 42)
        v2 = _bbob2009_unif(100, 42)
        assert v1 == v2

    def test_gaussian_deterministic(self):
        """Same seed should produce same Gaussian sequence."""
        from pyMOFL.utils.bbob_instance import _bbob2009_gauss

        g1 = _bbob2009_gauss(50, 42)
        g2 = _bbob2009_gauss(50, 42)
        assert g1 == g2

    def test_gaussian_distribution(self):
        """Gaussian values should have mean ≈ 0 and std ≈ 1."""
        from pyMOFL.utils.bbob_instance import _bbob2009_gauss

        g = _bbob2009_gauss(10000, 12345)
        arr = np.array(g)
        assert abs(np.mean(arr)) < 0.05
        assert abs(np.std(arr) - 1.0) < 0.05

    def test_rseed_formula(self):
        """rseed = fid + 10000 * iid for most functions."""
        from pyMOFL.utils.bbob_instance import _compute_rseed

        assert _compute_rseed(1, 1) == 10001
        assert _compute_rseed(10, 3) == 30010
        assert _compute_rseed(4, 1) == 10003  # special case: base=3
        assert _compute_rseed(18, 2) == 20017  # special case: base=17
