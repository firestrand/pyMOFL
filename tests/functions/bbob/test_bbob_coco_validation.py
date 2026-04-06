"""
Strict COCO reference validation for BBOB functions.

Compares pyMOFL BBOB functions against the COCO (cocoex) reference
implementation with tight numerical tolerances. This validates that
pyMOFL produces bit-for-bit compatible results with the COCO framework.

Requires the `cocoex` package (pip install coco-experiment). Skipped if not installed.
"""

import numpy as np
import pytest

cocoex = pytest.importorskip("cocoex")

from pyMOFL.factories.bbob_suite_factory import BBOBSuiteFactory  # noqa: E402
from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator  # noqa: E402

# Functions with full COCO parity (external transform chains match exactly)
PARITY_FIDS = [1, 2, 8, 10, 12, 13, 15]

# Functions with monolithic COCO implementations (can't match via external transforms)
# f5: COCO uses sqrt(10) slopes with boundary clipping, no shift
# f20: COCO has complex internal chain (x_hat, z_hat, conditioning)
# f24: COCO has internal x_hat, affine, two-basin, penalty*1e4
MONOLITHIC_FIDS = [5, 20, 24]

ALL_TEST_FIDS = PARITY_FIDS + MONOLITHIC_FIDS
TEST_DIMS = [2, 10, 40]
N_TEST_POINTS = 10
ATOL = 1e-8
RTOL = 1e-10


def _is_monolithic(fid: int) -> bool:
    return fid in MONOLITHIC_FIDS


@pytest.fixture(scope="module")
def factory():
    return BBOBSuiteFactory()


@pytest.fixture(scope="module")
def instance_gen():
    return BBOBInstanceGenerator()


def _get_coco_function(fid: int, dim: int, iid: int = 1):
    """Get a COCO benchmark function."""
    suite = cocoex.Suite("bbob", f"instances:{iid}", f"function_indices:{fid} dimensions:{dim}")
    for problem in suite:
        return problem
    return None


class TestStrictCOCOParity:
    """Strict numeric comparison of pyMOFL vs COCO at random test points."""

    @pytest.mark.parametrize("fid", ALL_TEST_FIDS)
    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_random_points(self, factory, fid, dim):
        """pyMOFL and COCO should agree within tight tolerance at random points."""
        if _is_monolithic(fid):
            pytest.xfail(
                f"f{fid} has monolithic COCO implementation; "
                "external transform chain cannot reproduce exact values"
            )

        coco_func = _get_coco_function(fid, dim)
        if coco_func is None:
            pytest.skip(f"COCO f{fid} dim={dim} not available")

        pymofl_func = factory.create_function(fid=fid, iid=1, dim=dim)

        rng = np.random.default_rng(fid * 1000 + dim)
        for i in range(N_TEST_POINTS):
            x = rng.uniform(-5, 5, size=dim)
            coco_val = coco_func(x)
            pymofl_val = pymofl_func.evaluate(x)

            np.testing.assert_allclose(
                pymofl_val,
                coco_val,
                atol=ATOL,
                rtol=RTOL,
                err_msg=(f"f{fid} dim={dim} point {i}: pyMOFL={pymofl_val}, COCO={coco_val}"),
            )


class TestCOCOAtOrigin:
    """Strict comparison of pyMOFL vs COCO at the origin."""

    @pytest.mark.parametrize("fid", ALL_TEST_FIDS)
    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_at_origin(self, factory, fid, dim):
        """pyMOFL and COCO should agree at the zero vector."""
        if _is_monolithic(fid):
            pytest.xfail(
                f"f{fid} has monolithic COCO implementation; "
                "external transform chain cannot reproduce exact values"
            )

        coco_func = _get_coco_function(fid, dim)
        if coco_func is None:
            pytest.skip(f"COCO f{fid} dim={dim} not available")

        pymofl_func = factory.create_function(fid=fid, iid=1, dim=dim)
        x = np.zeros(dim)

        coco_val = coco_func(x)
        pymofl_val = pymofl_func.evaluate(x)

        np.testing.assert_allclose(
            pymofl_val,
            coco_val,
            atol=ATOL,
            rtol=RTOL,
            err_msg=f"f{fid} dim={dim} at origin: pyMOFL={pymofl_val}, COCO={coco_val}",
        )


class TestCOCOAtOptimum:
    """Validate that pyMOFL returns fopt at its own xopt."""

    @pytest.mark.parametrize("fid", ALL_TEST_FIDS)
    @pytest.mark.parametrize("dim", TEST_DIMS)
    def test_at_optimum(self, factory, instance_gen, fid, dim):
        """pyMOFL(xopt) should equal fopt (within tolerance)."""
        if _is_monolithic(fid):
            pytest.xfail(
                f"f{fid} has monolithic COCO implementation; "
                "xopt/fopt may not match COCO's internal optimum"
            )

        pymofl_func = factory.create_function(fid=fid, iid=1, dim=dim)
        params = instance_gen.generate_instance(fid=fid, iid=1, dim=dim)
        xopt = params["xopt"]
        fopt = params["fopt"]

        pymofl_val = pymofl_func.evaluate(xopt)

        np.testing.assert_allclose(
            pymofl_val,
            fopt,
            atol=ATOL,
            rtol=RTOL,
            err_msg=(f"f{fid} dim={dim} at optimum: pyMOFL(xopt)={pymofl_val}, fopt={fopt}"),
        )
