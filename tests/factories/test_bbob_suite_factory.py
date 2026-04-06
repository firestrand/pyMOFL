"""
Tests for BBOBSuiteFactory - BBOB noiseless suite generation.

Validates that all 24 BBOB functions can be instantiated with correct
base types, transform chains, and finite evaluations.
"""

import numpy as np
import pytest

from pyMOFL.functions.transformations.composed import ComposedFunction


class TestBBOBSuiteFactory:
    """Tests for BBOBSuiteFactory."""

    @pytest.fixture
    def factory(self):
        from pyMOFL.factories.bbob_suite_factory import BBOBSuiteFactory

        return BBOBSuiteFactory()

    def test_create_function_all_24_dim2(self, factory):
        """All 24 functions should instantiate at dim 2."""
        for fid in range(1, 25):
            func = factory.create_function(fid=fid, iid=1, dim=2)
            assert isinstance(func, ComposedFunction), f"fid={fid} not ComposedFunction"
            assert func.dimension == 2, f"fid={fid} wrong dimension"

    def test_create_function_all_24_dim10(self, factory):
        """All 24 functions should instantiate at dim 10."""
        for fid in range(1, 25):
            func = factory.create_function(fid=fid, iid=1, dim=10)
            assert isinstance(func, ComposedFunction), f"fid={fid} not ComposedFunction"
            assert func.dimension == 10, f"fid={fid} wrong dimension"

    def test_create_function_all_24_dim20(self, factory):
        """All 24 functions should instantiate at dim 20."""
        for fid in range(1, 25):
            func = factory.create_function(fid=fid, iid=1, dim=20)
            assert isinstance(func, ComposedFunction), f"fid={fid} not ComposedFunction"
            assert func.dimension == 20, f"fid={fid} wrong dimension"

    def test_finite_evaluation(self, factory):
        """All 24 functions should produce finite values."""
        rng = np.random.default_rng(42)
        for fid in range(1, 25):
            func = factory.create_function(fid=fid, iid=1, dim=5)
            x = rng.uniform(-4, 4, size=5)
            result = func.evaluate(x)
            assert np.isfinite(result), f"fid={fid} non-finite: {result}"

    def test_different_instances_different_evals(self, factory):
        """Different instances should produce different function values."""
        x = np.ones(5) * 2.0
        for fid in [1, 3, 8, 15, 20]:
            f1 = factory.create_function(fid=fid, iid=1, dim=5)
            f2 = factory.create_function(fid=fid, iid=2, dim=5)
            v1 = f1.evaluate(x)
            v2 = f2.evaluate(x)
            assert v1 != pytest.approx(v2, rel=1e-6), (
                f"fid={fid}: instances 1 and 2 gave same value"
            )

    def test_correct_base_types_separable(self, factory):
        """f1-f5 should use the correct base function types."""
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction
        from pyMOFL.functions.benchmark.elliptic import HighConditionedElliptic
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction
        from pyMOFL.functions.benchmark.rastrigin import RastriginFunction
        from pyMOFL.functions.benchmark.sphere import SphereFunction

        expected = {
            1: SphereFunction,
            2: HighConditionedElliptic,
            3: RastriginFunction,
            4: BucheRastriginFunction,
            5: LinearSlopeFunction,
        }
        for fid, cls in expected.items():
            func = factory.create_function(fid=fid, iid=1, dim=5)
            assert isinstance(func.base_function, cls), (
                f"fid={fid} base is {type(func.base_function).__name__}, expected {cls.__name__}"
            )

    def test_correct_base_types_moderate(self, factory):
        """f6-f9 should use the correct base function types."""
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction
        from pyMOFL.functions.benchmark.rosenbrock import RosenbrockFunction
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        expected = {
            6: AttractiveSectorFunction,
            7: StepEllipsoidFunction,
            8: RosenbrockFunction,
            9: RosenbrockFunction,
        }
        for fid, cls in expected.items():
            func = factory.create_function(fid=fid, iid=1, dim=5)
            assert isinstance(func.base_function, cls), (
                f"fid={fid} base is {type(func.base_function).__name__}, expected {cls.__name__}"
            )

    def test_correct_base_types_ill_conditioned(self, factory):
        """f10-f14 should use the correct base function types."""
        from pyMOFL.functions.benchmark.bent_cigar import BentCigarFunction, DiscusFunction
        from pyMOFL.functions.benchmark.different_powers import DifferentPowersFunction
        from pyMOFL.functions.benchmark.elliptic import HighConditionedElliptic
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        expected = {
            10: HighConditionedElliptic,
            11: DiscusFunction,
            12: BentCigarFunction,
            13: SharpRidgeFunction,
            14: DifferentPowersFunction,
        }
        for fid, cls in expected.items():
            func = factory.create_function(fid=fid, iid=1, dim=5)
            assert isinstance(func.base_function, cls), (
                f"fid={fid} base is {type(func.base_function).__name__}, expected {cls.__name__}"
            )

    def test_correct_base_types_multimodal(self, factory):
        """f15-f19 should use the correct base function types."""
        from pyMOFL.functions.benchmark.rastrigin import RastriginFunction
        from pyMOFL.functions.benchmark.rosenbrock import GriewankOfRosenbrock
        from pyMOFL.functions.benchmark.schaffer import SchaffersF7Function
        from pyMOFL.functions.benchmark.weierstrass import WeierstrassFunction

        expected = {
            15: RastriginFunction,
            16: WeierstrassFunction,
            17: SchaffersF7Function,
            18: SchaffersF7Function,
            19: GriewankOfRosenbrock,
        }
        for fid, cls in expected.items():
            func = factory.create_function(fid=fid, iid=1, dim=5)
            assert isinstance(func.base_function, cls), (
                f"fid={fid} base is {type(func.base_function).__name__}, expected {cls.__name__}"
            )

    def test_correct_base_types_weakly_structured(self, factory):
        """f20-f24 should use the correct base function types."""
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction
        from pyMOFL.functions.benchmark.katsuura import KatsuuraFunction
        from pyMOFL.functions.benchmark.lunacek import LunacekBiRastriginFunction
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction

        expected = {
            20: SchwefelSinFunction,
            21: GallagherPeaksFunction,
            22: GallagherPeaksFunction,
            23: KatsuuraFunction,
            24: LunacekBiRastriginFunction,
        }
        for fid, cls in expected.items():
            func = factory.create_function(fid=fid, iid=1, dim=5)
            assert isinstance(func.base_function, cls), (
                f"fid={fid} base is {type(func.base_function).__name__}, expected {cls.__name__}"
            )

    def test_bias_present(self, factory):
        """All functions should have a bias (fopt) output transform."""
        from pyMOFL.functions.transformations import BiasTransform

        for fid in range(1, 25):
            func = factory.create_function(fid=fid, iid=1, dim=5)
            has_bias = any(isinstance(t, BiasTransform) for t in func.output_transforms)
            assert has_bias, f"fid={fid} missing bias transform"

    def test_penalty_present_where_expected(self, factory):
        """Functions that have f_pen in COCO: f4, f7, f16, f17, f18, f23."""
        from pyMOFL.functions.transformations import BoundaryPenaltyTransform

        fids_with_penalty = {4, 7, 16, 17, 18, 23}
        for fid in fids_with_penalty:
            func = factory.create_function(fid=fid, iid=1, dim=5)
            has_penalty = any(
                isinstance(t, BoundaryPenaltyTransform) for t in func.penalty_transforms
            )
            assert has_penalty, f"fid={fid} missing boundary penalty transform"

    def test_create_suite(self, factory):
        """create_suite should return 24 functions."""
        suite = factory.create_suite(iid=1, dim=5)
        assert len(suite) == 24
        for func in suite:
            assert isinstance(func, ComposedFunction)

    def test_get_function_info(self, factory):
        """get_function_info should return metadata dict."""
        for fid in range(1, 25):
            info = factory.get_function_info(fid)
            assert "name" in info
            assert "category" in info
            assert "base_function" in info

    def test_invalid_fid(self, factory):
        """Invalid fid should raise ValueError."""
        with pytest.raises(ValueError, match=r"Unknown.*function"):
            factory.create_function(fid=0, iid=1, dim=5)
        with pytest.raises(ValueError, match=r"Unknown.*function"):
            factory.create_function(fid=25, iid=1, dim=5)

    def test_evaluate_batch(self, factory):
        """Batch evaluation should work for all functions."""
        X = np.random.default_rng(42).uniform(-4, 4, size=(5, 3))
        for fid in range(1, 25):
            func = factory.create_function(fid=fid, iid=1, dim=3)
            results = func.evaluate_batch(X)
            assert results.shape == (5,), f"fid={fid} wrong batch shape"
            assert np.all(np.isfinite(results)), f"fid={fid} non-finite batch results"

    def test_at_optimum_bias_matches_fopt(self, factory):
        """At x_opt, functions should return fopt (for simple separable functions)."""
        from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

        gen = BBOBInstanceGenerator()
        # f1 (sphere): at x_opt, sphere(0)=0, so result should be fopt
        params = gen.generate_instance(fid=1, iid=1, dim=5)
        func = factory.create_function(fid=1, iid=1, dim=5)
        result = func.evaluate(params["xopt"])
        # Should be close to fopt (sphere(shifted_origin) = 0 + fopt)
        np.testing.assert_allclose(result, params["fopt"], atol=1e-6)
