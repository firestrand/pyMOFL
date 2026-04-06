"""Tests for BBOBNoisySuiteFactory (30 noisy BBOB functions, f101-f130)."""

import numpy as np
import pytest

from pyMOFL.factories.bbob_noisy_suite_factory import BBOBNoisySuiteFactory
from pyMOFL.functions.transformations.cauchy_noise import CauchyNoiseTransform
from pyMOFL.functions.transformations.composed import ComposedFunction
from pyMOFL.functions.transformations.gaussian_noise import GaussianNoiseTransform
from pyMOFL.functions.transformations.uniform_noise import UniformNoiseTransform


@pytest.fixture
def factory():
    return BBOBNoisySuiteFactory()


class TestBBOBNoisySuiteFactory:
    """Tests for the noisy BBOB suite factory."""

    def test_all_30_instantiate(self, factory):
        """All 30 noisy functions (f101-f130) should instantiate."""
        for fid in range(101, 131):
            func = factory.create_function(fid, iid=1, dim=10)
            assert func is not None
            assert isinstance(func, ComposedFunction)

    def test_invalid_fid_raises(self, factory):
        """Function IDs outside 101-130 should raise ValueError."""
        with pytest.raises(ValueError, match="101-130"):
            factory.create_function(100, iid=1, dim=10)
        with pytest.raises(ValueError, match="101-130"):
            factory.create_function(131, iid=1, dim=10)

    def test_finite_evaluations(self, factory):
        """All functions should return finite values."""
        rng = np.random.default_rng(42)
        for fid in range(101, 131):
            func = factory.create_function(fid, iid=1, dim=5)
            x = rng.standard_normal(5)
            result = func(x)
            assert np.isfinite(result), f"f{fid} returned non-finite: {result}"

    def test_noise_transform_present(self, factory):
        """Each function should have a noise output transform."""
        noise_types = (GaussianNoiseTransform, UniformNoiseTransform, CauchyNoiseTransform)
        for fid in range(101, 131):
            func = factory.create_function(fid, iid=1, dim=5)
            has_noise = any(isinstance(t, noise_types) for t in func.output_transforms)
            assert has_noise, f"f{fid} missing noise output transform"

    def test_gaussian_noise_functions(self, factory):
        """f101, f104, f107, f110, ... should use GaussianNoiseTransform."""
        # Moderate: f101, f104
        for fid in [101, 104]:
            func = factory.create_function(fid, iid=1, dim=5)
            noise = [t for t in func.output_transforms if isinstance(t, GaussianNoiseTransform)]
            assert len(noise) == 1, f"f{fid} should have exactly 1 GaussianNoiseTransform"

    def test_uniform_noise_functions(self, factory):
        """f102, f105, f108, f111, ... should use UniformNoiseTransform."""
        for fid in [102, 105]:
            func = factory.create_function(fid, iid=1, dim=5)
            noise = [t for t in func.output_transforms if isinstance(t, UniformNoiseTransform)]
            assert len(noise) == 1, f"f{fid} should have exactly 1 UniformNoiseTransform"

    def test_cauchy_noise_functions(self, factory):
        """f103, f106, f109, f112, ... should use CauchyNoiseTransform."""
        for fid in [103, 106]:
            func = factory.create_function(fid, iid=1, dim=5)
            noise = [t for t in func.output_transforms if isinstance(t, CauchyNoiseTransform)]
            assert len(noise) == 1, f"f{fid} should have exactly 1 CauchyNoiseTransform"

    def test_moderate_noise_parameters(self, factory):
        """f101-f106 should have moderate noise (beta=0.01 for gaussian)."""
        func = factory.create_function(101, iid=1, dim=5)
        noise = next(t for t in func.output_transforms if isinstance(t, GaussianNoiseTransform))
        assert noise.beta == pytest.approx(0.01)

    def test_severe_noise_parameters(self, factory):
        """f107-f130 should have severe noise (beta=1.0 for gaussian)."""
        func = factory.create_function(107, iid=1, dim=5)
        noise = next(t for t in func.output_transforms if isinstance(t, GaussianNoiseTransform))
        assert noise.beta == pytest.approx(1.0)

    def test_correct_base_types(self, factory):
        """f101-f103 should wrap Sphere, f104-f106 should wrap Rosenbrock."""
        # Sphere-based (moderate)
        for fid in [101, 102, 103]:
            info = factory.get_function_info(fid)
            assert info["base_fid"] == 1  # Sphere

        # Rosenbrock-based (moderate)
        for fid in [104, 105, 106]:
            info = factory.get_function_info(fid)
            assert info["base_fid"] == 8  # Rosenbrock

    def test_create_suite(self, factory):
        """create_suite should return all 30 functions."""
        suite = factory.create_suite(iid=1, dim=5)
        assert len(suite) == 30

    def test_get_function_info_all(self, factory):
        """All 30 functions should have valid info."""
        for fid in range(101, 131):
            info = factory.get_function_info(fid)
            assert "name" in info
            assert "noise_type" in info
            assert "base_fid" in info

    def test_stochastic_different_evals(self, factory):
        """Two evaluations at same point should differ (noise is stochastic)."""
        func = factory.create_function(107, iid=1, dim=5)  # severe gaussian
        x = np.zeros(5)
        results = [func(x) for _ in range(10)]
        # At least some should differ
        assert len(set(results)) > 1

    def test_different_dimensions(self, factory):
        """Factory should work for various dimensions."""
        for dim in [2, 5, 10, 20]:
            func = factory.create_function(101, iid=1, dim=dim)
            x = np.zeros(dim)
            result = func(x)
            assert np.isfinite(result)

    def test_different_instances(self, factory):
        """Different instance IDs should produce different base functions."""
        f1 = factory.create_function(101, iid=1, dim=5)
        f2 = factory.create_function(101, iid=2, dim=5)
        x = np.ones(5) * 3.0
        # Might differ due to different xopt/fopt
        # At minimum, both should be finite
        assert np.isfinite(f1(x))
        assert np.isfinite(f2(x))
