"""Tests for GNBGSuiteFactory."""

import numpy as np
import pytest

from pyMOFL.core.function import OptimizationFunction
from pyMOFL.factories.gnbg_suite_factory import GNBGSuiteFactory


class TestGNBGSuiteFactory:
    """Test suite for GNBGSuiteFactory."""

    @pytest.fixture
    def factory(self):
        return GNBGSuiteFactory()

    def test_factory_initialization(self, factory):
        assert factory.suite_id == "gnbg_suite"
        assert "GNBG" in factory.name
        assert factory.NUM_FUNCTIONS == 24
        assert 10 in factory.SUPPORTED_DIMENSIONS

    def test_create_function_by_int_and_str(self, factory):
        f_int = factory.create_function(1, dim=10)
        f_str = factory.create_function("gnbg_f01", dim=10)
        f_short = factory.create_function("f01", dim=10)

        assert isinstance(f_int, OptimizationFunction)
        assert isinstance(f_str, OptimizationFunction)
        assert isinstance(f_short, OptimizationFunction)

        x = np.ones(10) * 0.5
        np.testing.assert_allclose(f_int.evaluate(x), f_str.evaluate(x))
        np.testing.assert_allclose(f_int.evaluate(x), f_short.evaluate(x))

    def test_create_all_24_functions_dim10(self, factory):
        """All 24 functions should instantiate and evaluate at D=10."""
        rng = np.random.default_rng(42)
        X = rng.uniform(-3, 3, size=(5, 10))

        for fid in range(1, 25):
            func = factory.create_function(fid, dim=10)
            assert func.dimension == 10

            # Single evaluation
            val = func.evaluate(X[0])
            assert np.isfinite(val), f"Function {fid} produced non-finite value: {val}"

            # Batch evaluation
            batch_vals = func.evaluate_batch(X)
            assert batch_vals.shape == (5,)
            assert np.all(np.isfinite(batch_vals))
            np.testing.assert_allclose(batch_vals[0], val)

    @pytest.mark.parametrize("dim", [2, 30, 50, 100])
    def test_create_function_supported_dims(self, factory, dim):
        """Test instantiation across supported dimensions."""
        for fid in range(1, 25):
            func = factory.create_function(fid, dim=dim)
            assert func.dimension == dim
            x = np.zeros(dim)
            val = func.evaluate(x)
            assert np.isfinite(val)

    def test_create_suite(self, factory):
        suite = factory.create_suite(dim=10)
        assert len(suite) == 24
        assert all(isinstance(f, OptimizationFunction) for f in suite)

    def test_invalid_fid(self, factory):
        with pytest.raises(ValueError, match="Unknown GNBG function"):
            factory.create_function(0, dim=10)

        with pytest.raises(ValueError, match="Unknown GNBG function"):
            factory.create_function(25, dim=10)

        with pytest.raises(ValueError, match="Unknown GNBG function"):
            factory.create_function("invalid_fid", dim=10)

    def test_invalid_dim(self, factory):
        with pytest.raises(ValueError, match="Dimension 7 not supported"):
            factory.create_function(1, dim=7)

    def test_get_function_info(self, factory):
        info_all = factory.get_function_info()
        assert len(info_all) == 24

        info_f1 = factory.get_function_info(1)
        assert info_f1["id"] == "gnbg_f01"
        assert info_f1["category"] == "Unimodal"

        info_f16 = factory.get_function_info("gnbg_f16")
        assert info_f16["id"] == "gnbg_f16"
        assert info_f16["category"] == "Multi-Component Multimodal"
