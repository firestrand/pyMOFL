"""
Tests for CEC function factory that builds functions from JSON configuration.
"""

from pathlib import Path

import numpy as np
import pytest

from pyMOFL.factories import FunctionFactory
from pyMOFL.factories.function_factory import DataLoader, FunctionRegistry
from pyMOFL.utils import find_suite_function_config, inject_dimension, load_suite_config
from tests.utils.validation import load_cec_validation_raw


class TestCECFactory:
    """Test cases for the CEC function factory."""

    @pytest.fixture
    def cec2005_config_path(self):
        """Path to CEC 2005 configuration file."""
        return Path("src/pyMOFL/constants/cec/2005/cec2005_suite.json")

    @pytest.fixture
    def cec2005_config(self, cec2005_config_path):
        """Load CEC 2005 configuration."""
        return load_suite_config(cec2005_config_path)

    @pytest.fixture
    def f01_validation_data(self):
        """Load F01 validation data."""
        return load_cec_validation_raw(2005, 1)

    def test_create_f01_dimension_10(self, cec2005_config, f01_validation_data):
        """Test creating F01 (shifted sphere) with dimension 10."""
        # Get F01 configuration
        f01_config = find_suite_function_config(cec2005_config, "cec05_f01_shifted_sphere")

        loader = DataLoader(base_path="src/pyMOFL/constants/cec/2005")
        registry = FunctionRegistry()
        factory = FunctionFactory(data_loader=loader, registry=registry)

        # Build function for dimension 10
        # Create function with dimension embedded in config
        func_config_with_dim = inject_dimension(f01_config, 10)
        f01 = factory.create_function(func_config_with_dim)

        # Check function properties
        assert f01.dimension == 10
        assert f01.initialization_bounds.low.shape == (10,)
        assert f01.initialization_bounds.high.shape == (10,)
        assert np.all(f01.initialization_bounds.low == -100)
        assert np.all(f01.initialization_bounds.high == 100)

        # Find validation case for dimension 10
        validation_case = None
        for case in f01_validation_data["cases"]:
            if case["dimension"] == 10:
                validation_case = case
                break

        assert validation_case is not None, "No validation data for dimension 10"

        # Test evaluation at optimum
        optimum = np.array(validation_case["optimum"])
        expected_value = validation_case["outputs"]["optimum"]

        # Evaluate at optimum
        value = f01.evaluate(optimum)

        # Check that we get the expected minimum value (with tolerance)
        assert np.isclose(value, expected_value, rtol=1e-6), (
            f"Expected {expected_value}, got {value}"
        )

    def test_create_f01_dimension_2(self, cec2005_config, f01_validation_data):
        """Test creating F01 with dimension 2."""
        # Get F01 configuration
        f01_config = find_suite_function_config(cec2005_config, "cec05_f01_shifted_sphere")

        loader = DataLoader(base_path="src/pyMOFL/constants/cec/2005")
        registry = FunctionRegistry()
        factory = FunctionFactory(data_loader=loader, registry=registry)

        # Build function for dimension 2
        # Create function with dimension embedded in config
        func_config_with_dim = inject_dimension(f01_config, 2)
        f01 = factory.create_function(func_config_with_dim)

        # Check function properties
        assert f01.dimension == 2

        # Find validation case for dimension 2
        validation_case = None
        for case in f01_validation_data["cases"]:
            if case["dimension"] == 2:
                validation_case = case
                break

        assert validation_case is not None

        # Test evaluation at optimum
        optimum = np.array(validation_case["optimum"])
        expected_value = validation_case["outputs"]["optimum"]

        # Evaluate at optimum
        value = f01.evaluate(optimum)

        # Check that we get the expected minimum value
        assert np.isclose(value, expected_value, rtol=1e-6)

    def test_factory_loads_shift_vector(self, cec2005_config):
        """Test that factory correctly loads shift vector from file."""
        # Get F01 configuration
        f01_config = find_suite_function_config(cec2005_config, "cec05_f01_shifted_sphere")

        loader = DataLoader(base_path="src/pyMOFL/constants/cec/2005")
        registry = FunctionRegistry()
        factory = FunctionFactory(data_loader=loader, registry=registry)

        # Build function for dimension 50 (to match the shift vector file)
        # Create function with dimension embedded in config
        func_config_with_dim = inject_dimension(f01_config, 50)
        f01 = factory.create_function(func_config_with_dim)

        # The shift should be applied internally
        # Test that the function works (shift vector loaded correctly)
        x = np.zeros(50)
        value = f01.evaluate(x)

        # Should not be -450 (the bias) because of the shift
        assert value != -450.0
