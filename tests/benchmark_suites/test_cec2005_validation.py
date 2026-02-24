"""
Validation tests for CEC 2005 benchmark suite.

This module tests ALL CEC 2005 functions against their optimum points
using the validation data. The tests are simple and trust-based:
- Use the generic FunctionFactory to create functions
- Test against validation data optimum points
- Output clear PASS/FAIL for each function
"""

import numpy as np
import pytest

from pyMOFL.factories import FunctionFactory
from pyMOFL.utils import find_suite_function_config, inject_dimension, load_suite_config
from tests.utils.validation import extract_cec_function_number, load_cec_suite_validation_data


class TestCEC2005Validation:
    """Test all CEC 2005 functions against their optimum points."""

    @pytest.fixture(scope="class")
    def cec2005_config(self):
        """Load CEC 2005 configuration."""
        return load_suite_config("src/pyMOFL/constants/cec/2005/cec2005_suite.json")

    @pytest.fixture(scope="class")
    def factory(self):
        """Create benchmark factory."""
        from pyMOFL.factories.function_factory import DataLoader, FunctionRegistry

        loader = DataLoader(base_path="src/pyMOFL/constants/cec/2005")
        registry = FunctionRegistry()
        return FunctionFactory(data_loader=loader, registry=registry)

    @pytest.fixture(scope="class")
    def validation_data(self):
        """Load all validation data files."""
        return load_cec_suite_validation_data(2005)

    def test_all_functions_at_optimum(self, cec2005_config, factory, validation_data):
        """Test all CEC 2005 functions at their optimum points."""
        results = []

        # Test each function in the suite
        for func_config in cec2005_config["functions"]:
            func_id = func_config["id"]
            func_num = extract_cec_function_number(func_id)
            if func_num is None:
                continue

            # Get validation data
            if func_num not in validation_data:
                results.append(f"{func_num.upper()}: SKIP - No validation data")
                continue

            val_data = validation_data[func_num]

            # Test multiple dimensions
            dimension_results = []
            for case in val_data["cases"]:
                dimension = case["dimension"]

                try:
                    # Create function
                    # Create function with dimension embedded in config
                    func = factory.create_function(
                        inject_dimension(func_config["function"], dimension)
                    )

                    # Get optimum point and expected value
                    optimum = np.array(case["optimum"])
                    expected_value = case["outputs"]["optimum"]

                    # Evaluate at optimum
                    actual_value = func.evaluate(optimum)

                    # Check if values match (with tolerance)
                    if np.isclose(actual_value, expected_value, rtol=1e-6, atol=1e-9):
                        dimension_results.append(f"D{dimension}: PASS")
                    else:
                        dimension_results.append(
                            f"D{dimension}: FAIL - Expected {expected_value:.6f}, Got {actual_value:.6f}"
                        )
                except Exception as e:
                    dimension_results.append(f"D{dimension}: ERROR - {e!s}")

            # Aggregate results for this function
            if all("PASS" in r for r in dimension_results):
                results.append(f"{func_num.upper()}: PASS")
            else:
                results.append(f"{func_num.upper()}: MIXED - {', '.join(dimension_results)}")

        # Print results summary
        print("\n" + "=" * 60)
        print("CEC 2005 VALIDATION RESULTS")
        print("=" * 60)
        for result in results:
            print(result)
        print("=" * 60)

        # Assert all functions pass
        failed_functions = [r for r in results if "FAIL" in r or "ERROR" in r]
        assert len(failed_functions) == 0, f"Failed functions: {failed_functions}"

    @pytest.mark.parametrize("func_num", [f"f{i:02d}" for i in range(1, 26)])
    def test_individual_function(self, cec2005_config, factory, validation_data, func_num):
        """Test individual CEC 2005 function at optimum."""
        # Skip if no validation data
        if func_num not in validation_data:
            pytest.skip(f"No validation data for {func_num}")

        # Find function config
        try:
            func_config = find_suite_function_config(cec2005_config, func_num)
        except ValueError:
            pytest.skip(f"No configuration for {func_num}")

        val_data = validation_data[func_num]

        # Test first available dimension
        case = val_data["cases"][0]
        dimension = case["dimension"]

        # Create function
        # Create function with dimension embedded in config
        func = factory.create_function(inject_dimension(func_config, dimension))

        # Get optimum point and expected value
        optimum = np.array(case["optimum"])
        expected_value = case["outputs"]["optimum"]

        # Evaluate at optimum
        actual_value = func.evaluate(optimum)

        # Check if values match
        assert np.isclose(actual_value, expected_value, rtol=1e-6, atol=1e-9), (
            f"{func_num.upper()}: Expected {expected_value:.6f}, Got {actual_value:.6f}"
        )

    def test_dimensions_consistency(self, cec2005_config, factory, validation_data):
        """Test that functions produce consistent results across dimensions."""
        for func_config in cec2005_config["functions"]:
            func_id = func_config["id"]
            func_num = extract_cec_function_number(func_id)
            if func_num is None:
                continue

            if func_num not in validation_data:
                continue

            val_data = validation_data[func_num]

            # Test that bias is consistent across dimensions
            biases = []
            for case in val_data["cases"]:
                dimension = case["dimension"]

                try:
                    # Create function
                    # Create function with dimension embedded in config
                    func = factory.create_function(
                        inject_dimension(func_config["function"], dimension)
                    )

                    # Evaluate at optimum
                    optimum = np.array(case["optimum"])
                    value = func.evaluate(optimum)

                    # The optimum value should be the bias
                    biases.append(value)
                except Exception:
                    pass

            # Check all biases are close
            if len(biases) > 1:
                assert np.allclose(biases, biases[0], rtol=1e-6), (
                    f"{func_num}: Inconsistent optimum values across dimensions: {biases}"
                )
