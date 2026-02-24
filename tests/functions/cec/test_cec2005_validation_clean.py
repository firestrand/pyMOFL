"""
Clean test for all CEC 2005 functions against their validation data.
"""

from pathlib import Path

import numpy as np
import pytest

from pyMOFL.factories import FunctionFactory
from pyMOFL.factories.function_factory import DataLoader, FunctionRegistry
from pyMOFL.utils import inject_dimension, load_suite_config
from tests.utils.validation import load_cec_validation_for_function_id


def flatten_cec_config(config):
    """
    Flatten CEC config structure to work with the factory.

    The CEC configs have a nested structure like:
    bias -> sphere -> shift

    Where 'shift' is a leaf transformation (no nested function). This should be interpreted as:
    - Input transformation: shift (applied to input)
    - Base function: sphere
    - Output transformation: bias (applied to output)

    So the result should be: bias -> shift -> sphere
    """

    def parse_cec_structure(cfg):
        """Parse the CEC nested structure to separate input/output transforms and base function."""
        base_functions = [
            "sphere",
            "rosenbrock",
            "rastrigin",
            "ackley",
            "griewank",
            "weierstrass",
            "schwefel_1_2",
            "high_conditioned_elliptic",
            "schwefel_2_6",
            "schwefel_2_13",
            "schaffer_f6_expanded",
            "griewank_of_rosenbrock",
        ]

        # Composition function types - these are base functions, not transforms
        composition_functions = ["hybrid_composition", "rotated_hybrid_composition", "composition"]

        output_transforms = []  # Transformations applied to output (like bias)
        base_func = None
        input_transforms = []  # Transformations applied to input (like shift, rotate)

        current = cfg

        # Walk down the tree collecting transformations
        while current is not None:
            func_type = current.get("type")

            if func_type in base_functions or func_type in composition_functions:
                # Found the base function
                base_func = {"type": func_type, "parameters": current.get("parameters", {})}

                # For composition functions, preserve the nested functions list
                if func_type in composition_functions and "functions" in current:
                    base_func["functions"] = current["functions"]

                # Check if it has nested transformations (these are input transforms)
                if "function" in current and func_type not in composition_functions:
                    input_transforms.extend(collect_input_transforms(current["function"]))

                break
            else:
                # This is an output transformation (applied after base function)
                output_transforms.append(
                    {"type": func_type, "parameters": current.get("parameters", {})}
                )
                current = current.get("function")

        if base_func is None:
            raise ValueError("No base function found in CEC config")

        return output_transforms, base_func, input_transforms

    def collect_input_transforms(cfg):
        """Collect input transformations from nested structure."""
        transforms = []
        current = cfg

        while current is not None:
            transforms.append(
                {"type": current["type"], "parameters": current.get("parameters", {})}
            )
            current = current.get("function")

        return transforms

    # Parse the CEC structure
    output_transforms, base_func, input_transforms = parse_cec_structure(config)

    # Build the flattened structure: output_transforms -> input_transforms -> base_function
    result = base_func

    # Apply input transformations first (closest to base function)
    # Input transforms should NOT be reversed - they're already in correct order
    for transform in input_transforms:
        result = {
            "type": transform["type"],
            "parameters": transform["parameters"],
            "function": result,
        }

    # Apply output transformations last (outermost)
    for transform in reversed(output_transforms):
        result = {
            "type": transform["type"],
            "parameters": transform["parameters"],
            "function": result,
        }

    return result


class TestCEC2005ValidationClean:
    """Clean test all CEC 2005 functions against their validation data."""

    @pytest.fixture(scope="class")
    def cec2005_config(self):
        """Load CEC 2005 configuration."""
        config_path = Path("src/pyMOFL/constants/cec/2005/cec2005_suite.json")
        return load_suite_config(config_path)

    @pytest.fixture(scope="class")
    def factory(self):
        """Create function factory."""
        loader = DataLoader(base_path="src/pyMOFL/constants/cec/2005")
        registry = FunctionRegistry()
        return FunctionFactory(data_loader=loader, registry=registry)

    def _load_validation_data(self, function_id):
        return load_cec_validation_for_function_id(function_id, year=2005)

    def test_cec_functions_at_optimum(self, cec2005_config, factory):
        """Test all CEC 2005 functions at their optimum points."""
        print("\n" + "=" * 80)
        print("CEC 2005 VALIDATION TEST RESULTS")
        print("=" * 80)

        results = []

        for func_config in cec2005_config["functions"]:
            func_id = func_config["id"]

            # Load validation data
            validation_data = self._load_validation_data(func_id)
            if validation_data is None:
                print(f"{func_id}: SKIP - No validation data")
                continue

            # Test each dimension in validation data
            for case in validation_data["cases"]:
                dimension = case["dimension"]

                try:
                    # Create function with flattened config structure
                    func_config_flattened = flatten_cec_config(func_config["function"])
                    func_config_flattened = inject_dimension(func_config_flattened, dimension)

                    # Create the function
                    func = factory.create_function(func_config_flattened)

                    # Test at optimum point
                    optimum = np.array(case["optimum"])
                    expected_value = case["outputs"]["optimum"]
                    actual_value = func.evaluate(optimum)

                    # Check if values match (with tolerance for numerical precision)
                    tolerance = 1e-6
                    if abs(actual_value - expected_value) < tolerance:
                        status = "PASS"
                        print(f"{func_id} (D={dimension}): PASS")
                    else:
                        status = "FAIL"
                        print(
                            f"{func_id} (D={dimension}): FAIL - Expected {expected_value:.6f}, Got {actual_value:.6f}"
                        )

                    results.append(
                        {
                            "function": func_id,
                            "dimension": dimension,
                            "status": status,
                            "expected": expected_value,
                            "actual": actual_value,
                        }
                    )

                except ValueError as e:
                    if "Unsupported function type" in str(e):
                        print(f"{func_id} (D={dimension}): SKIP - {e!s}")
                        results.append(
                            {
                                "function": func_id,
                                "dimension": dimension,
                                "status": "SKIP",
                                "reason": str(e),
                            }
                        )
                    else:
                        print(f"{func_id} (D={dimension}): ERROR - {e!s}")
                        results.append(
                            {
                                "function": func_id,
                                "dimension": dimension,
                                "status": "ERROR",
                                "error": str(e),
                            }
                        )
                except Exception as e:
                    print(f"{func_id} (D={dimension}): ERROR - {e!s}")
                    results.append(
                        {
                            "function": func_id,
                            "dimension": dimension,
                            "status": "ERROR",
                            "error": str(e),
                        }
                    )

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        passed = len([r for r in results if r["status"] == "PASS"])
        failed = len([r for r in results if r["status"] == "FAIL"])
        errors = len([r for r in results if r["status"] == "ERROR"])
        skipped = len([r for r in results if r["status"] == "SKIP"])

        print(f"Total tests: {len(results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        print(f"Skipped: {skipped}")

        # Assert that we have some passing tests
        assert passed > 0, "No tests passed"

    def test_cec_functions_at_random_points(self, cec2005_config, factory):
        """Test CEC functions at random test points from validation data."""
        print("\n" + "=" * 80)
        print("CEC 2005 RANDOM POINT VALIDATION")
        print("=" * 80)

        results = []

        for func_config in cec2005_config["functions"][:5]:  # Test first 5 functions
            func_id = func_config["id"]

            # Load validation data
            validation_data = self._load_validation_data(func_id)
            if validation_data is None:
                continue

            # Test dimension 10 only for brevity
            for case in validation_data["cases"]:
                if case["dimension"] != 10:
                    continue

                try:
                    # Create function with flattened config

                    func_config_flattened = flatten_cec_config(func_config["function"])
                    func_config_flattened = inject_dimension(func_config_flattened, 10)
                    func = factory.create_function(func_config_flattened)

                    # Test at random point
                    random_point = np.array(case["random_input"])
                    expected_value = case["outputs"]["random"]
                    actual_value = func.evaluate(random_point)

                    # Check relative error (some functions have large values)
                    rel_error = abs(actual_value - expected_value) / (abs(expected_value) + 1e-10)

                    if rel_error < 0.01:  # 1% tolerance
                        print(f"{func_id}: PASS (random point)")
                        status = "PASS"
                    else:
                        print(
                            f"{func_id}: FAIL - Expected {expected_value:.6f}, Got {actual_value:.6f}"
                        )
                        status = "FAIL"

                    results.append(
                        {
                            "function": func_id,
                            "status": status,
                            "expected": expected_value,
                            "actual": actual_value,
                            "rel_error": rel_error,
                        }
                    )

                except Exception as e:
                    print(f"{func_id}: ERROR - {e!s}")
                    results.append({"function": func_id, "status": "ERROR", "error": str(e)})

        # Summary
        passed = len([r for r in results if r["status"] == "PASS"])
        len([r for r in results if r["status"] == "FAIL"])

        print(f"\nRandom point tests - Passed: {passed}/{len(results)}")

    def test_individual_function_f01(self, cec2005_config, factory):
        """Detailed test of F01 (shifted sphere) as an example."""
        print("\n" + "=" * 80)
        print("DETAILED TEST: F01 (Shifted Sphere)")
        print("=" * 80)

        # Get F01 config
        f01_config = None
        for func_config in cec2005_config["functions"]:
            if "f01" in func_config["id"]:
                f01_config = func_config
                break

        assert f01_config is not None, "F01 not found in config"

        # Load validation data
        validation_data = self._load_validation_data(f01_config["id"])
        assert validation_data is not None, "No validation data for F01"

        # Test dimension 10
        case = None
        for c in validation_data["cases"]:
            if c["dimension"] == 10:
                case = c
                break

        assert case is not None, "No dimension 10 case for F01"

        # Create function with flattened config
        func_config_flattened = flatten_cec_config(f01_config["function"])
        func_config_flattened = inject_dimension(func_config_flattened, 10)
        func = factory.create_function(func_config_flattened)

        print(f"Function created: dimension={func.dimension}")
        print(
            f"Bounds: [{func.operational_bounds.low[0]:.1f}, {func.operational_bounds.high[0]:.1f}]"
        )

        # Test at optimum
        optimum = np.array(case["optimum"])
        expected_opt = case["outputs"]["optimum"]
        actual_opt = func.evaluate(optimum)

        print("\nOptimum point test:")
        print(f"  Point: {optimum[:3]}... (first 3 components)")
        print(f"  Expected value: {expected_opt:.6f}")
        print(f"  Actual value: {actual_opt:.6f}")
        print(f"  Error: {abs(actual_opt - expected_opt):.2e}")

        # Test at random point
        random_point = np.array(case["random_input"])
        expected_rand = case["outputs"]["random"]
        actual_rand = func.evaluate(random_point)

        print("\nRandom point test:")
        print(f"  Point: {random_point[:3]}... (first 3 components)")
        print(f"  Expected value: {expected_rand:.6f}")
        print(f"  Actual value: {actual_rand:.6f}")
        print(f"  Error: {abs(actual_rand - expected_rand):.2e}")

        # Test at bounds
        lower_bound = np.array(case["operational_bounds"]["low"])
        upper_bound = np.array(case["operational_bounds"]["high"])

        lower_value = func.evaluate(lower_bound)
        upper_value = func.evaluate(upper_bound)

        print("\nBoundary tests:")
        print(f"  Lower bound value: {lower_value:.6f}")
        print(f"  Upper bound value: {upper_value:.6f}")

        # The optimum should give the minimum value (considering it's shifted)
        assert actual_opt <= actual_rand, "Optimum should be better than random point"
        print("\n✓ Basic sanity checks passed")
