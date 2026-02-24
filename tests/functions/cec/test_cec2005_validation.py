"""
Test all CEC 2005 functions against their validation data.
"""

from pathlib import Path

import numpy as np
import pytest

from pyMOFL.factories import BenchmarkFactory
from pyMOFL.utils import inject_dimension, load_suite_config
from tests.utils.validation import (
    extract_cec_function_number,
    load_cec_validation_for_function_id,
    load_cec_validation_raw,
)


class TestCEC2005Validation:
    """Test all CEC 2005 functions against their validation data."""

    @pytest.fixture
    def cec2005_config(self):
        """Load CEC 2005 configuration."""
        config_path = Path("src/pyMOFL/constants/cec/2005/cec2005_suite.json")
        return load_suite_config(config_path)

    @pytest.fixture
    def factory(self):
        """Create benchmark factory."""
        return BenchmarkFactory(data_path="src/pyMOFL/constants/cec/2005")

    def test_all_cec2005_functions_at_optimum(self, cec2005_config, factory):
        """Test all CEC 2005 functions at their optimum points."""
        results = []

        for func_config in cec2005_config["functions"]:
            func_id = func_config["id"]

            # Load validation data
            validation_data = load_cec_validation_for_function_id(func_id, year=2005)
            if validation_data is None:
                results.append(
                    {"function": func_id, "status": "SKIP", "reason": "No validation data"}
                )
                continue

            # Test each dimension in validation data
            for case in validation_data["cases"]:
                dimension = case["dimension"]

                try:
                    # Create function using BenchmarkFactory
                    # Ensure dimension is propagated to nested nodes for consistency
                    func_config_with_dim = inject_dimension(func_config["function"], dimension)
                    func = factory.create_function(func_config_with_dim, dimension=dimension)

                    # Check if function has noise
                    has_noise = self._has_noise(func_config["function"])

                    # Test points
                    test_points = {
                        "optimum": np.array(case["optimum"]),
                        "random": np.array(case["random_input"]),
                        "lower": np.array(case["operational_bounds"]["low"]),
                        "upper": np.array(case["operational_bounds"]["high"]),
                    }

                    # Expected values
                    expected_values = case["outputs"]
                    from tests.utils.tolerance import ToleranceChecker

                    tol = ToleranceChecker()

                    # Test each point
                    all_passed = True
                    point_results = []

                    for point_name, point in test_points.items():
                        # Skip non-optimum points for functions with noise
                        if has_noise and point_name != "optimum":
                            point_results.append(
                                {
                                    "point": point_name,
                                    "actual": "SKIPPED",
                                    "expected": expected_values[point_name],
                                    "passed": True,
                                    "reason": "Function has noise",
                                }
                            )
                            continue

                        expected = expected_values[point_name]
                        actual = func.evaluate(point)
                        # Check with magnitude-aware tolerances; treat optimum as deterministic
                        is_close = tol.check(
                            expected,
                            actual,
                            is_noisy=has_noise,
                            is_optimal=(point_name == "optimum"),
                        )

                        if not is_close:
                            all_passed = False

                        point_results.append(
                            {
                                "point": point_name,
                                "actual": actual,
                                "expected": expected,
                                "passed": is_close,
                            }
                        )

                    # Create summary message
                    if all_passed:
                        status = "PASS"
                        message = "All points passed"
                    else:
                        status = "FAIL"
                        failed_points = [r for r in point_results if not r["passed"]]
                        message = f"{len(failed_points)} points failed: "
                        for r in failed_points:
                            message += f"{r['point']}(diff={abs(r['actual'] - r['expected']):.2e}) "

                    results.append(
                        {
                            "function": func_id,
                            "dimension": dimension,
                            "status": status,
                            "message": message,
                            "details": point_results,
                        }
                    )

                except Exception as e:
                    results.append(
                        {
                            "function": func_id,
                            "dimension": dimension,
                            "status": "ERROR",
                            "message": str(e),
                        }
                    )

        # Print results summary
        print("\n" + "=" * 80)
        print("CEC 2005 VALIDATION TEST RESULTS")
        print("=" * 80)

        passed = 0
        failed = 0
        errors = 0
        skipped = 0

        for result in results:
            if "dimension" in result:
                print(
                    f"{result['function']} (D={result['dimension']}): {result['status']} - {result['message']}"
                )
            else:
                print(f"{result['function']}: {result['status']} - {result['reason']}")

            if result["status"] == "PASS":
                passed += 1
            elif result["status"] == "FAIL":
                failed += 1
            elif result["status"] == "ERROR":
                errors += 1
            elif result["status"] == "SKIP":
                skipped += 1

        print("\n" + "-" * 80)
        print(f"SUMMARY: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped")
        print("-" * 80)

        # Assert all tests passed (excluding skipped)
        assert failed == 0 and errors == 0, "Some tests failed or had errors"

    def _has_noise(self, func_config):
        """Check if a function configuration contains noise.

        Detects both outer-level noise wrappers (e.g., F4, F17) and
        composition-level ``component_noise`` flags (e.g., F24, F25
        whose 10th component is Sphere with Noise in Fitness).
        """
        if isinstance(func_config, dict):
            if func_config.get("type") == "noise":
                return True
            # Check for component_noise flag in composition parameters
            if func_config.get("type") == "composition" and func_config.get("parameters", {}).get(
                "component_noise", False
            ):
                return True
            # Recursively check nested function
            if "function" in func_config:
                return self._has_noise(func_config["function"])
        return False

    def test_all_cec2005_functions_comprehensive(self):
        """Test all CEC 2005 functions at all validation points with detailed output."""
        # Load CEC 2005 suite configuration
        suite_data = load_suite_config("src/pyMOFL/constants/cec/2005/cec2005_suite.json")

        # Create factory
        from pyMOFL.factories import FunctionFactory
        from pyMOFL.factories.function_factory import DataLoader, FunctionRegistry

        loader = DataLoader(base_path="src/pyMOFL/constants/cec/2005")
        registry = FunctionRegistry()
        factory = FunctionFactory(data_loader=loader, registry=registry)

        results = []

        # Test each function
        for func_config in suite_data["functions"]:
            func_id = func_config["id"]

            # Load validation data
            function_num = extract_cec_function_number(func_id)
            if function_num is None:
                continue
            try:
                validation_data = load_cec_validation_raw(2005, int(function_num[1:]))
            except FileNotFoundError:
                continue

            # Test dimension 10 only for detailed view
            for case in validation_data["cases"]:
                if case["dimension"] != 10:
                    continue

                dimension = case["dimension"]

                try:
                    # Create function with dimension embedded in config
                    func_config_with_dim = inject_dimension(func_config["function"], dimension)
                    func = factory.create_function(func_config_with_dim)

                    # Test points
                    test_points = {
                        "optimum": np.array(case["optimum"]),
                        "random": np.array(case["random_input"]),
                        "lower": np.array(case["operational_bounds"]["low"]),
                        "upper": np.array(case["operational_bounds"]["high"]),
                    }

                    # Expected values
                    expected_values = case["outputs"]

                    # Check if function has noise
                    has_noise = self._has_noise(func_config["function"])

                    # Test each point
                    func_results = {
                        "function": func_id,
                        "dimension": dimension,
                        "points": {},
                        "has_noise": has_noise,
                    }

                    for point_name, point in test_points.items():
                        # Skip non-optimum points for functions with noise
                        if has_noise and point_name != "optimum":
                            func_results["points"][point_name] = {
                                "actual": "SKIPPED",
                                "expected": expected_values[point_name],
                                "diff": None,
                                "passed": True,  # Consider skipped as passed
                                "reason": "Function has noise",
                            }
                            continue

                        expected = expected_values[point_name]
                        actual = func.evaluate(point)
                        is_close = np.isclose(actual, expected, rtol=1e-6, atol=1e-10)

                        func_results["points"][point_name] = {
                            "actual": actual,
                            "expected": expected,
                            "diff": abs(actual - expected),
                            "passed": is_close,
                        }

                    results.append(func_results)

                except Exception as e:
                    results.append({"function": func_id, "dimension": dimension, "error": str(e)})

        # Print detailed results
        print("\n" + "=" * 100)
        print("CEC 2005 COMPREHENSIVE VALIDATION (D=10)")
        print("=" * 100)

        for result in results:
            if "error" in result:
                print(f"\n{result['function']}: ERROR - {result['error']}")
                continue

            func_id = result["function"]
            function_num = extract_cec_function_number(func_id)
            if function_num is None:
                continue
            func_num = int(function_num[1:])

            all_passed = all(p["passed"] for p in result["points"].values())
            status = "PASS" if all_passed else "FAIL"

            print(f"\nF{func_num:02d}: {status}")
            print(f"{'Point':<10} {'Actual':>12} {'Expected':>12} {'Diff':>12} {'Status':>8}")
            print("-" * 60)

            for point_name, point_data in result["points"].items():
                status = "PASS" if point_data["passed"] else "FAIL"
                if point_data["actual"] == "SKIPPED":
                    print(
                        f"{point_name:<10} {'SKIPPED':>12} {point_data['expected']:12.6f} "
                        f"{'N/A':>12} {status:>8} (noise)"
                    )
                else:
                    print(
                        f"{point_name:<10} {point_data['actual']:12.6f} {point_data['expected']:12.6f} "
                        f"{point_data['diff']:12.2e} {status:>8}"
                    )
