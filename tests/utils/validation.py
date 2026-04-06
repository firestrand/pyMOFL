"""
Utilities for validating function implementations against reference validation data.

This module contains functions to load validation data and compare function outputs
with reference values generated from other implementations.
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_CEC_FUNCTION_NUMBER_RE = re.compile(r"(?:^|_)f(\d{2})(?:_|$)")


def extract_cec_function_number(function_id: str) -> str | None:
    """Extract canonical CEC function token like ``f01`` from a function id."""

    match = _CEC_FUNCTION_NUMBER_RE.search(function_id.lower())
    if match is None:
        return None
    return f"f{int(match.group(1)):02d}"


def load_cec_validation_data(
    year: int, function_number: int, dimensions: int | None = None
) -> list[dict[str, Any]] | dict[int, list[dict[str, Any]]]:
    """
    Load validation data for a CEC benchmark function (new format: see tests/validation_data/cec/2005/README.md).
    Args:
        year: The CEC benchmark year (e.g., 2005)
        function_number: The function number (1-25 for CEC2005)
        dimensions: Specific dimensionality to extract (if None, returns all available dimensions)
    Returns:
        A list of test cases (dicts with 'input', 'expected', 'label')
    """
    # Construct the path to the validation data file
    base_dir = Path(__file__).parent.parent
    file_path = base_dir / f"validation_data/cec/{year}/f{function_number:02d}.json"

    # Load the validation data
    try:
        with file_path.open() as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Validation data for CEC{year} F{function_number} not found at {file_path}"
        ) from None

    # New format: 'cases' is a list of dicts, each with 'dimension', 'optimum', 'random_input', 'outputs'
    if dimensions is not None:
        # Find the case for the requested dimension
        for case in data.get("cases", []):
            if case["dimension"] == dimensions:
                return format_cec2005_case(case)
        raise ValueError(f"Validation data for dimension {dimensions} not found in {file_path}")
    else:
        # Return all available dimensions as a dict of lists
        all_cases: dict[int, list[dict[str, Any]]] = {}
        for case in data.get("cases", []):
            dim = case["dimension"]
            all_cases[dim] = format_cec2005_case(case)
        return all_cases


def load_cec_validation_raw(year: int, function_number: int) -> dict:
    """Load raw CEC validation JSON as a dict."""
    data = load_cec_validation_file(year, function_number)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a validation object for CEC{year} F{function_number}")
    return data


def load_cec_validation_file(year: int, function_number: int) -> dict:
    """Load the raw CEC validation JSON for a function."""
    base_dir = Path(__file__).parent.parent
    file_path = base_dir / f"validation_data/cec/{year}/f{function_number:02d}.json"
    with file_path.open() as f:
        return json.load(f)


def load_cec_validation_for_function_id(function_id: str, year: int) -> dict | None:
    """Load raw validation data for a function id like 'cec05_f01_shifted_sphere'."""
    func_num = extract_cec_function_number(function_id)
    if func_num is None:
        return None

    function_number = int(func_num[1:])
    try:
        return load_cec_validation_raw(year, function_number)
    except FileNotFoundError:
        return None


def load_cec_suite_validation_data(year: int, start: int = 1, end: int = 25) -> dict[str, dict]:
    """Load raw validation data files for a CEC suite as a numbered dictionary."""
    data: dict[str, dict] = {}
    for function_number in range(start, end + 1):
        func_num = f"f{function_number:02d}"
        try:
            data[func_num] = load_cec_validation_raw(year, function_number)
        except FileNotFoundError:
            continue
    return data


def format_cec2005_case(case: dict[str, Any]) -> list[dict[str, float | int | str]]:
    """
    Format a single CEC2005 validation case into a list of test_cases for regression testing.
    Args:
        case: The case dict for a specific dimension
    Returns:
        List of test_cases: each is a dict with {input, expected, label}
    """
    test_cases = []
    # Optimum
    test_cases.append(
        {"input": case["optimum"], "expected": case["outputs"]["optimum"], "label": "optimum"}
    )
    # Random input
    test_cases.append(
        {"input": case["random_input"], "expected": case["outputs"]["random"], "label": "random"}
    )
    # Lower bound
    lb = case["initialization_bounds"]["low"]
    test_cases.append({"input": lb, "expected": case["outputs"]["lower"], "label": "lower"})
    # Upper bound
    ub = case["initialization_bounds"]["high"]
    test_cases.append({"input": ub, "expected": case["outputs"]["upper"], "label": "upper"})
    return test_cases


def validate_function(func, validation_data: dict, rtol: float = 1e-5, atol: float = 1e-8) -> dict:
    """
    Validate a function implementation against reference data.

    Args:
        func: The function object to validate
        validation_data: Dictionary containing test points and expected values
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Dictionary with validation results including passed/failed tests and error statistics
    """
    results = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "failures": [],
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
    }

    abs_errors = []

    # Process each test point
    for test_case in validation_data.get("test_cases", []):
        point = np.array(test_case["input"])
        expected = test_case["expected"]

        # Evaluate the function
        actual = func.evaluate(point)

        # Check if values match within tolerance
        is_close = np.isclose(actual, expected, rtol=rtol, atol=atol)
        abs_error = abs(actual - expected)
        abs_errors.append(abs_error)

        results["total_tests"] += 1

        if is_close:
            results["passed_tests"] += 1
        else:
            results["failed_tests"] += 1
            results["failures"].append(
                {
                    "input": point.tolist(),
                    "expected": expected,
                    "actual": actual,
                    "abs_error": abs_error,
                }
            )

    # Calculate error statistics
    if abs_errors:
        results["max_abs_error"] = max(abs_errors)
        results["mean_abs_error"] = sum(abs_errors) / len(abs_errors)

    return results


def print_validation_summary(results: dict):
    """
    Print a summary of validation results.

    Args:
        results: Dictionary with validation results
    """
    print("Validation Summary:")
    print(f"  Total tests: {results['total_tests']}")
    print(
        f"  Passed: {results['passed_tests']} ({results['passed_tests'] / max(1, results['total_tests']) * 100:.2f}%)"
    )
    print(f"  Failed: {results['failed_tests']}")
    print(f"  Mean absolute error: {results['mean_abs_error']:.8e}")
    print(f"  Max absolute error: {results['max_abs_error']:.8e}")

    if results["failures"]:
        print("\nFirst 5 failures:")
        for i, failure in enumerate(results["failures"][:5]):
            print(f"  Test {i + 1}:")
            print(f"    Input: {failure['input']}")
            print(f"    Expected: {failure['expected']}")
            print(f"    Actual: {failure['actual']}")
            print(f"    Absolute error: {failure['abs_error']:.8e}")
