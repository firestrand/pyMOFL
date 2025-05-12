"""
Utilities for validating function implementations against reference validation data.

This module contains functions to load validation data and compare function outputs
with reference values generated from other implementations.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

def load_cec_validation_data(year: int, function_number: int, dimensions: int = None) -> dict:
    """
    Load validation data for a CEC benchmark function.
    
    Args:
        year: The CEC benchmark year (e.g., 2005)
        function_number: The function number (1-25 for CEC2005)
        dimensions: Specific dimensionality to extract (if None, returns all available dimensions)
        
    Returns:
        A dictionary containing test points and expected function values
    """
    # Construct the path to the validation data file
    base_dir = Path(__file__).parent.parent
    file_path = base_dir / f"validation_data/cec/{year}/f{function_number:02d}.json"
    
    # Load the validation data
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Validation data for CEC{year} F{function_number} not found at {file_path}")
    
    # If dimensions specified, filter to only that dimension
    if dimensions is not None:
        dim_key = str(dimensions)
        if dim_key not in data.get("dimensions", {}):
            raise ValueError(f"Validation data for dimension {dimensions} not found in {file_path}")
        
        # Create test cases from the specified dimension
        return format_validation_data_for_dimension(data, dim_key)
    
    # Otherwise return a processed version with test cases for all dimensions
    return format_all_validation_data(data)

def format_validation_data_for_dimension(data: Dict[str, Any], dim_key: str) -> Dict[str, Any]:
    """
    Format validation data for a specific dimension.
    
    Args:
        data: The raw validation data
        dim_key: The dimension key to filter for
    
    Returns:
        Formatted validation data with test cases
    """
    result = {
        "function_id": data.get("function_id"),
        "function_name": data.get("function_name"),
        "dimension": int(dim_key),
        "test_cases": []
    }
    
    # Extract test cases from the specific dimension
    dim_data = data.get("dimensions", {}).get(dim_key, {}).get("results", {})
    
    # Add min, max, optimal points as test cases
    for case_type in ["min", "max", "optimal", "random"]:
        if case_type in dim_data:
            case = dim_data[case_type]
            result["test_cases"].append({
                "input": case.get("input_vector"),
                "expected": case.get("objective_value")
            })
    
    # Add any additional random test points if available
    if "additional_points" in dim_data:
        for point in dim_data["additional_points"]:
            result["test_cases"].append({
                "input": point.get("input_vector"),
                "expected": point.get("objective_value")
            })
    
    return result

def format_all_validation_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format validation data for all available dimensions.
    
    Args:
        data: The raw validation data
    
    Returns:
        Formatted validation data with test cases for all dimensions
    """
    result = {
        "function_id": data.get("function_id"),
        "function_name": data.get("function_name"),
        "dimensions": {}
    }
    
    # Process each dimension
    for dim_key in data.get("dimensions", {}):
        result["dimensions"][dim_key] = format_validation_data_for_dimension(
            data, dim_key
        )
    
    return result

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
        "mean_abs_error": 0.0
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
            results["failures"].append({
                "input": point.tolist(),
                "expected": expected,
                "actual": actual,
                "abs_error": abs_error
            })
    
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
    print(f"Validation Summary:")
    print(f"  Total tests: {results['total_tests']}")
    print(f"  Passed: {results['passed_tests']} ({results['passed_tests']/max(1, results['total_tests'])*100:.2f}%)")
    print(f"  Failed: {results['failed_tests']}")
    print(f"  Mean absolute error: {results['mean_abs_error']:.8e}")
    print(f"  Max absolute error: {results['max_abs_error']:.8e}")
    
    if results["failures"]:
        print("\nFirst 5 failures:")
        for i, failure in enumerate(results["failures"][:5]):
            print(f"  Test {i+1}:")
            print(f"    Input: {failure['input']}")
            print(f"    Expected: {failure['expected']}")
            print(f"    Actual: {failure['actual']}")
            print(f"    Absolute error: {failure['abs_error']:.8e}") 