# CEC2005 Validation Data

This directory contains validation data for the CEC2005 benchmark functions. The data is used to validate the Python implementations against the reference C implementation.

## Data Format

Each file (`f01.json` through `f25.json`) contains test cases for a specific function with the following structure:

```json
{
  "function": "F01 - Shifted Sphere Function",
  "dimension": 10,
  "test_cases": [
    {
      "input": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
      "expected": 12345.6789
    },
    {
      "input": [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
      "expected": 9876.5432
    }
    // More test cases...
  ]
}
```

## Usage

These validation files are used by the test suite to verify that the Python implementations of the CEC2005 functions produce the correct results. See `tests/functions/cec/test_cec2005.py` for examples of how these files are used.

## Generation

These files were generated from the reference C implementation of the CEC2005 benchmark functions available at: `/Users/firestrand/Projects/cec-benchmarks/validation_data/CEC2005` 