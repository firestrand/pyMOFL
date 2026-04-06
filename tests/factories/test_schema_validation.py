"""Tests for benchmark suite JSON schema validation."""

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate

from pyMOFL.utils import load_suite_config

SCHEMA_PATH = Path("src/pyMOFL/constants/schemas/benchmark_suite_schema.json")
SUITE_PATH = Path("src/pyMOFL/constants/cec/2005/cec2005_suite.json")


@pytest.fixture(scope="module")
def schema():
    with SCHEMA_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def suite():
    return load_suite_config(SUITE_PATH)


def _minimal_suite(function_node: dict) -> dict:
    """Wrap a function node in a minimal valid suite structure."""
    return {
        "suite_id": "test",
        "name": "test",
        "description": "test",
        "functions": [
            {
                "id": "test_f1",
                "function": function_node,
                "dimensions": {
                    "supported": [10],
                    "default": 10,
                    "custom_dimensions": {"min": 2, "step": 1},
                },
                "search_space": {"default_bounds": {"min": -100, "max": 100}},
            }
        ],
    }


class TestCEC2005SuiteValidation:
    """The existing CEC 2005 suite must validate without modification."""

    def test_full_suite_validates(self, schema, suite):
        validate(instance=suite, schema=schema)

    def test_all_25_functions_present(self, suite):
        assert len(suite["functions"]) == 25


class TestTypoDetection:
    """Schema should reject typos in parameter names for known types."""

    @pytest.mark.parametrize(
        "func_type, bad_params, description",
        [
            ("bias", {"valu": -450}, "typo in 'value'"),
            ("shift", {"vectr": "f01/shift.txt"}, "typo in 'vector'"),
            ("rotate", {"matrx": "f03/rot.txt"}, "typo in 'matrix'"),
            (
                "noise",
                {"modle": "multiplicative", "factor": 0.4, "distribution": "normal"},
                "typo in 'model'",
            ),
            ("weierstrass", {"a": 0.5, "b": 3, "kmax": 20}, "typo in 'k_max'"),
            (
                "schwefel_2_13",
                {"ma": "a.txt", "b": "b.txt", "alpha": "a.txt"},
                "typo in 'a'",
            ),
        ],
    )
    def test_parameter_typo_rejected(self, schema, func_type, bad_params, description):
        suite = _minimal_suite({"type": func_type, "parameters": bad_params})
        with pytest.raises(ValidationError):
            validate(instance=suite, schema=schema)

    def test_extra_parameter_rejected(self, schema):
        suite = _minimal_suite({"type": "bias", "parameters": {"value": -450, "extra": True}})
        with pytest.raises(ValidationError):
            validate(instance=suite, schema=schema)


class TestMissingRequiredParams:
    """Schema should reject missing required parameters."""

    def test_bias_requires_value(self, schema):
        suite = _minimal_suite({"type": "bias", "parameters": {}})
        with pytest.raises(ValidationError):
            validate(instance=suite, schema=schema)

    def test_shift_requires_vector(self, schema):
        suite = _minimal_suite({"type": "shift", "parameters": {}})
        with pytest.raises(ValidationError):
            validate(instance=suite, schema=schema)

    def test_weierstrass_requires_all_three(self, schema):
        suite = _minimal_suite({"type": "weierstrass", "parameters": {"a": 0.5, "b": 3}})
        with pytest.raises(ValidationError):
            validate(instance=suite, schema=schema)

    def test_composition_requires_functions_array(self, schema):
        suite = _minimal_suite(
            {
                "type": "composition",
                "parameters": {
                    "num_functions": 2,
                    "dominance_suppression": True,
                    "shift_file": "test.txt",
                    "lambdas": [1.0],
                    "sigmas": [1.0],
                    "biases": [0.0],
                    "C": 2000.0,
                },
            }
        )
        with pytest.raises(ValidationError):
            validate(instance=suite, schema=schema)

    def test_composition_requires_core_params(self, schema):
        suite = _minimal_suite(
            {
                "type": "composition",
                "parameters": {"num_functions": 2},
                "functions": [{"type": "sphere"}],
            }
        )
        with pytest.raises(ValidationError):
            validate(instance=suite, schema=schema)


class TestExtensibility:
    """Unknown types should pass validation (future suite support)."""

    def test_unknown_type_accepted(self, schema):
        suite = _minimal_suite({"type": "zakharov", "parameters": {"custom": 42}})
        validate(instance=suite, schema=schema)

    def test_unknown_type_no_params_accepted(self, schema):
        suite = _minimal_suite({"type": "bent_cigar"})
        validate(instance=suite, schema=schema)


class TestValidConfigs:
    """Well-formed configs for each known type should validate."""

    def test_bias(self, schema):
        suite = _minimal_suite({"type": "bias", "parameters": {"value": -450}})
        validate(instance=suite, schema=schema)

    def test_shift(self, schema):
        suite = _minimal_suite({"type": "shift", "parameters": {"vector": "f01/shift.txt"}})
        validate(instance=suite, schema=schema)

    def test_shift_with_bounds_mapping(self, schema):
        suite = _minimal_suite(
            {
                "type": "shift",
                "parameters": {"vector": "f08/shift.txt", "bounds_mapping": "alternate"},
            }
        )
        validate(instance=suite, schema=schema)

    def test_shift_with_alias_shift_key(self, schema):
        suite = _minimal_suite(
            {
                "type": "shift",
                "parameters": {"shift": "f01/shift.txt"},
            }
        )
        validate(instance=suite, schema=schema)

    def test_rotate(self, schema):
        suite = _minimal_suite({"type": "rotate", "parameters": {"matrix": "f03/rot.txt"}})
        validate(instance=suite, schema=schema)

    def test_rotate_with_alias_rotation_key(self, schema):
        suite = _minimal_suite(
            {
                "type": "rotate",
                "parameters": {"rotation": "f07/rot.txt"},
            }
        )
        validate(instance=suite, schema=schema)

    def test_rotate_with_condition_and_noise(self, schema):
        suite = _minimal_suite(
            {
                "type": "rotate",
                "parameters": {"matrix": "f07/rot.txt", "condition": 3, "noise": 0.3},
            }
        )
        validate(instance=suite, schema=schema)

    def test_offset(self, schema):
        suite = _minimal_suite({"type": "offset", "parameters": {"value": 1}})
        validate(instance=suite, schema=schema)

    def test_noise(self, schema):
        suite = _minimal_suite(
            {
                "type": "noise",
                "parameters": {
                    "model": "multiplicative",
                    "factor": 0.4,
                    "distribution": "absolute_normal",
                },
            }
        )
        validate(instance=suite, schema=schema)

    def test_noise_with_level_alias(self, schema):
        suite = _minimal_suite(
            {
                "type": "noise",
                "parameters": {
                    "model": "multiplicative",
                    "level": 0.4,
                    "distribution": "absolute_normal",
                },
            }
        )
        validate(instance=suite, schema=schema)

    def test_non_continuous_no_params(self, schema):
        suite = _minimal_suite({"type": "non_continuous"})
        validate(instance=suite, schema=schema)

    def test_non_continuous_empty_params(self, schema):
        suite = _minimal_suite({"type": "non_continuous", "parameters": {}})
        validate(instance=suite, schema=schema)

    def test_sphere_no_params(self, schema):
        suite = _minimal_suite({"type": "sphere"})
        validate(instance=suite, schema=schema)

    def test_sphere_empty_params(self, schema):
        suite = _minimal_suite({"type": "sphere", "parameters": {}})
        validate(instance=suite, schema=schema)

    def test_sphere_with_dimension(self, schema):
        suite = _minimal_suite({"type": "sphere", "parameters": {"dimension": 10}})
        validate(instance=suite, schema=schema)

    def test_high_conditioned_elliptic(self, schema):
        suite = _minimal_suite(
            {
                "type": "high_conditioned_elliptic",
                "parameters": {"condition": 1000000.0},
            }
        )
        validate(instance=suite, schema=schema)

    def test_schwefel_2_6(self, schema):
        suite = _minimal_suite(
            {
                "type": "schwefel_2_6",
                "parameters": {
                    "A": "f05/matrix_data_A_D50.txt",
                    "B": "f05/vector_B_D10.txt",
                },
            }
        )
        validate(instance=suite, schema=schema)

    def test_schwefel_2_13(self, schema):
        suite = _minimal_suite(
            {
                "type": "schwefel_2_13",
                "parameters": {"a": "a.txt", "b": "b.txt", "alpha": "alpha.txt"},
            }
        )
        validate(instance=suite, schema=schema)

    def test_weierstrass(self, schema):
        suite = _minimal_suite(
            {
                "type": "weierstrass",
                "parameters": {"a": 0.5, "b": 3, "k_max": 20},
            }
        )
        validate(instance=suite, schema=schema)

    def test_composition_minimal(self, schema):
        suite = _minimal_suite(
            {
                "type": "composition",
                "parameters": {
                    "num_functions": 2,
                    "dominance_suppression": True,
                    "shift_file": "test.txt",
                    "lambdas": [1.0, 1.0],
                    "sigmas": [1.0, 1.0],
                    "biases": [0.0, 100.0],
                    "C": 2000.0,
                },
                "functions": [{"type": "sphere"}, {"type": "rastrigin"}],
            }
        )
        validate(instance=suite, schema=schema)

    def test_composition_all_optional_params(self, schema):
        suite = _minimal_suite(
            {
                "type": "composition",
                "parameters": {
                    "num_functions": 2,
                    "dominance_suppression": True,
                    "shift_file": "test.txt",
                    "rotation_file": "rot.txt",
                    "lambdas": [1.0, 1.0],
                    "sigmas": [1.0, 1.0],
                    "biases": [0.0, 100.0],
                    "C": 2000.0,
                    "zero_last_optimum": True,
                    "optimum_pattern": "alternate_odds",
                    "non_continuous": True,
                    "component_noise": True,
                    "reference_point": 5.0,
                    "global_bias": 0.0,
                },
                "functions": [{"type": "sphere"}, {"type": "rastrigin"}],
            }
        )
        validate(instance=suite, schema=schema)

    def test_composition_with_alias_optima_file(self, schema):
        suite = _minimal_suite(
            {
                "type": "composition",
                "parameters": {
                    "num_functions": 2,
                    "dominance_suppression": False,
                    "optima_file": "test.txt",
                    "lambdas": [1.0, 1.0],
                    "sigmas": [1.0, 1.0],
                    "biases": [0.0, 100.0],
                    "C": 2000.0,
                },
                "functions": [{"type": "sphere"}, {"type": "rastrigin"}],
            }
        )
        validate(instance=suite, schema=schema)

    def test_composition_null_rotation(self, schema):
        suite = _minimal_suite(
            {
                "type": "composition",
                "parameters": {
                    "num_functions": 1,
                    "dominance_suppression": False,
                    "shift_file": "test.txt",
                    "rotation_file": None,
                    "lambdas": [1.0],
                    "sigmas": [1.0],
                    "biases": [0.0],
                    "C": 2000.0,
                },
                "functions": [{"type": "sphere"}],
            }
        )
        validate(instance=suite, schema=schema)


class TestNestedFunctions:
    """Nested function configs should validate recursively."""

    def test_bias_wrapping_sphere(self, schema):
        suite = _minimal_suite(
            {
                "type": "bias",
                "parameters": {"value": -450},
                "function": {"type": "sphere"},
            }
        )
        validate(instance=suite, schema=schema)

    def test_deep_nesting(self, schema):
        suite = _minimal_suite(
            {
                "type": "bias",
                "parameters": {"value": -450},
                "function": {
                    "type": "sphere",
                    "parameters": {},
                    "function": {
                        "type": "shift",
                        "parameters": {"vector": "f01/shift.txt"},
                    },
                },
            }
        )
        validate(instance=suite, schema=schema)

    def test_non_continuous_wrapping_function(self, schema):
        suite = _minimal_suite(
            {
                "type": "non_continuous",
                "function": {"type": "rastrigin"},
            }
        )
        validate(instance=suite, schema=schema)
