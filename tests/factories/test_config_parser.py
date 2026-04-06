"""Tests for ConfigParser — nested JSON config parsing."""

import pytest

from pyMOFL.factories.config_parser import ConfigParser

# Known base types matching what FunctionRegistry currently supports
KNOWN_BASES = frozenset(
    {
        "sphere",
        "ackley",
        "rastrigin",
        "griewank",
        "weierstrass",
        "rosenbrock",
        "elliptic",
        "high_conditioned_elliptic",
        "schwefel_1_2",
        "schwefel_2_6",
        "schwefel_2_13",
        "schaffer_f6_expanded",
        "griewank_of_rosenbrock",
    }
)


@pytest.fixture
def parser():
    return ConfigParser(known_base_types=KNOWN_BASES)


class TestParseSimple:
    """Test parsing simple (non-composition) configs."""

    def test_bare_base_function(self, parser):
        """A config with just a base function and no transforms."""
        config = {"type": "sphere", "parameters": {"dimension": 10}}
        result = parser.parse(config)

        assert result.base_type == "sphere"
        assert result.base_params["dimension"] == 10
        assert result.transforms == []
        assert result.is_composition is False
        assert result.raw_composition_config is None

    def test_single_transform_wrapping_base(self, parser):
        """bias(sphere(x)) — one output transform."""
        config = {
            "type": "bias",
            "parameters": {"value": -450},
            "function": {"type": "sphere", "parameters": {"dimension": 10}},
        }
        result = parser.parse(config)

        assert result.base_type == "sphere"
        assert result.base_params["dimension"] == 10
        assert len(result.transforms) == 1
        assert result.transforms[0] == ("bias", {"value": -450})

    def test_nested_transforms_application_order(self, parser):
        """bias(sphere(rotate(shift(x)))) — transforms in application order."""
        config = {
            "type": "bias",
            "parameters": {"value": -450},
            "function": {
                "type": "sphere",
                "parameters": {"dimension": 10},
                "function": {
                    "type": "rotate",
                    "parameters": {"matrix": "rot.txt"},
                    "function": {
                        "type": "shift",
                        "parameters": {"vector": "shift.txt"},
                    },
                },
            },
        }
        result = parser.parse(config)

        assert result.base_type == "sphere"
        # Application order: shift first, then rotate, then bias
        types = [t for t, _ in result.transforms]
        assert types == ["shift", "rotate", "bias"]

    def test_cec_f01_shifted_sphere(self, parser):
        """CEC F01: bias(sphere(shift(x)))."""
        config = {
            "type": "bias",
            "parameters": {"value": -450},
            "function": {
                "type": "sphere",
                "parameters": {},
                "function": {
                    "type": "shift",
                    "parameters": {"vector": "f01/vector_shift_D50.txt"},
                },
            },
        }
        result = parser.parse(config)

        assert result.base_type == "sphere"
        types = [t for t, _ in result.transforms]
        assert types == ["shift", "bias"]

    def test_cec_f06_shifted_rosenbrock_with_offset(self, parser):
        """CEC F06: bias(rosenbrock(offset(shift(x))))."""
        config = {
            "type": "bias",
            "parameters": {"value": 390},
            "function": {
                "type": "rosenbrock",
                "parameters": {},
                "function": {
                    "type": "offset",
                    "parameters": {"value": 1},
                    "function": {
                        "type": "shift",
                        "parameters": {"vector": "f06/vector_shift_D50.txt"},
                    },
                },
            },
        }
        result = parser.parse(config)

        assert result.base_type == "rosenbrock"
        types = [t for t, _ in result.transforms]
        assert types == ["shift", "offset", "bias"]

    def test_weight_nodes_are_skipped(self, parser):
        """Weight nodes are transparent wrappers, skipped during parsing."""
        config = {
            "type": "weight",
            "parameters": {"sigma": 1.0},
            "function": {"type": "sphere", "parameters": {"dimension": 5}},
        }
        result = parser.parse(config)

        assert result.base_type == "sphere"
        assert result.transforms == []


class TestParseComposition:
    """Test parsing composition configs."""

    def test_composition_detected(self, parser):
        """Composition nodes are detected and returned as raw config."""
        config = {
            "type": "composition",
            "parameters": {"num_functions": 10},
            "functions": [
                {"type": "weight", "function": {"type": "sphere", "parameters": {"dimension": 2}}}
            ],
        }
        result = parser.parse(config)

        assert result.is_composition is True
        assert result.base_type == "composition"
        assert result.raw_composition_config is not None
        assert result.raw_composition_config["type"] == "composition"

    def test_composition_with_outer_bias(self, parser):
        """bias(composition{...}) — outer transform collected."""
        config = {
            "type": "bias",
            "parameters": {"value": 120},
            "function": {
                "type": "composition",
                "parameters": {"num_functions": 10},
                "functions": [],
            },
        }
        result = parser.parse(config)

        assert result.is_composition is True
        assert result.base_type == "composition"
        # The bias is an outer transform
        types = [t for t, _ in result.transforms]
        assert types == ["bias"]


class TestDimensionExtraction:
    """Test dimension extraction from config trees."""

    def test_dimension_from_base_params(self, parser):
        config = {"type": "sphere", "parameters": {"dimension": 10}}
        result = parser.parse(config)
        assert result.base_params["dimension"] == 10

    def test_dimension_propagated_from_inner_level(self, parser):
        """Dimension found at an inner level is propagated to base_params."""
        config = {
            "type": "bias",
            "parameters": {"value": -450},
            "function": {
                "type": "sphere",
                "parameters": {},
                "function": {
                    "type": "shift",
                    "parameters": {"dimension": 30, "vector": "shift.txt"},
                },
            },
        }
        result = parser.parse(config)
        assert result.base_params.get("dimension") == 30

    def test_extract_dimension_static(self):
        config = {
            "type": "bias",
            "parameters": {},
            "function": {
                "type": "sphere",
                "parameters": {"dimension": 50},
            },
        }
        assert ConfigParser.extract_dimension(config) == 50

    def test_extract_dimension_nested(self):
        config = {
            "type": "bias",
            "parameters": {},
            "function": {
                "type": "shift",
                "parameters": {},
                "function": {
                    "type": "sphere",
                    "parameters": {"dimension": 10},
                },
            },
        }
        assert ConfigParser.extract_dimension(config) == 10

    def test_extract_dimension_returns_none_if_missing(self):
        config = {"type": "sphere", "parameters": {}}
        assert ConfigParser.extract_dimension(config) is None

    def test_extract_dimension_from_functions_array(self):
        config = {
            "type": "composition",
            "parameters": {},
            "functions": [
                {
                    "type": "weight",
                    "function": {"type": "sphere", "parameters": {"dimension": 20}},
                }
            ],
        }
        assert ConfigParser.extract_dimension(config) == 20


class TestCaseInsensitivity:
    """Test that type names are case-insensitive in configs."""

    def test_base_type_uppercase(self, parser):
        """PascalCase/uppercase base type names should be recognized."""
        config = {"type": "Sphere", "parameters": {"dimension": 5}}
        result = parser.parse(config)
        assert result.base_type == "sphere"

    def test_base_type_mixed_case(self, parser):
        config = {"type": "ACKLEY", "parameters": {"dimension": 5}}
        result = parser.parse(config)
        assert result.base_type == "ackley"

    def test_transform_type_uppercase(self, parser):
        """Transform type names should also be case-insensitive."""
        config = {
            "type": "Bias",
            "parameters": {"value": -100},
            "function": {"type": "sphere", "parameters": {"dimension": 5}},
        }
        result = parser.parse(config)
        assert result.base_type == "sphere"
        assert len(result.transforms) == 1
        # The transform type should be stored lowercase
        assert result.transforms[0][0] == "bias"

    def test_composition_type_uppercase(self, parser):
        config = {
            "type": "Composition",
            "parameters": {"num_functions": 1},
            "functions": [
                {"type": "weight", "function": {"type": "sphere", "parameters": {"dimension": 2}}}
            ],
        }
        result = parser.parse(config)
        assert result.is_composition is True


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_config(self, parser):
        result = parser.parse({})
        assert result.base_type is None
        assert result.transforms == []

    def test_no_base_function_found(self, parser):
        """Config with only transforms and no recognized base function."""
        config = {
            "type": "shift",
            "parameters": {"vector": [1, 2, 3]},
        }
        result = parser.parse(config)
        assert result.base_type is None
        assert len(result.transforms) == 1

    def test_dimension_defaults_to_2_when_missing(self, parser):
        """When no dimension is found anywhere, default to 2."""
        config = {
            "type": "bias",
            "parameters": {"value": 0},
            "function": {"type": "sphere", "parameters": {}},
        }
        result = parser.parse(config)
        assert result.base_params.get("dimension") == 2
