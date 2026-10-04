"""Authorized guard requests copy real configs; originals remain unchanged."""

import copy

import numpy as np
import pytest

from pyMOFL.factories.function_factory import FunctionFactory
from pyMOFL.loader import _apply_fixed_bounds, _resolve_suite_path, get_suite, load
from pyMOFL.utils.suite_config import load_suite_config


@pytest.fixture
def selected_config():
    entry = load_suite_config(_resolve_suite_path("spso2011"))["functions"][0]
    return entry["function"]


@pytest.mark.parametrize("dimension", [0, -1, True, np.bool_(True), 2.5, "2"])
def test_invalid_fixed_dimensions_preserve_config(selected_config, dimension):
    before = copy.deepcopy(selected_config)
    with pytest.raises(ValueError, match="positive integer"):
        FunctionFactory().create_function(selected_config, fixed_dimension=dimension)
    assert selected_config == before


@pytest.mark.parametrize("key", ["dim", "dimension"])
@pytest.mark.parametrize("location", ["outer", "nested"])
def test_conflicting_explicit_dimensions_preserve_config(selected_config, key, location):
    config = copy.deepcopy(selected_config)
    node = config
    if location == "nested":
        while isinstance(node.get("function"), dict):
            node = node["function"]
    node.setdefault("parameters", {})[key] = 3
    before = copy.deepcopy(config)
    with pytest.raises(ValueError, match="conflicts"):
        FunctionFactory().create_function(config, fixed_dimension=2)
    assert config == before
    assert selected_config != config


def test_composition_delegation_rejected_preserves_real_config():
    entries = load_suite_config(_resolve_suite_path("cec2014"))["functions"]
    config = next(entry["function"] for entry in entries if "f23" in entry["id"])
    before = copy.deepcopy(config)
    with pytest.raises(ValueError, match="composition delegation"):
        FunctionFactory().create_function(config, fixed_dimension=10)
    assert config == before


def test_actual_fixed_base_mismatch_preserves_config(selected_config):
    before = copy.deepcopy(selected_config)
    with pytest.raises(ValueError, match="Constructed base dimension"):
        FunctionFactory().create_function(selected_config, fixed_dimension=3)
    assert selected_config == before


def test_fixed_suite_public_dimension_and_selector_guards():
    with pytest.raises(ValueError, match="no single dimension"):
        get_suite("spso2011", dimension=2)
    with pytest.raises(ValueError, match="match the fixed suite"):
        load("spso2011_f21", suite="spso2011", dimension=2)
    with pytest.raises(ValueError, match="not found"):
        load("5", suite="spso2011")
    # Omitted dimensions retain the existing config-driven loading path.
    function = load("cec2005_f01", suite="cec2005")
    assert (
        function.dimension
        == load("cec2005_f01", suite="cec2005", dimension=function.dimension).dimension
    )


@pytest.mark.parametrize(
    "mutation", ["dimension", "low_shape", "high_shape", "steps_shape", "multiple_fractional"]
)
def test_fixed_metadata_invalid_copies_preserve_actual_function(mutation):
    entries = load_suite_config(_resolve_suite_path("spso2011"))["functions"]
    original = next(entry for entry in entries if entry["dimension"] == 3)
    entry = copy.deepcopy(original)
    function = load(original["id"], suite="spso2011")
    low, high = function.operational_bounds.low.copy(), function.operational_bounds.high.copy()
    if mutation == "dimension":
        entry["dimension"] = 2
    elif mutation == "multiple_fractional":
        entry["bounds"]["steps"][1] = 0.5
    else:
        entry["bounds"][mutation.removesuffix("_shape")].pop()
    with pytest.raises(ValueError):
        _apply_fixed_bounds(function, entry)
    np.testing.assert_array_equal(function.operational_bounds.low, low)
    np.testing.assert_array_equal(function.operational_bounds.high, high)
    assert original != entry
