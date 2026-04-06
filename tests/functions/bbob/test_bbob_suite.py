"""
Tests for BBOB suite integration: JSON parsing, factory construction, and
config-driven vs factory-driven parity.

These tests validate:
1. Suite JSON is parseable and all 24 entries findable
2. All 24 functions instantiate for standard dimensions via BBOBSuiteFactory
3. Factory configs are ConfigParser-compatible
4. Config-driven and factory-driven construction agree
5. Template structure in bbob_suite.json matches factory-generated config
"""

import re
from pathlib import Path

import numpy as np
import pytest

from pyMOFL.factories.bbob_suite_factory import BBOBSuiteFactory
from pyMOFL.factories.function_factory import FunctionFactory
from pyMOFL.functions.transformations.composed import ComposedFunction
from pyMOFL.utils.suite_config import find_suite_function_config, load_suite_config

BBOB_SUITE_PATH = (
    Path(__file__).parents[3] / "src" / "pyMOFL" / "constants" / "bbob" / "bbob_suite.json"
)
STANDARD_DIMS = [2, 3, 5, 10, 20, 40]
ALL_FIDS = list(range(1, 25))


def _fid_from_string_id(string_id: str) -> int:
    match = re.search(r"f(\d{2})", string_id)
    assert match is not None
    return int(match.group(1))


def _extract_nesting_types(config: dict) -> list[str]:
    """Extract the sequence of 'type' values from a nested config, outermost first."""
    types = []
    node = config
    while isinstance(node, dict) and "type" in node:
        types.append(node["type"])
        node = node.get("function", {})
    return types


@pytest.fixture(scope="module")
def suite_data():
    return load_suite_config(BBOB_SUITE_PATH)


@pytest.fixture(scope="module")
def factory():
    return BBOBSuiteFactory()


# --- 1. Suite JSON is parseable ---


class TestSuiteJSONParseable:
    def test_load_succeeds(self, suite_data):
        assert isinstance(suite_data, dict)
        assert suite_data["suite_id"] == "bbob_noiseless"

    def test_all_24_entries_found_by_string_id(self, suite_data):
        """All 24 functions should be findable via their string IDs."""
        for func_entry in suite_data["functions"]:
            string_id = func_entry["id"]
            config = find_suite_function_config(suite_data, string_id)
            assert isinstance(config, dict)
            assert "type" in config

    def test_all_24_entries_found_by_numeric_code(self, suite_data):
        """All 24 functions should be findable by bare number string."""
        for fid in ALL_FIDS:
            config = find_suite_function_config(suite_data, str(fid))
            assert isinstance(config, dict)

    def test_all_24_entries_found_by_fcode(self, suite_data):
        """All 24 functions should be findable by fNN code."""
        for fid in ALL_FIDS:
            config = find_suite_function_config(suite_data, f"f{fid:02d}")
            assert isinstance(config, dict)


# --- 2. All 24 functions instantiate for standard dims ---


class TestAllFunctionsInstantiate:
    @pytest.mark.parametrize("dim", STANDARD_DIMS)
    def test_all_24_instantiate(self, factory, dim):
        """All 24 functions should instantiate as ComposedFunction."""
        for fid in ALL_FIDS:
            func = factory.create_function(fid=fid, iid=1, dim=dim)
            assert isinstance(func, ComposedFunction), f"f{fid} dim={dim}"
            assert func.dimension == dim, f"f{fid} dim mismatch"

    @pytest.mark.parametrize("dim", STANDARD_DIMS)
    def test_all_24_evaluate_finitely(self, factory, dim):
        """All 24 functions should produce finite values at random points."""
        rng = np.random.default_rng(42)
        for fid in ALL_FIDS:
            func = factory.create_function(fid=fid, iid=1, dim=dim)
            x = rng.uniform(-4, 4, size=dim)
            val = func.evaluate(x)
            assert np.isfinite(val), f"f{fid} dim={dim} non-finite: {val}"


# --- 3. Factory configs are ConfigParser-compatible ---


class TestFactoryConfigsAreParserCompatible:
    @pytest.fixture
    def parser(self):
        from pyMOFL.factories.config_parser import ConfigParser

        ff = FunctionFactory()
        return ConfigParser(frozenset(ff.registry.base_functions))

    def test_build_config_returns_dict(self, factory):
        """build_config should return a nested dict for each fid."""
        for fid in ALL_FIDS:
            config = factory.build_config(fid=fid, iid=1, dim=10)
            assert isinstance(config, dict)
            assert "type" in config

    def test_build_config_parseable_by_config_parser(self, factory, parser):
        """ConfigParser should successfully parse each factory config."""
        for fid in ALL_FIDS:
            config = factory.build_config(fid=fid, iid=1, dim=10)
            parsed = parser.parse(config)
            assert parsed.base_type is not None, f"f{fid}: base_type is None"
            assert not parsed.is_composition, f"f{fid}: unexpectedly a composition"

    def test_parsed_config_has_correct_base_type(self, factory, parser):
        """Parsed base_type should match the expected BBOB base function."""
        from pyMOFL.factories.bbob_suite_factory import _BBOB_INFO

        for fid in ALL_FIDS:
            config = factory.build_config(fid=fid, iid=1, dim=10)
            parsed = parser.parse(config)
            expected = _BBOB_INFO[fid]["base_function"]
            assert parsed.base_type == expected, (
                f"f{fid}: parsed base_type={parsed.base_type!r}, expected={expected!r}"
            )

    def test_parsed_config_has_transforms(self, factory, parser):
        """Parsed config should have at least one transform for every function."""
        for fid in ALL_FIDS:
            config = factory.build_config(fid=fid, iid=1, dim=10)
            parsed = parser.parse(config)
            assert len(parsed.transforms) > 0, f"f{fid}: no transforms"


# --- 4. Config-driven and factory-driven construction agree ---


class TestConfigVsFactoryParity:
    @pytest.mark.parametrize("fid", [1, 5, 8, 10, 15, 20, 24])
    def test_factory_vs_function_factory_agree(self, factory, fid):
        """FunctionFactory.create_function(build_config(...)) should produce
        the same values as BBOBSuiteFactory.create_function(fid, iid, dim)."""
        from pyMOFL.factories.function_factory import FunctionFactory

        dim = 10
        iid = 1

        # Factory-driven
        factory_func = factory.create_function(fid=fid, iid=iid, dim=dim)

        # Config-driven
        config = factory.build_config(fid=fid, iid=iid, dim=dim)
        config_func = FunctionFactory().create_function(config)

        # Evaluate at 5 random points
        rng = np.random.default_rng(fid * 100 + dim)
        for _ in range(5):
            x = rng.uniform(-4, 4, size=dim)
            factory_val = factory_func.evaluate(x)
            config_val = config_func.evaluate(x)
            np.testing.assert_allclose(
                factory_val,
                config_val,
                atol=1e-12,
                rtol=1e-12,
                err_msg=f"f{fid}: factory vs config mismatch",
            )


# --- 5. Template structure matches factory-generated config ---


class TestTemplateStructureMatches:
    def test_nesting_depth_matches(self, factory, suite_data):
        """Template nesting depth should match factory config depth."""
        for func_entry in suite_data["functions"]:
            fid = _fid_from_string_id(func_entry["id"])
            template = func_entry["function"]
            factory_config = factory.build_config(fid=fid, iid=1, dim=10)

            template_types = _extract_nesting_types(template)
            factory_types = _extract_nesting_types(factory_config)

            assert len(template_types) == len(factory_types), (
                f"f{fid}: template depth {len(template_types)} != "
                f"factory depth {len(factory_types)}\n"
                f"  template: {template_types}\n"
                f"  factory:  {factory_types}"
            )

    def test_transform_type_order_matches(self, factory, suite_data):
        """Template transform types should be in the same order as factory config."""
        for func_entry in suite_data["functions"]:
            fid = _fid_from_string_id(func_entry["id"])
            template = func_entry["function"]
            factory_config = factory.build_config(fid=fid, iid=1, dim=10)

            template_types = _extract_nesting_types(template)
            factory_types = _extract_nesting_types(factory_config)

            assert template_types == factory_types, (
                f"f{fid}: type sequence mismatch\n"
                f"  template: {template_types}\n"
                f"  factory:  {factory_types}"
            )
