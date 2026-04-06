"""
Tests for BBOB suite metadata (bbob_suite.json).

Validates that the metadata file is well-formed and consistent
with the BBOBSuiteFactory.
"""

import json
import re
from pathlib import Path

import pytest


def _fid_from_string_id(string_id: str) -> int:
    """Extract numeric function ID from string like 'bbob_f01_sphere'."""
    match = re.search(r"f(\d{2})", string_id)
    assert match is not None, f"Cannot extract fid from '{string_id}'"
    return int(match.group(1))


@pytest.fixture
def suite_metadata():
    """Load bbob_suite.json."""
    path = Path(__file__).parents[3] / "src" / "pyMOFL" / "constants" / "bbob" / "bbob_suite.json"
    with path.open() as f:
        return json.load(f)


class TestBBOBSuiteMetadata:
    """Tests for bbob_suite.json metadata."""

    def test_file_exists(self):
        path = (
            Path(__file__).parents[3] / "src" / "pyMOFL" / "constants" / "bbob" / "bbob_suite.json"
        )
        assert path.exists()

    def test_has_required_fields(self, suite_metadata):
        assert "suite_id" in suite_metadata
        assert "name" in suite_metadata
        assert "functions" in suite_metadata
        assert suite_metadata["suite_id"] == "bbob_noiseless"

    def test_has_24_functions(self, suite_metadata):
        assert len(suite_metadata["functions"]) == 24

    def test_function_ids_1_to_24(self, suite_metadata):
        """String IDs should contain f01 through f24."""
        fids = [_fid_from_string_id(f["id"]) for f in suite_metadata["functions"]]
        assert fids == list(range(1, 25))

    def test_ids_are_strings(self, suite_metadata):
        """All IDs should be strings (not integers)."""
        for func in suite_metadata["functions"]:
            assert isinstance(func["id"], str), f"ID {func['id']} should be a string"

    def test_all_functions_have_required_fields(self, suite_metadata):
        required = {
            "id",
            "name",
            "category",
            "base_function",
            "transforms",
            "properties",
            "function",
        }
        for func in suite_metadata["functions"]:
            missing = required - set(func.keys())
            assert not missing, f"Function {func['id']} missing fields: {missing}"

    def test_all_functions_have_nested_config(self, suite_metadata):
        """Each function entry should have a nested 'function' config template."""
        for func in suite_metadata["functions"]:
            assert isinstance(func.get("function"), dict), (
                f"Function {func['id']} missing nested config template"
            )

    def test_categories_are_valid(self, suite_metadata):
        valid_categories = {
            "separable",
            "moderate",
            "ill-conditioned",
            "multimodal",
            "weakly-structured",
        }
        for func in suite_metadata["functions"]:
            assert func["category"] in valid_categories, (
                f"Function {func['id']} has invalid category: {func['category']}"
            )

    def test_base_functions_match_factory(self, suite_metadata):
        """Metadata base_function names should match BBOBSuiteFactory's _BBOB_INFO."""
        from pyMOFL.factories.bbob_suite_factory import _BBOB_INFO

        for func in suite_metadata["functions"]:
            fid = _fid_from_string_id(func["id"])
            assert fid in _BBOB_INFO
            assert func["base_function"] == _BBOB_INFO[fid]["base_function"], (
                f"Function {fid}: metadata says {func['base_function']!r}, "
                f"factory says {_BBOB_INFO[fid]['base_function']!r}"
            )

    def test_names_match_factory(self, suite_metadata):
        """Metadata names should match BBOBSuiteFactory's _BBOB_INFO."""
        from pyMOFL.factories.bbob_suite_factory import _BBOB_INFO

        for func in suite_metadata["functions"]:
            fid = _fid_from_string_id(func["id"])
            assert func["name"] == _BBOB_INFO[fid]["name"], (
                f"Function {fid}: metadata says {func['name']!r}, "
                f"factory says {_BBOB_INFO[fid]['name']!r}"
            )

    def test_categories_match_factory(self, suite_metadata):
        """Metadata categories should match BBOBSuiteFactory's _BBOB_INFO."""
        from pyMOFL.factories.bbob_suite_factory import _BBOB_INFO

        for func in suite_metadata["functions"]:
            fid = _fid_from_string_id(func["id"])
            assert func["category"] == _BBOB_INFO[fid]["category"], (
                f"Function {fid}: metadata says {func['category']!r}, "
                f"factory says {_BBOB_INFO[fid]['category']!r}"
            )

    def test_instance_generation_is_programmatic(self, suite_metadata):
        assert suite_metadata.get("instance_generation") == "programmatic"

    def test_dimensions_field(self, suite_metadata):
        """Suite should declare supported dimensions."""
        dims = suite_metadata.get("dimensions", {})
        assert "supported" in dims
        assert dims["supported"] == [2, 3, 5, 10, 20, 40]
        assert "default" in dims

    def test_search_space_field(self, suite_metadata):
        """Suite should declare search space bounds."""
        ss = suite_metadata.get("search_space", {})
        assert "default_bounds" in ss
        assert ss["default_bounds"]["min"] == -5
        assert ss["default_bounds"]["max"] == 5
