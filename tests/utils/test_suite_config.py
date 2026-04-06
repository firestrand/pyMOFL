"""Tests for suite_config utility functions.

Validates ID lookup logic including integer/numeric ID support
for BBOB-style suite configs.
"""

import pytest

from pyMOFL.utils.suite_config import (
    _extract_function_code,
    find_suite_function_config,
    supported_dimensions,
)

# --- Fixtures ---


@pytest.fixture
def string_id_suite():
    """Suite with CEC-style string IDs."""
    return {
        "functions": [
            {
                "id": "cec05_f01_shifted_sphere",
                "function": {"type": "sphere", "parameters": {"dimension": 10}},
            },
            {
                "id": "cec05_f02_shifted_schwefel",
                "function": {"type": "schwefel_1_2", "parameters": {"dimension": 10}},
            },
        ]
    }


@pytest.fixture
def integer_id_suite():
    """Suite with BBOB-style integer IDs."""
    return {
        "functions": [
            {
                "id": 1,
                "name": "Sphere",
                "function": {"type": "sphere", "parameters": {"dimension": 10}},
            },
            {
                "id": 2,
                "name": "Ellipsoidal",
                "function": {"type": "elliptic", "parameters": {"dimension": 10}},
            },
            {
                "id": 10,
                "name": "Ellipsoidal (rotated)",
                "function": {"type": "elliptic", "parameters": {"dimension": 10}},
            },
        ]
    }


@pytest.fixture
def bbob_string_id_suite():
    """Suite with BBOB-style string IDs (after enrichment)."""
    return {
        "functions": [
            {
                "id": "bbob_f01_sphere",
                "function": {"type": "sphere", "parameters": {"dimension": 10}},
            },
            {
                "id": "bbob_f02_ellipsoidal",
                "function": {"type": "elliptic", "parameters": {"dimension": 10}},
            },
        ]
    }


# --- Tests for _extract_function_code ---


class TestExtractFunctionCode:
    def test_cec_string_id(self):
        assert _extract_function_code("cec05_f01_shifted_sphere") == "f01"

    def test_bbob_string_id(self):
        assert _extract_function_code("bbob_f01_sphere") == "f01"

    def test_bare_code(self):
        assert _extract_function_code("f01") == "f01"

    def test_no_code(self):
        assert _extract_function_code("sphere") is None

    def test_two_digit_code(self):
        assert _extract_function_code("bbob_f24_lunacek") == "f24"


# --- Tests for find_suite_function_config with string IDs ---


class TestFindSuiteFunctionConfigStringIDs:
    def test_exact_match(self, string_id_suite):
        config = find_suite_function_config(string_id_suite, "cec05_f01_shifted_sphere")
        assert config["type"] == "sphere"

    def test_code_match(self, string_id_suite):
        config = find_suite_function_config(string_id_suite, "f01")
        assert config["type"] == "sphere"

    def test_not_found(self, string_id_suite):
        with pytest.raises(ValueError, match="not found"):
            find_suite_function_config(string_id_suite, "f99")


# --- Tests for find_suite_function_config with integer IDs ---


class TestFindSuiteFunctionConfigIntegerIDs:
    def test_integer_id_lookup_by_bare_number_string(self, integer_id_suite):
        """Looking up '1' should match entry with id=1."""
        config = find_suite_function_config(integer_id_suite, "1")
        assert config["type"] == "sphere"

    def test_integer_id_lookup_by_integer(self, integer_id_suite):
        """Looking up integer 1 should match entry with id=1."""
        config = find_suite_function_config(integer_id_suite, 1)
        assert config["type"] == "sphere"

    def test_integer_id_lookup_two_digits(self, integer_id_suite):
        """Looking up '10' should match entry with id=10."""
        config = find_suite_function_config(integer_id_suite, "10")
        assert config["type"] == "elliptic"

    def test_integer_id_not_found(self, integer_id_suite):
        with pytest.raises(ValueError, match="not found"):
            find_suite_function_config(integer_id_suite, "99")

    def test_integer_id_not_skipped(self, integer_id_suite):
        """Integer IDs should NOT be silently skipped (old bug)."""
        # All 3 entries should be findable
        find_suite_function_config(integer_id_suite, "1")
        find_suite_function_config(integer_id_suite, "2")
        find_suite_function_config(integer_id_suite, "10")


# --- Tests for BBOB string ID format ---


class TestFindSuiteFunctionConfigBBOBStringIDs:
    def test_bbob_exact_match(self, bbob_string_id_suite):
        config = find_suite_function_config(bbob_string_id_suite, "bbob_f01_sphere")
        assert config["type"] == "sphere"

    def test_bbob_code_match(self, bbob_string_id_suite):
        config = find_suite_function_config(bbob_string_id_suite, "f01")
        assert config["type"] == "sphere"

    def test_bbob_bare_number(self, bbob_string_id_suite):
        """Bare '1' should match 'bbob_f01_sphere' via code extraction."""
        config = find_suite_function_config(bbob_string_id_suite, "1")
        assert config["type"] == "sphere"


# --- Tests for supported_dimensions ---


class TestSupportedDimensions:
    def test_with_supported_list(self):
        entry = {"dimensions": {"supported": [2, 5, 10, 20, 40]}}
        assert supported_dimensions(entry) == [2, 5, 10, 20, 40]

    def test_empty(self):
        assert supported_dimensions({}) == []

    def test_no_supported_key(self):
        entry = {"dimensions": {"default": 10}}
        assert supported_dimensions(entry) == []
