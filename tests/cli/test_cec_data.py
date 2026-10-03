import re
from pathlib import Path

from typer.testing import CliRunner

from pyMOFL.cli.main import app

runner = CliRunner(env={"NO_COLOR": "1", "FORCE_COLOR": "0", "TERM": "dumb"})


def _clean(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)


class TestSuiteCLI:
    """Tests for suite utility commands."""

    def test_suite_help(self):
        result = runner.invoke(app, ["suite", "--help"])
        assert result.exit_code == 0
        output = _clean(result.output)
        assert "list" in output
        assert "validate" in output

    def test_suite_list_help(self):
        result = runner.invoke(app, ["suite", "list", "--help"])
        assert result.exit_code == 0
        output = _clean(result.output)
        assert "Path to a suite JSON configuration" in output
        assert "--suite-id" in output
        assert "--search" in output

    def test_suite_list_gnbg(self):
        result = runner.invoke(app, ["suite", "list", "--suite-id", "gnbg_suite"])
        assert result.exit_code == 0
        output = _clean(result.output)
        assert "gnbg_f01" in output
        assert "gnbg_f24" in output
        assert "2, 10, 30, 50, 100" in output

    def test_suite_validate_help(self):
        result = runner.invoke(app, ["suite", "validate", "--help"])
        assert result.exit_code == 0
        output = _clean(result.output)
        assert "Path to a suite JSON configuration" in output
        assert "--suite-id" in output
        assert "--strict" in output

    def test_validate_default_suite(self):
        result = runner.invoke(app, ["suite", "validate", "--suite-id", "cec2005_suite"])
        assert result.exit_code == 0
        assert "All referenced files exist." in result.output

    def test_validate_competition_suites(self):
        for sid in ["cec2008", "cec2010", "cec2015_niching", "cec2024", "cec2025"]:
            result = runner.invoke(app, ["suite", "validate", "--suite-id", sid])
            assert result.exit_code == 0
            assert "All referenced files exist." in result.output

    def test_suite_list_competition_suites(self):
        r_08 = runner.invoke(app, ["suite", "list", "--suite-id", "cec2008"])
        assert r_08.exit_code == 0
        assert "cec08_f01" in r_08.output

        r_10 = runner.invoke(app, ["suite", "list", "--suite-id", "cec2010"])
        assert r_10.exit_code == 0
        assert "cec10_f01" in r_10.output

        r_nich = runner.invoke(app, ["suite", "list", "--suite-id", "cec2015_niching"])
        assert r_nich.exit_code == 0
        assert "cec15_nich_f01" in r_nich.output

        r_24 = runner.invoke(app, ["suite", "list", "--suite-id", "cec2024"])
        assert r_24.exit_code == 0
        assert "cec24_f01" in r_24.output

        r_25 = runner.invoke(app, ["suite", "list", "--suite-id", "cec2025"])
        assert r_25.exit_code == 0
        assert "cec25_f01" in r_25.output

    def test_validate_missing_reference(self, tmp_path: Path):
        suite_dir = tmp_path / "suite"
        suite_dir.mkdir()
        config = tmp_path / "suite.json"
        config.write_text(
            """
            {
              "suite_id": "local_suite",
              "name": "Local Test Suite",
              "description": "Suite for CLI validation",
              "functions": [
                {
                  "id": "local_shifted",
                  "category": "Unimodal",
                  "dimensions": {
                    "supported": [10, 20],
                    "default": 10
                  },
                  "function": {
                    "type": "sphere",
                    "parameters": {},
                    "function": {
                      "type": "shift",
                      "parameters": {"vector": "missing_shift_D{dim}.txt"}
                    }
                  }
                }
              ]
            }
            """.strip()
        )

        result = runner.invoke(
            app,
            ["suite", "validate", "--suite", str(config), "--suite-dir", str(suite_dir), "--json"],
        )
        assert result.exit_code == 0
        assert '"command": "suite.validate"' in result.output
        assert '"missing_count": 2' in result.output
        assert '"resolved": "missing_shift_D10.txt"' in result.output
        assert '"resolved": "missing_shift_D20.txt"' in result.output
