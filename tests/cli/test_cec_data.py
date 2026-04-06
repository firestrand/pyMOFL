"""Tests for generic suite utility CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from pyMOFL.cli.main import app

runner = CliRunner()


class TestSuiteCLI:
    """Tests for suite utility commands."""

    def test_suite_help(self):
        result = runner.invoke(app, ["suite", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "validate" in result.output

    def test_suite_list_help(self):
        result = runner.invoke(app, ["suite", "list", "--help"])
        assert result.exit_code == 0
        assert "Path to a suite JSON configuration" in result.output
        assert "--suite-id" in result.output
        assert "--search" in result.output

    def test_suite_validate_help(self):
        result = runner.invoke(app, ["suite", "validate", "--help"])
        assert result.exit_code == 0
        assert "Path to a suite JSON configuration" in result.output
        assert "--suite-id" in result.output
        assert "--strict" in result.output

    def test_validate_default_suite(self):
        result = runner.invoke(app, ["suite", "validate", "--suite-id", "cec2005_suite"])
        assert result.exit_code == 0
        assert "All referenced files exist." in result.output

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
