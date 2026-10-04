import json
from pathlib import Path

from typer.testing import CliRunner

from pyMOFL import __version__
from pyMOFL.cli.main import app

runner = CliRunner(env={"NO_COLOR": "1", "FORCE_COLOR": "0", "TERM": "dumb"})


class TestCliErgonomics:
    """Tests for info, eval, list, and suites CLI commands."""

    def test_root_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "info" in result.output
        assert "eval" in result.output
        assert "list" in result.output
        assert "suites" in result.output
        assert "suite" in result.output

    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert result.output.strip() == __version__

    # -------------------------------------------------------------
    # Info Command Tests
    # -------------------------------------------------------------

    def test_info_sphere(self):
        result = runner.invoke(app, ["info", "sphere"])
        assert result.exit_code == 0
        assert "Sphere" in result.output
        assert "classical" in result.output
        assert "[-100, 100]^10" in result.output
        assert "0" in result.output  # global minimum

    def test_info_cec17(self):
        result = runner.invoke(app, ["info", "cec17_f01", "-d", "10"])
        assert result.exit_code == 0
        assert "cec17_f01" in result.output
        assert "cec2017" in result.output
        assert "Unimodal" in result.output
        assert "100" in result.output  # optimal value

    def test_info_bbob(self):
        result = runner.invoke(app, ["info", "bbob_f01", "-d", "5", "-i", "1"])
        assert result.exit_code == 0
        assert "bbob_f01" in result.output
        assert "bbob" in result.output

    def test_info_json(self):
        result = runner.invoke(app, ["info", "sphere", "-d", "5", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "Sphere"
        assert data["dimension"] == 5
        assert data["global_minimum"]["value"] == 0.0
        assert "operational_bounds" in data

    def test_info_invalid_function(self):
        result = runner.invoke(app, ["info", "non_existent_function_xyz"])
        assert result.exit_code == 1
        assert "Could not" in result.output or "Error" in result.output

    # -------------------------------------------------------------
    # Eval Command Tests
    # -------------------------------------------------------------

    def test_eval_zeros_default(self):
        result = runner.invoke(app, ["eval", "sphere", "-d", "5"])
        assert result.exit_code == 0
        assert "f(x) = 0" in result.output
        assert "Vector of zeros" in result.output

    def test_eval_ones(self):
        result = runner.invoke(app, ["eval", "sphere", "-d", "5", "--ones"])
        assert result.exit_code == 0
        assert "f(x) = 5" in result.output

    def test_eval_custom_x(self):
        result = runner.invoke(app, ["eval", "sphere", "-x", "1.0, 2.0, 3.0, 4.0, 5.0"])
        assert result.exit_code == 0
        assert "f(x) = 55" in result.output
        assert "Dimension                 5" in result.output

    def test_eval_optimum(self):
        result = runner.invoke(app, ["eval", "sphere", "-d", "10", "--optimum"])
        assert result.exit_code == 0
        assert "f(x) = 0" in result.output
        assert "Error |f(x) - f(x*)|      0" in result.output

    def test_eval_cec17_optimum(self):
        result = runner.invoke(app, ["eval", "cec17_f01", "-d", "10", "--optimum"])
        assert result.exit_code == 0
        assert "Output f(x)               100" in result.output
        assert "Error |f(x) - f(x*)|      0" in result.output

    def test_eval_random(self):
        result = runner.invoke(app, ["eval", "sphere", "-d", "5", "--random"])
        assert result.exit_code == 0
        assert "f(x) =" in result.output

    def test_eval_random_batch(self):
        result = runner.invoke(app, ["eval", "sphere", "-d", "10", "--random-batch", "500"])
        assert result.exit_code == 0
        assert "Evaluated Points          500" in result.output
        assert "Throughput" in result.output
        assert "Best f(x)" in result.output

    def test_eval_random_batch_json(self):
        result = runner.invoke(
            app, ["eval", "sphere", "-d", "10", "--random-batch", "100", "--json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["count"] == 100
        assert data["dimension"] == 10
        assert "best_value" in data
        assert "throughput_evals_per_sec" in data

    def test_eval_points_file(self, tmp_path: Path):
        pts_file = tmp_path / "points.txt"
        pts_file.write_text("1, 1, 1\n2, 2, 2\n")
        result = runner.invoke(app, ["eval", "sphere", "-d", "3", "--points-file", str(pts_file)])
        assert result.exit_code == 0
        assert "Evaluated Points          2" in result.output
        assert "Best f(x) (Min)           3" in result.output

    def test_eval_dimension_mismatch(self):
        result = runner.invoke(app, ["eval", "sphere", "-d", "5", "-x", "1,2,3"])
        assert result.exit_code != 0
        assert "expected 5" in result.output

    # -------------------------------------------------------------
    # List Command Tests
    # -------------------------------------------------------------

    def test_list_default(self):
        result = runner.invoke(app, ["list", "--limit", "10"])
        assert result.exit_code == 0
        assert "Function ID" in result.output
        assert "pyMOFL Functions" in result.output

    def test_list_suite_cec2017(self):
        result = runner.invoke(app, ["list", "--suite", "cec2017"])
        assert result.exit_code == 0
        assert "cec17_f01" in result.output
        assert "cec2017" in result.output

    def test_list_suite_classical(self):
        result = runner.invoke(app, ["list", "--suite", "classical", "--limit", "15"])
        assert result.exit_code == 0
        assert "Ackley" in result.output
        assert "classical" in result.output

    def test_list_search(self):
        result = runner.invoke(app, ["list", "--search", "rosen"])
        assert result.exit_code == 0
        assert "Rosenbrock" in result.output or "rosenbrock" in result.output

    def test_list_json(self):
        result = runner.invoke(app, ["list", "--suite", "bbob", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) >= 20
        assert data[0]["suite"] == "bbob_noiseless"

    # -------------------------------------------------------------
    # Suites Command Tests
    # -------------------------------------------------------------

    def test_suites_default(self):
        result = runner.invoke(app, ["suites"])
        assert result.exit_code == 0
        assert "cec2017" in result.output
        assert "bbob_noiseless" in result.output
        assert "classical" in result.output

    def test_suites_search(self):
        result = runner.invoke(app, ["suites", "--search", "2017"])
        assert result.exit_code == 0
        assert "cec2017" in result.output

    def test_suites_json(self):
        result = runner.invoke(app, ["suites", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        suite_ids = [s["suite_id"] for s in data]
        assert "cec2017" in suite_ids
        assert "classical" in suite_ids
