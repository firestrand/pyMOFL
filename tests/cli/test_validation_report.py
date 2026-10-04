"""The optional CLI consumes actual producer records and observed results."""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pyMOFL.cli.main import app
from pyMOFL.reference_validation import prepare_reference_manifest, validate_reference_manifest

runner = CliRunner(env={"NO_COLOR": "1", "FORCE_COLOR": "0", "TERM": "dumb"})


def test_validate_command_is_available():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "validate" in result.output, "public validation command is unavailable"


@pytest.mark.parametrize("global_flags", [[], ["--json"], ["--quiet"]])
def test_real_validation_report_workflow(tmp_path, global_flags):
    root = os.environ.get("PYMOFL_REFERENCE_CAPTURE_ROOT")
    if root is None:
        pytest.skip("explicit approved reference captures are not provisioned")
    capture_root = Path(root)
    manifest = prepare_reference_manifest(capture_root)
    expected = validate_reference_manifest(manifest, capture_root=capture_root)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False))
    report_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            *global_flags,
            "validate",
            "--capture-root",
            str(capture_root),
            "--manifest",
            str(manifest_path),
            "--report",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(report_path.read_text()) == expected
    assert expected["scalar_executed"] == expected["batch_executed"] == 450
    if "--json" in global_flags:
        assert json.loads(result.output) == expected
    elif "--quiet" in global_flags:
        assert result.output == ""
    else:
        assert "450" in result.output


@pytest.mark.parametrize("kind", ["missing", "malformed", "schema", "nonfinite"])
def test_authorized_invalid_transport_reports_failure(tmp_path, kind):
    path = tmp_path / "manifest.json"
    if kind == "malformed":
        path.write_text("{")
    elif kind == "schema":
        path.write_text('{"schema_version": 2}')
    elif kind == "nonfinite":
        path.write_text('{"schema_version": NaN}')
    report_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            "--json",
            "validate",
            "--capture-root",
            str(tmp_path / "absent-root"),
            "--manifest",
            str(path),
            "--report",
            str(report_path),
        ],
    )
    assert result.exit_code == 1
    report = json.loads(report_path.read_text())
    assert json.loads(result.output) == report
    assert report["success"] is False
    assert report["status_counts"] == {"passed": 0, "failed": 0, "unavailable": 450, "deviation": 0}
    assert report["scalar_executed"] == report["batch_executed"] == 0


@pytest.mark.parametrize("kind", ["existing", "missing_parent", "directory"])
def test_authorized_report_creation_failure_preserves_destination(tmp_path, kind):
    report_path = tmp_path / "report.json"
    if kind == "existing":
        report_path.write_text("existing user report")
    elif kind == "missing_parent":
        report_path = tmp_path / "missing" / "report.json"
    else:
        report_path.mkdir()
    result = runner.invoke(
        app,
        [
            "validate",
            "--capture-root",
            str(tmp_path / "absent-root"),
            "--manifest",
            str(tmp_path / "absent-manifest"),
            "--report",
            str(report_path),
        ],
    )
    assert result.exit_code == 1
    assert "Could not create the validation report" in result.output
    if kind == "existing":
        assert report_path.read_text() == "existing user report"
    elif kind == "directory":
        assert report_path.is_dir()
    else:
        assert not report_path.exists()


@pytest.mark.parametrize("global_flags", [[], ["--json"], ["--quiet"]])
def test_actual_execution_failure_reaches_cli_report(tmp_path, monkeypatch, global_flags):
    import pyMOFL.reference_validation as reference

    configured = os.environ.get("PYMOFL_REFERENCE_CAPTURE_ROOT")
    if configured is None:
        pytest.skip("explicit approved reference captures are not provisioned")
    root = Path(configured)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(reference.prepare_reference_manifest(root), allow_nan=False)
    )
    original_load = reference.load
    failed_loads = []

    def failed_load(name, **kwargs):
        if name == "cec2014_f01" and kwargs["dimension"] == 10:
            failed_loads.append(name)
            raise TypeError("authorized construction failure")
        return original_load(name, **kwargs)

    monkeypatch.setattr(reference, "load", failed_load)
    destination = tmp_path / "report.json"
    result = runner.invoke(
        app,
        [
            *global_flags,
            "validate",
            "--capture-root",
            str(root),
            "--manifest",
            str(manifest_path),
            "--report",
            str(destination),
        ],
    )
    assert result.exit_code == 1
    report = json.loads(destination.read_text())
    assert report["status_counts"] == {"passed": 445, "failed": 5, "unavailable": 0, "deviation": 0}
    assert report["scalar_executed"] == report["batch_executed"] == 445
    assert failed_loads == ["cec2014_f01"]
    if "--json" in global_flags:
        assert json.loads(result.output) == report
    elif "--quiet" in global_flags:
        assert result.output == ""
    else:
        assert "'failed': 5" in result.output
