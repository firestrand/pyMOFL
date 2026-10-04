"""Required coverage checks use the actual approved capture producer."""

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def reference_api():
    assert importlib.util.find_spec("pyMOFL.reference_validation") is not None, (
        "optional required-reference producer is unavailable"
    )
    module = importlib.import_module("pyMOFL.reference_validation")
    assert callable(module.prepare_reference_manifest)
    assert callable(module.validate_reference_manifest)
    return module


def test_required_reference_api_is_available():
    reference_api()


@pytest.fixture
def capture_root():
    root = os.environ.get("PYMOFL_REFERENCE_CAPTURE_ROOT")
    if root is None:
        pytest.skip("explicit approved reference captures are not provisioned")
    return Path(root)


def test_real_producer_and_every_required_execution(capture_root, monkeypatch):
    api = reference_api()
    manifest = api.prepare_reference_manifest(capture_root)
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    observed = {"scalar": 0, "batch": 0, "batch_rows": 0}
    original_load = api.load

    class ObservedFunction:
        def __init__(self, function):
            self.function = function

        def evaluate(self, x):
            observed["scalar"] += 1
            return self.function.evaluate(x)

        def evaluate_batch(self, X):
            observed["batch"] += 1
            observed["batch_rows"] += len(X)
            return self.function.evaluate_batch(X)

    def observed_load(*args, **kwargs):
        return ObservedFunction(original_load(*args, **kwargs))

    monkeypatch.setattr(api, "load", observed_load)
    report = api.validate_reference_manifest(manifest, capture_root=capture_root)
    json.dumps(report, allow_nan=False)

    assert manifest["schema_version"] == report["schema_version"] == 1
    assert len(manifest["captures"]) == 90
    assert manifest["allowed_deviations"] == []
    assert manifest["verification"]["required_data_files"] == 144
    assert report["required_cases"] == len(report["cases"]) == 450
    assert report["success"] is True
    assert report["status_counts"] == {
        "passed": 450,
        "failed": 0,
        "unavailable": 0,
        "deviation": 0,
    }
    assert report["scalar_executed"] == observed["scalar"] == 450
    assert observed["batch"] == 90
    assert report["batch_executed"] == observed["batch_rows"] == 450
    expected_ids = {
        f"cec2014/f{fid:02d}/D{dim}/{name}"
        for fid in range(1, 31)
        for dim in (10, 30, 50)
        for name in ("shift", "zeros", "random", "bounds_min", "bounds_max")
    }
    assert {case["id"] for case in report["cases"]} == expected_ids
    for case in report["cases"]:
        assert case["scalar_executed"] and case["batch_executed"]
        assert case["status"] == "passed"
        assert case["scalar_abs_diff"] < case["tolerance"]
        assert case["batch_abs_diff"] < case["tolerance"]
        assert "x" not in case


def test_required_script_missing_data_is_nonzero(tmp_path):
    output = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts/verify_reference_manifest.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--capture-root",
            str(tmp_path / "absent-root"),
            "--manifest",
            str(tmp_path / "absent-manifest"),
            "--report",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 1
    report = json.loads(output.read_text())
    assert report["success"] is False
    assert report["status_counts"]["unavailable"] == 450
    assert report["scalar_executed"] == report["batch_executed"] == 0


def test_actual_required_script(capture_root, tmp_path):
    api = reference_api()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(api.prepare_reference_manifest(capture_root), allow_nan=False))
    output = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[1] / "scripts/verify_reference_manifest.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--capture-root",
            str(capture_root),
            "--manifest",
            str(manifest),
            "--report",
            str(output),
        ],
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    report = json.loads(output.read_text())
    assert report["success"] is True
    assert report["scalar_executed"] == report["batch_executed"] == 450
    assert "450" in result.stdout


@pytest.mark.parametrize(
    "mutation", ["schema", "source", "capture_hash", "missing", "duplicate", "deviation"]
)
def test_authorized_manifest_mutations_are_unavailable(capture_root, mutation):
    api = reference_api()
    manifest = api.prepare_reference_manifest(capture_root)
    if mutation == "schema":
        manifest["schema_version"] = 2
    elif mutation == "source":
        manifest["source_identity"]["archive_sha256"] = "0" * 64
    elif mutation == "capture_hash":
        manifest["captures"][0]["sha256"] = "0" * 64
    elif mutation == "missing":
        manifest["captures"].pop()
    elif mutation == "duplicate":
        manifest["captures"][1] = manifest["captures"][0]
    else:
        manifest["allowed_deviations"] = [{"id": "cec2014/f01/D10/shift", "reason": "unauthorized"}]
    report = api.validate_reference_manifest(manifest, capture_root=capture_root)
    assert report["success"] is False
    assert report["status_counts"] == {"passed": 0, "failed": 0, "unavailable": 450, "deviation": 0}
    assert report["scalar_executed"] == report["batch_executed"] == 0
    assert all(case["reason"] != "not executed" for case in report["cases"])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_file",
        "missing_case",
        "duplicate_case",
        "wrong_id",
        "bool_id",
        "wrong_dimension",
        "wrong_shape",
        "bool_coordinate",
        "nan",
        "infinity",
        "input_hash",
        "file_hash",
        "missing_inventory",
        "duplicate_inventory",
        "missing_data",
        "source",
        "escape",
        "record_count",
        "duplicate_data",
        "data_escape",
        "symlink_escape",
    ],
)
def test_authorized_reference_copies_fail_deep_guards(capture_root, tmp_path, mutation):
    api = reference_api()
    original_manifest = api.prepare_reference_manifest(capture_root)
    root = tmp_path / "adversarial-reference"
    shutil.copytree(capture_root, root)
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    item = provenance["captures"][0]
    path = root / item["path"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing_file":
        path.unlink()
    elif mutation == "missing_inventory":
        provenance["captures"].pop()
    elif mutation == "duplicate_inventory":
        provenance["captures"][1] = item
    elif mutation == "missing_data":
        provenance["data_files"] = []
    elif mutation == "duplicate_data":
        provenance["data_files"].append(provenance["data_files"][0])
    elif mutation == "data_escape":
        provenance["data_files"][0]["path"] = "cec14-c-code/input_data/../../../outside.txt"
    elif mutation == "symlink_escape":
        source = root / "engine/capture"
        outside = tmp_path / "outside-capture"
        shutil.copyfile(source, outside)
        source.unlink()
        source.symlink_to(outside)
    elif mutation == "record_count":
        item["records"] = True
    elif mutation == "source":
        provenance["archive_sha256"] = "0" * 64
    elif mutation == "escape":
        item["path"] = "../outside.jsonl"
    elif mutation == "file_hash":
        item["sha256"] = "0" * 64
    elif mutation == "input_hash":
        item["input_sha256"] = "0" * 64
    else:
        if mutation == "missing_case":
            rows.pop()
        elif mutation == "duplicate_case":
            rows[1] = rows[0]
        elif mutation == "wrong_id":
            rows[0]["func_id"] += 1
        elif mutation == "bool_id":
            rows[0]["func_id"] = True
        elif mutation == "wrong_dimension":
            rows[0]["dim"] += 1
        elif mutation == "wrong_shape":
            rows[0]["x"].pop()
        elif mutation == "bool_coordinate":
            rows[0]["x"][0] = True
        else:
            rows[0]["x"][0] = float("nan" if mutation == "nan" else "inf")
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        # Reach the metadata/completeness guard beyond the file checksum.
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps(provenance))
    with pytest.raises((ValueError, OSError, TypeError, KeyError)):
        api.prepare_reference_manifest(root)
    report = api.validate_reference_manifest(original_manifest, capture_root=root)
    assert report["success"] is False
    assert report["status_counts"]["unavailable"] == 450
    assert report["scalar_executed"] == report["batch_executed"] == 0
    assert api.prepare_reference_manifest(capture_root) == original_manifest


@pytest.mark.parametrize(
    "fault",
    [
        "construction",
        "scalar_exception",
        "batch_exception",
        "scalar_bool",
        "scalar_string",
        "scalar_nan",
        "scalar_difference",
        "batch_list",
        "batch_shape",
        "batch_complex",
        "batch_nan",
        "batch_difference",
    ],
)
def test_authorized_execution_faults_report_observed_counts(capture_root, monkeypatch, fault):
    api = reference_api()
    manifest = api.prepare_reference_manifest(capture_root)
    real_load = api.load
    calls = {"scalar": 0, "batch": 0}

    class FaultedRealFunction:
        def __init__(self, delegate):
            self.delegate = delegate

        def evaluate(self, x):
            calls["scalar"] += 1
            if fault == "scalar_exception":
                raise TypeError("authorized scalar failure")
            value = self.delegate.evaluate(x)
            if fault == "scalar_bool":
                return True
            if fault == "scalar_string":
                return str(value)
            if fault == "scalar_nan":
                return float("nan")
            if fault == "scalar_difference":
                return value + max(1.0, abs(value)) * 1e-5
            return value

        def evaluate_batch(self, X):
            calls["batch"] += 1
            if fault == "batch_exception":
                raise TypeError("authorized batch failure")
            values = self.delegate.evaluate_batch(X)
            if fault == "batch_list":
                return values.tolist()
            if fault == "batch_shape":
                return values[:, None]
            if fault == "batch_complex":
                return values.astype(complex)
            if fault == "batch_nan":
                return values * float("nan")
            if fault == "batch_difference":
                return values + np.maximum(1.0, np.abs(values)) * 1e-5
            return values

    def faulted_load(name, **kwargs):
        if name == "cec2014_f01" and kwargs["dimension"] == 10:
            if fault == "construction":
                raise TypeError("authorized construction failure")
            return FaultedRealFunction(real_load(name, **kwargs))
        return real_load(name, **kwargs)

    monkeypatch.setattr(api, "load", faulted_load)
    report = api.validate_reference_manifest(manifest, capture_root=capture_root)
    assert report["success"] is False
    assert report["status_counts"] == {"passed": 445, "failed": 5, "unavailable": 0, "deviation": 0}
    assert report["scalar_executed"] == (
        445 if fault in {"construction", "scalar_exception"} else 450
    )
    assert report["batch_executed"] == (
        445 if fault in {"construction", "batch_exception"} else 450
    )
    assert calls == (
        {"scalar": 0, "batch": 0} if fault == "construction" else {"scalar": 5, "batch": 1}
    )
    assert all(case["reason"] for case in report["cases"])
    json.dumps(report, allow_nan=False)
