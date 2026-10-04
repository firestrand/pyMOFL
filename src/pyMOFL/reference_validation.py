"""Optional local reference validation; no engines, downloads or CLI imports."""

import hashlib
import json
import math
import platform
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, NoReturn, NotRequired, TypedDict

import numpy as np

from . import __version__
from .loader import load

_PINS = {
    "source": "https://github.com/P-N-Suganthan/CEC2014",
    "revision": "98488087d590c29aaded9978ccfe2a356d10dd63",
    "archive_sha256": "1a210560398ca7a50be6adf1e5e90602222519ef23b6e31aba8847e109761876",
    "original_source_sha256": "4228bcf7d6d8da94bab3dfdb8837251edc26952f498ae9757da57ca4b1dbd650",
    "patched_source_sha256": "3e1e17d6ec6d2722b8f475bc953d869e79aee159260927fd9ec8a043ef660417",
    "portability_patch_sha256": "85b723ad6636372e1a1c429bcc57324347f52def262158be835336c58ba23f67",
    "driver_sha256": "65eb30c309309a0d143ffd0306289e1a22093a3304a46b5a3faf311f9d7413c1",
    "data01_sha256": "5e7268c46c288a287e1909b5986518631ea87f08e16cb2ace0517cc84ef0142a",
}
_DIMENSIONS = (10, 30, 50)
_NAMES = ("shift", "zeros", "random", "bounds_min", "bounds_max")


class _CaseResult(TypedDict):
    id: str
    status: str
    scalar_executed: bool
    batch_executed: bool
    reason: str
    tolerance: NotRequired[float]
    scalar_abs_diff: NotRequired[float | None]
    batch_abs_diff: NotRequired[float | None]


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> Any:
    """Keep untrusted JSON confined to the transport boundary."""
    return json.loads(path.read_text(), parse_constant=_reject_constant)


def _confined(root: Path, relative: str) -> Path:
    candidate = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Reference path escapes capture root")
    candidate = candidate.resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise ValueError("Reference path escapes capture root")
    return candidate


def _check_file(root: Path, relative: str, expected: str) -> bytes:
    data = _confined(root, relative).read_bytes()
    if _hash(data) != expected:
        raise ValueError("Reference file hash mismatch")
    return data


def _required() -> Iterator[tuple[int, int, str]]:
    for fid in range(1, 31):
        for dim in _DIMENSIONS:
            for name in _NAMES:
                yield fid, dim, name


def _case_id(fid: int, dim: int, name: str) -> str:
    return f"cec2014/f{fid:02d}/D{dim}/{name}"


def _prepare(root: Path) -> tuple[dict[str, Any], dict[tuple[int, int], list[dict[str, Any]]]]:
    provenance_path = _confined(root, "provenance.json")
    provenance_bytes = provenance_path.read_bytes()
    provenance = _read_json(provenance_path)
    if not isinstance(provenance, dict) or any(
        provenance.get(key) != value for key, value in _PINS.items()
    ):
        raise ValueError("Unapproved reference source identity")
    for key, filename in (
        ("patched_source_sha256", "cec14_test_func.cpp"),
        ("portability_patch_sha256", "portability.patch"),
        ("driver_sha256", "driver.cpp"),
        ("executable_sha256", "capture"),
    ):
        _check_file(root, f"engine/{filename}", provenance[key])
    data_paths = set()
    for item in provenance["data_files"]:
        relative = item["path"]
        prefix = "cec14-c-code/input_data/"
        if (
            not isinstance(relative, str)
            or not relative.startswith(prefix)
            or relative in data_paths
        ):
            raise ValueError("Invalid reference data inventory")
        data_paths.add(relative)
        _check_file(root, f"engine/input_data/{relative.removeprefix(prefix)}", item["sha256"])
    required_data = {f"cec14-c-code/input_data/shift_data_{fid}.txt" for fid in range(1, 31)}
    required_data.update(
        f"cec14-c-code/input_data/M_{fid}_D{dim}.txt" for fid in range(1, 31) for dim in _DIMENSIONS
    )
    required_data.update(
        f"cec14-c-code/input_data/shuffle_data_{fid}_D{dim}.txt"
        for fid in (*range(17, 23), 29, 30)
        for dim in _DIMENSIONS
    )
    if not required_data.issubset(data_paths):
        raise ValueError("Required reference data inventory is incomplete")
    expected_paths = {
        f"datasets/CEC2014/func_{fid}_D{dim}/golden.jsonl"
        for fid in range(1, 31)
        for dim in _DIMENSIONS
    }
    inventory = provenance["captures"]
    if (
        not isinstance(inventory, list)
        or len(inventory) != 90
        or {item["path"] for item in inventory} != expected_paths
    ):
        raise ValueError("Required reference files are incomplete or duplicated")
    indexed = {item["path"]: item for item in inventory}
    groups = {}
    captures = []
    for fid in range(1, 31):
        for dim in _DIMENSIONS:
            relative = f"datasets/CEC2014/func_{fid}_D{dim}/golden.jsonl"
            item = indexed[relative]
            if type(item["records"]) is not int or item["records"] != 5:
                raise ValueError("Required record count is invalid")
            raw = _check_file(root, relative, item["sha256"])
            rows = [json.loads(line, parse_constant=_reject_constant) for line in raw.splitlines()]
            if len(rows) != 5 or {row["case"] for row in rows} != set(_NAMES):
                raise ValueError("Required cases are incomplete or duplicated")
            for row in rows:
                if (
                    type(row["func_id"]) is not int
                    or row["func_id"] != fid
                    or type(row["dim"]) is not int
                    or row["dim"] != dim
                    or not isinstance(row["x"], list)
                    or len(row["x"]) != dim
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in row["x"])
                    or type(row["value"]) not in (int, float)
                    or not math.isfinite(row["value"])
                ):
                    raise ValueError("Invalid reference record metadata, shape or numerical data")
            rows = sorted(rows, key=lambda row: _NAMES.index(row["case"]))
            payload = (
                f"{fid} {dim} 5\n"
                + "\n".join(" ".join(format(v, ".17g") for v in row["x"]) for row in rows)
                + "\n"
            )
            if _hash(payload.encode()) != item["input_sha256"]:
                raise ValueError("Reference input provenance hash mismatch")
            groups[fid, dim] = rows
            captures.append(dict(item))
    manifest = {
        "schema_version": 1,
        "suite": "cec2014",
        "required_cases": 450,
        "source_identity": dict(_PINS),
        "provenance_sha256": _hash(provenance_bytes),
        "setup": {
            key: provenance[key]
            for key in ("compiler", "compiler_flags", "platform", "machine", "rights")
        },
        "captures": captures,
        "case_names": list(_NAMES),
        "dimensions": list(_DIMENSIONS),
        "functions": list(range(1, 31)),
        "allowed_deviations": [],
        "verification": {
            "recorded_identity_only": ["archive_sha256", "original_source_sha256"],
            "rehashed_available_files": 4 + len(data_paths) + len(captures),
            "required_data_files": len(required_data),
            "capture_input_hashes": len(captures),
        },
        "tolerance": {"relative_scale": 1e-6, "minimum_scale": 1.0, "comparison": "strict"},
    }
    return manifest, groups


def prepare_reference_manifest(capture_root: str | Path) -> dict[str, Any]:
    """Create a strict JSON manifest from actual approved local CEC2014 captures.

    Source/archive identities are pinned setup provenance; their absent original
    bytes are not rehashed. Available patched source, patch, driver, executable,
    data and capture files are rehashed without executing the reference engine.
    Coverage and approved deviation policy are independent of supplied records.
    Hashes detect mismatches, not authenticity or latest-release status.
    """
    manifest, _ = _prepare(Path(capture_root).resolve(strict=True))
    return manifest


def validate_reference_manifest(
    manifest: Mapping[str, object] | Path, *, capture_root: str | Path
) -> dict[str, Any]:
    """Observe every required scalar/batch result or report an explicit failure.

    A manifest may be a producer-created mapping or an explicitly supplied JSON
    path. All report entries identify required cases, but execution flags/counts
    describe only calls actually completed. Malformed/provenance/operational
    errors cannot become deviations. The approved CEC2014 deviation set is empty.
    This helper never downloads, compiles, imports or executes reference engines.
    """
    cases: list[_CaseResult] = [
        {
            "id": _case_id(fid, dim, name),
            "status": "unavailable",
            "scalar_executed": False,
            "batch_executed": False,
            "reason": "not executed",
        }
        for fid, dim, name in _required()
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "suite": "cec2014",
        "source_identity": dict(_PINS),
        "software": {
            "pymofl_version": __version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
        },
        "required_cases": 450,
        "manifest_sha256": None,
        "provenance_sha256": None,
        "provenance_verified": False,
        "allowed_deviations": [],
        "cases": cases,
    }
    try:
        supplied = _read_json(manifest) if isinstance(manifest, Path) else dict(manifest)
        canonical = _json_bytes(supplied)
        report["manifest_sha256"] = _hash(canonical)
        current, groups = _prepare(Path(capture_root).resolve(strict=True))
        if canonical != _json_bytes(current):
            raise ValueError("Manifest differs from approved required capture definition")
        report["provenance_sha256"] = current["provenance_sha256"]
        report["provenance_verified"] = True
        report["setup"] = current["setup"]
        report["verification"] = current["verification"]
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        for case in cases:
            case["reason"] = "required manifest or reference provenance could not be verified"
        return _finish(report)
    by_id = {case["id"]: case for case in cases}
    for (fid, dim), rows in groups.items():
        entries = [by_id[_case_id(fid, dim, row["case"])] for row in rows]
        try:
            function = load(f"cec2014_f{fid:02d}", suite="cec2014", dimension=dim)
        except Exception as error:
            for entry in entries:
                entry.update(status="failed", reason=f"construction error: {type(error).__name__}")
            continue
        X = np.asarray([row["x"] for row in rows], dtype=np.float64)
        for entry, row, x in zip(entries, rows, X, strict=True):
            tolerance = 1e-6 * max(1.0, abs(row["value"]))
            entry.update(tolerance=tolerance, status="failed", reason="evaluation incomplete")
            try:
                scalar = function.evaluate(x.copy())
                entry["scalar_executed"] = True
                if isinstance(scalar, (bool, np.bool_)) or not isinstance(
                    scalar, (int, float, np.integer, np.floating)
                ):
                    raise ValueError("Invalid scalar result type")
                difference = abs(scalar - row["value"])
                entry["scalar_abs_diff"] = float(difference) if math.isfinite(difference) else None
            except Exception as error:
                entry["reason"] = f"scalar evaluation error: {type(error).__name__}"
        try:
            values = function.evaluate_batch(X.copy())
            for entry in entries:
                entry["batch_executed"] = True
            if (
                not isinstance(values, np.ndarray)
                or values.shape != (len(rows),)
                or values.dtype.kind not in "iuf"
            ):
                raise ValueError("Invalid batch result shape")
            for entry, row, value in zip(entries, rows, values, strict=True):
                difference = abs(value - row["value"])
                entry["batch_abs_diff"] = float(difference) if math.isfinite(difference) else None
        except Exception as error:
            for entry in entries:
                entry["reason"] = f"batch evaluation error: {type(error).__name__}"
        for entry in entries:
            scalar_diff = entry.get("scalar_abs_diff")
            batch_diff = entry.get("batch_abs_diff")
            if scalar_diff is not None and batch_diff is not None:
                if scalar_diff < entry["tolerance"] and batch_diff < entry["tolerance"]:
                    entry.update(status="passed", reason="within reference tolerance")
                else:
                    entry["reason"] = "numerical difference exceeds reference tolerance"
    return _finish(report)


def _finish(report: dict[str, Any]) -> dict[str, Any]:
    cases = report["cases"]
    report["scalar_executed"] = sum(case["scalar_executed"] for case in cases)
    report["batch_executed"] = sum(case["batch_executed"] for case in cases)
    report["status_counts"] = {
        status: sum(case["status"] == status for case in cases)
        for status in ("passed", "failed", "unavailable", "deviation")
    }
    report["success"] = all(case["status"] == "passed" for case in cases)
    return report
