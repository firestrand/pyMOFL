"""Optional deterministic definition records; replay uses public loading only."""

import base64
import hashlib
import json
import math
import platform
import re
import sys
from collections.abc import Mapping
from dataclasses import fields
from enum import Enum
from operator import index
from pathlib import Path
from typing import Any, cast

import numpy as np

from . import __version__
from .core.bounds import Bounds
from .core.function import OptimizationFunction
from .factories.bbob_suite_factory import BBOBSuiteFactory
from .factories.gnbg_suite_factory import GNBGSuiteFactory
from .functions.transformations.base import PenaltyTransform, ScalarTransform, VectorTransform
from .loader import _resolve_suite_path, load
from .utils.suite_config import _extract_function_code, iter_file_references, load_suite_config

_ROOT = Path(__file__).resolve().parent
_FAMILIES = frozenset({"cec2005", "cec2014", "gnbg", "spso2007", "spso2011"})
_NOISE_TYPES = frozenset({"noise", "gaussian_noise", "uniform_noise", "cauchy_noise", "quartic"})


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def _file_record(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    if not path.is_relative_to(_ROOT):
        raise ValueError("Definition artifacts must be confined to the package")
    return {
        "path": path.relative_to(_ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _snapshot(value: object) -> object:
    """Observe only fresh library-owned parameter data; never recreate objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Unsupported nonfinite scalar parameter")
        return value
    if isinstance(value, np.generic):
        return _snapshot(value.item())
    if isinstance(value, Enum):
        if not type(value).__module__.startswith("pyMOFL.core."):
            raise ValueError("Unsupported parameter enum")
        return {"enum": f"{type(value).__module__}.{type(value).__qualname__}", "name": value.name}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            if not all(isinstance(item, Enum) for item in value.flat):
                raise ValueError("Only library enums are supported in object arrays")
            return {
                "dtype": "object",
                "shape": list(value.shape),
                "enum_values": [_snapshot(item) for item in value.flat],
            }
        if value.dtype.kind not in "biufc":
            raise ValueError("Unsupported parameter array dtype")
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "bytes_base64": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, (list, tuple)):
        return [_snapshot(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Parameter mapping keys must be strings")
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, Bounds):
        return {
            "type": "pyMOFL.core.bounds.Bounds",
            "parameters": {
                field.name: _snapshot(getattr(value, field.name)) for field in fields(value)
            },
        }
    if isinstance(
        value, (OptimizationFunction, VectorTransform, ScalarTransform, PenaltyTransform)
    ):
        cls = type(value)
        if not cls.__module__.startswith(("pyMOFL.functions.", "pyMOFL.compositions.")):
            raise ValueError("Only fresh library-owned components can be observed")
        return {
            "type": f"{cls.__module__}.{cls.__qualname__}",
            "parameters": {key: _snapshot(item) for key, item in vars(value).items()},
        }
    raise ValueError(f"Unsupported definition parameter type: {type(value).__name__}")


def _reject_noise(config: dict[str, Any]) -> None:
    nodes = [config]
    while nodes:
        node = nodes.pop()
        if str(node.get("type", "")).lower() in _NOISE_TYPES:
            raise ValueError("Noisy definitions and RNG state replay are unsupported")
        if isinstance(node.get("function"), dict):
            nodes.append(node["function"])
        nodes.extend(item for item in node.get("functions", []) if isinstance(item, dict))


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    number = index(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def export_definition(
    name_or_id: str | int,
    *,
    suite: str,
    dimension: int | None = None,
    instance: int | None = None,
) -> dict[str, Any]:
    """Create a version-1 record from a fresh supported deterministic definition.

    Parameters
    ----------
    name_or_id, suite, dimension, instance
        Existing suite loading selectors. BBOB defaults to actual instance 1;
        other supported families reject an irrelevant instance selector.

    Returns
    -------
    dict
        Strict JSON-compatible config, resolved parameter observations,
        selected artifacts, numerical code and software identity. Snapshots
        are audit data, not instructions for arbitrary object reconstruction.

    Notes
    -----
    No current object, cache or RNG state is exported. Unsupported state fails
    explicitly. Infinite numerical array metadata is encoded as typed bytes.
    Hashes detect content mismatch, not authenticity. The optional helper
    neither fetches resources nor changes benchmark evaluation methods.
    """
    dim = None if dimension is None else _positive_integer(dimension, "dimension")
    family = suite.strip().lower()
    query = str(name_or_id).strip()
    code = _extract_function_code(query)
    if code is None and re.fullmatch(r"f?\d+", query.lower()):
        code = f"f{int(query.lower().removeprefix('f')):02d}"
    artifacts = []
    config_path = None
    if family in {"bbob", "bbob_noiseless"}:
        if code is None:
            raise ValueError("BBOB definition requires a function number")
        iid = 1 if instance is None else _positive_integer(instance, "instance")
        family = "bbob"
        identity = f"bbob_{code}"
        config = BBOBSuiteFactory().build_config(int(code[1:]), iid, dim or 2)
    else:
        if instance is not None:
            raise ValueError("This suite has no instance selector")
        if family in {"gnbg", "gnbg_suite"}:
            # Observe the existing producer's selected source, including its
            # cec2025 preference; do not replicate that selection policy.
            config_path = GNBGSuiteFactory()._suite_config_path
            family = "gnbg"
        else:
            config_path = _resolve_suite_path(family)
            family = config_path.stem.removesuffix("_suite")
        if family not in _FAMILIES:
            raise ValueError("Unsupported definition suite")
        payload = load_suite_config(config_path)
        entry = next(
            (
                entry
                for entry in payload["functions"]
                if str(entry["id"]).lower() == query.lower()
                or (code is not None and _extract_function_code(str(entry["id"])) == code)
            ),
            None,
        )
        if entry is None:
            raise ValueError("Function identifier not found in selected suite")
        identity = str(entry["id"])
        config = entry["function"]
        iid = None
        artifacts.append(_file_record(config_path))
    _reject_noise(config)
    function = load(identity, suite=family, dimension=dim, instance=iid)
    if config_path is not None:
        for reference in sorted(set(iter_file_references(config))):
            artifacts.append(
                _file_record(
                    config_path.parent / reference.replace("{dim}", str(function.dimension))
                )
            )
    code_paths = [
        path
        for path in _ROOT.rglob("*.py")
        if path.relative_to(_ROOT).parts[0]
        in {"core", "functions", "compositions", "factories", "utils"}
        or (
            path.parent == _ROOT
            and path.name in {"__init__.py", "loader.py", "registry.py", "definition.py"}
        )
    ]
    record: dict[str, Any] = {
        "schema_version": 1,
        "request": {
            "function_id": identity,
            "suite": family,
            "dimension": function.dimension,
            "instance": iid,
        },
        "original_config": _snapshot(config),
        "resolved_parameters": _snapshot(function),
        "artifacts": artifacts,
        "numerical_code": [_file_record(path) for path in sorted(code_paths)],
        "software": {
            "pymofl_version": __version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "python_minor": list(sys.version_info[:2]),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
        },
    }
    record["integrity_sha256"] = _digest(record)
    return record


def reconstruct_definition(manifest: Mapping[str, object]) -> OptimizationFunction:
    """Validate a producer record and load a fresh matching deterministic definition.

    Matching code, NumPy, Python minor and platform identities are required.
    Python patch is recorded as informational environment metadata. No stored
    parameter or path is used to instantiate classes or open arbitrary files.
    Changed BLAS/runtime numerical identity is not promised.
    """
    record = dict(manifest)
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("Unsupported definition schema")
    integrity = record.pop("integrity_sha256", None)
    if not isinstance(integrity, str) or integrity != _digest(record):
        raise ValueError("Definition integrity hash does not match")
    request = record.get("request")
    if not isinstance(request, dict) or set(request) != {
        "function_id",
        "suite",
        "dimension",
        "instance",
    }:
        raise ValueError("Invalid definition request")
    # The key set was validated above; values remain untrusted until narrowed.
    request_fields = cast(Mapping[str, object], request)
    function_id = request_fields["function_id"]
    suite = request_fields["suite"]
    dimension = request_fields["dimension"]
    instance = request_fields["instance"]
    if not isinstance(function_id, str) or not isinstance(suite, str):
        raise ValueError("Invalid definition identity")
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise ValueError("Invalid definition dimension")
    if instance is not None and (not isinstance(instance, int) or isinstance(instance, bool)):
        raise ValueError("Invalid definition instance")
    current = export_definition(
        function_id,
        suite=suite,
        dimension=dimension,
        instance=instance,
    )
    current.pop("integrity_sha256")
    software = record.get("software")
    if not isinstance(software, dict) or not all(isinstance(key, str) for key in software):
        raise ValueError("Invalid software metadata")
    python_version = cast(Mapping[str, object], software).get("python_version")
    if not isinstance(python_version, str):
        raise ValueError("Invalid software metadata")
    # Patch releases are informational; all declared compatibility fields match.
    current["software"]["python_version"] = python_version
    # Python container equality conflates bool/int/float JSON observations.
    # Compare their canonical representations to preserve the typed record.
    if _digest(record) != _digest(current):
        raise ValueError("Definition parameters, artifacts or software are incompatible")
    return load(
        function_id,
        suite=suite,
        dimension=dimension,
        instance=instance,
    )
