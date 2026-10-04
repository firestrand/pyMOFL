"""V7.4: canonical existing requests seed the actual definition producer."""

import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

import pyMOFL
from pyMOFL.factories.gnbg_suite_factory import GNBGSuiteFactory
from pyMOFL.loader import _resolve_suite_path
from pyMOFL.utils.suite_config import _extract_function_code, load_suite_config

# These actual CEC2005 entries contain explicit noise nodes and are outside
# the deterministic record contract. Their configs and numerical APIs remain.
_NOISY_DEFINITION_IDS = frozenset(
    {
        "cec05_f04_shifted_schwefel_1_2_with_noise",
        "cec05_f17_rotated_hybrid_composition_1_noise",
    }
)


def _real_definition_requests():
    requests = []
    for suite, dimension in (
        ("cec2005", 10),
        ("cec2014", 10),
        ("gnbg", 30),
        ("bbob", 2),
        ("spso2007", None),
        ("spso2011", None),
    ):
        path = (
            GNBGSuiteFactory()._suite_config_path if suite == "gnbg" else _resolve_suite_path(suite)
        )
        for entry in load_suite_config(path)["functions"]:
            name = entry["id"]
            if name not in _NOISY_DEFINITION_IDS:
                requests.append(
                    pytest.param(name, suite, dimension, 1 if suite == "bbob" else None, id=name)
                )
    return requests


def _retained_definition_points(name, suite, dimension):
    """Return unchanged retained coordinates, or explicitly metadata-only."""
    code = _extract_function_code(name)
    assert code is not None
    if suite == "cec2005":
        payload = json.loads(
            (Path(__file__).parent / f"validation_data/cec/2005/{code}.json").read_text()
        )
        case = next(case for case in payload["cases"] if case["dimension"] == dimension)
        return np.asarray([case["optimum"], case["random_input"]], dtype=np.float64)
    if suite == "cec2014" and (root := os.environ.get("PYMOFL_REFERENCE_CAPTURE_ROOT")):
        path = Path(root) / f"datasets/CEC2014/func_{int(code[1:])}_D{dimension}/golden.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return np.asarray([row["x"] for row in rows], dtype=np.float64)
    if suite.startswith("spso") and (root := os.environ.get("PYMOFL_SPSO_REFERENCE_PATH")):
        path = Path(root) / suite.removeprefix("spso") / "release.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return np.asarray(
            [row["x"] for row in rows if row["func_id"] == int(code[1:]) and "case" in row],
            dtype=np.float64,
        )
    return None


def definition_api():
    assert importlib.util.find_spec("pyMOFL.definition") is not None, (
        "public optional definition helper is unavailable"
    )
    module = importlib.import_module("pyMOFL.definition")
    assert callable(module.export_definition)
    assert callable(module.reconstruct_definition)
    return module


@pytest.mark.parametrize(
    ("name", "suite", "dimension", "instance"),
    [
        ("cec2005_f01", "cec2005", 10, None),
        ("bbob_f01", "bbob", 2, None),
        ("bbob_f01", "bbob", 2, 1),
        ("gnbg_f24", "gnbg", 30, None),
        ("cec2014_f23", "cec2014", 30, None),
        ("spso2007_f21", "spso2007", None, None),
        ("spso2011_f21", "spso2011", None, None),
    ],
)
def test_real_definition_producer_json_replay(name, suite, dimension, instance):
    api = definition_api()
    original = pyMOFL.load(name, suite=suite, dimension=dimension, instance=instance)
    record = api.export_definition(name, suite=suite, dimension=dimension, instance=instance)
    text = json.dumps(record, sort_keys=True, allow_nan=False)
    restored = api.reconstruct_definition(json.loads(text))
    assert restored is not original
    assert restored.dimension == original.dimension
    assert record["schema_version"] == 1
    assert record["request"]["dimension"] == original.dimension
    assert record["request"]["instance"] == (1 if suite == "bbob" else None)
    assert record["original_config"]
    assert record["resolved_parameters"]
    if suite == "bbob":
        assert record["artifacts"] == []  # The factory generates the configuration.
    else:
        assert record["artifacts"]
    assert record["numerical_code"]
    assert record["integrity_sha256"]
    assert record["software"]["pymofl_version"] == pyMOFL.__version__
    # Reuse unchanged DATA01 coordinates; source-native bounds for fixed suites.
    if suite.startswith("spso"):
        X = original.operational_bounds.low[None, :]
    else:
        cases = json.loads(
            (Path(__file__).parent / "validation_data/cec/2005/f01.json").read_text()
        )["cases"]
        source_dimension = 10 if dimension == 2 else dimension
        case = next(case for case in cases if case["dimension"] == source_dimension)
        X = np.asarray([case["optimum"], case["random_input"]], dtype=np.float64)
        X = X[:, :dimension]
    before = X.copy()
    np.testing.assert_array_equal(restored.evaluate_batch(X), original.evaluate_batch(X))
    np.testing.assert_array_equal(X, before)
    assert (
        api.export_definition(name, suite=suite, dimension=dimension, instance=instance) == record
    )


@pytest.mark.parametrize(("name", "suite", "dimension", "instance"), _real_definition_requests())
def test_real_definition_breadth(name, suite, dimension, instance, request):
    api = definition_api()
    record = api.export_definition(name, suite=suite, dimension=dimension, instance=instance)
    transported = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
    restored = api.reconstruct_definition(transported)
    assert transported == record
    assert restored.dimension == record["request"]["dimension"]
    # A second real producer observes fresh config/parameters/artifacts/code;
    # compare all observations without inventing an expected manifest.
    assert (
        api.export_definition(name, suite=suite, dimension=dimension, instance=instance) == record
    )
    X = _retained_definition_points(name, suite, restored.dimension)
    request.node.user_properties.append(
        ("replay_evidence", "retained-coordinate-replay" if X is not None else "metadata-only")
    )
    if X is not None:
        original = pyMOFL.load(name, suite=suite, dimension=dimension, instance=instance)
        before = X.copy()
        np.testing.assert_array_equal(restored.evaluate_batch(X), original.evaluate_batch(X))
        np.testing.assert_array_equal(X, before)


@pytest.fixture
def produced_definition():
    return definition_api().export_definition("cec2005_f01", suite="cec2005", dimension=10)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "schema_bool",
        "missing_hash",
        "bad_hash",
        "request_missing",
        "request_shape",
        "function_type",
        "suite_type",
        "dimension_bool",
        "dimension_type",
        "dimension_zero",
        "instance_bool",
        "instance_type",
        "parameters",
        "artifacts",
        "escape",
        "software_shape",
        "python_type",
        "numpy_version",
        "python_minor",
        "platform",
        "missing_config",
        "json_type",
        "observed_json_type",
    ],
)
def test_authorized_corrupt_definition_rejected(produced_definition, mutation):
    record = copy.deepcopy(produced_definition)
    if mutation == "schema":
        record["schema_version"] = 2
    elif mutation == "schema_bool":
        record["schema_version"] = True
    elif mutation == "missing_hash":
        record.pop("integrity_sha256")
    elif mutation == "bad_hash":
        record["integrity_sha256"] = "0" * 64
    elif mutation == "request_missing":
        record.pop("request")
    elif mutation == "request_shape":
        record["request"].pop("instance")
    elif mutation in {
        "function_type",
        "suite_type",
        "dimension_bool",
        "dimension_type",
        "dimension_zero",
        "instance_bool",
        "instance_type",
    }:
        key, value = {
            "function_type": ("function_id", None),
            "suite_type": ("suite", None),
            "dimension_bool": ("dimension", True),
            "dimension_type": ("dimension", "10"),
            "dimension_zero": ("dimension", 0),
            "instance_bool": ("instance", True),
            "instance_type": ("instance", "1"),
        }[mutation]
        record["request"][key] = value
    elif mutation == "parameters":
        record["resolved_parameters"] = {}
    elif mutation == "json_type":
        record["request"]["dimension"] = 10.0
    elif mutation == "observed_json_type":
        record["resolved_parameters"]["parameters"]["dimension"] = 10.0
    elif mutation == "artifacts":
        record["artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "escape":
        record["artifacts"][0]["path"] = "../outside.json"
    elif mutation == "software_shape":
        record["software"] = None
    elif mutation == "python_type":
        record["software"]["python_version"] = None
    elif mutation == "numpy_version":
        record["software"]["numpy_version"] += ".incompatible"
    elif mutation == "python_minor":
        record["software"]["python_minor"][1] += 1
    elif mutation == "platform":
        record["software"]["platform_machine"] += ".incompatible"
    else:
        record.pop("original_config")
    if mutation not in {"missing_hash", "bad_hash"}:
        record.pop("integrity_sha256", None)
        record["integrity_sha256"] = hashlib.sha256(
            json.dumps(record, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        ).hexdigest()
    with pytest.raises((ValueError, TypeError)):
        definition_api().reconstruct_definition(record)
    assert produced_definition["request"]["dimension"] == 10


@pytest.mark.parametrize("dimension", [True, np.bool_(True), 0, -1])
def test_definition_export_rejects_invalid_dimensions(dimension):
    with pytest.raises((ValueError, TypeError)):
        definition_api().export_definition("cec2005_f01", suite="cec2005", dimension=dimension)


@pytest.mark.parametrize(
    "name,suite,instance",
    [
        ("f01", "cec2005", 1),
        ("unknown", "bbob", 1),
        ("f01", "cec2013", None),
        ("f99", "cec2005", None),
        ("f04", "cec2005", None),
    ],
)
def test_actual_unsupported_or_noisy_definition_requests(name, suite, instance):
    with pytest.raises(ValueError):
        definition_api().export_definition(name, suite=suite, dimension=10, instance=instance)


def test_actual_numeric_bbob_selector_and_instance_default():
    api = definition_api()
    record = api.export_definition(1, suite="bbob", dimension=2)
    assert record == api.export_definition("bbob_f01", suite="bbob", dimension=2, instance=1)


@pytest.mark.parametrize(
    "kind",
    [
        "nonfinite",
        "object",
        "external_enum",
        "string_array",
        "mapping_key",
        "path",
        "external_component",
    ],
)
def test_definition_observation_rejects_unsupported_state(kind):
    from enum import Enum

    from pyMOFL.functions.benchmark.sphere import SphereFunction

    class ExternalEnum(Enum):
        VALUE = 1

    class ExternalSphere(SphereFunction):
        pass

    value = {
        "nonfinite": float("inf"),
        "object": np.asarray([object()], dtype=object),
        "external_enum": ExternalEnum.VALUE,
        "string_array": np.asarray(["unsupported"]),
        "mapping_key": {1: "unsupported"},
        "path": Path("unsupported"),
        "external_component": ExternalSphere(dimension=2),
    }[kind]
    with pytest.raises(ValueError):
        definition_api()._snapshot(value)


def test_definition_observation_library_enum_and_numpy_scalars():
    from pyMOFL.core.bound_mode_enum import BoundModeEnum

    api = definition_api()
    scalar = np.int64(10)
    assert api._snapshot(scalar) == scalar.item()
    enum = next(iter(BoundModeEnum))
    observed = api._snapshot(enum)
    assert observed["name"] == enum.name
    array = api._snapshot(np.asarray([enum], dtype=object))
    assert array["enum_values"] == [observed]
    assert array["shape"] == [1]


def test_definition_artifact_confined_to_package(tmp_path):
    source = _resolve_suite_path("cec2005")
    external = tmp_path / source.name
    external.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="confined to the package"):
        definition_api()._file_record(external)
