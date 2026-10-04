"""Selected SPSO source cases; explicit external reference setup, no downloads."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import pyMOFL
from pyMOFL.factories.function_factory import FunctionFactory
from tests.utils.benchmark_validation import BenchmarkValidator


@pytest.mark.parametrize("steps", [[-1.0], [float("nan")], [float("inf")], [[1.0]]])
def test_authorized_quantization_invalid_steps(steps):
    from pyMOFL.functions.transformations.half_up_quantization import HalfUpQuantizationTransform

    with pytest.raises(ValueError, match="finite nonnegative"):
        HalfUpQuantizationTransform(steps)


def test_authorized_quantization_shape_guards():
    from pyMOFL.functions.transformations.half_up_quantization import HalfUpQuantizationTransform

    suite = pyMOFL.get_suite("spso2011")
    actual = suite["f21"].operational_bounds.low
    quantizer = HalfUpQuantizationTransform([1.0, 0.0, 0.001])
    with pytest.raises(ValueError, match="same shape"):
        quantizer(actual[:-1])
    with pytest.raises(ValueError, match="shape"):
        quantizer.transform_batch(actual)
    with pytest.raises(ValueError, match="shape"):
        quantizer.transform_batch(actual[None, :-1])


def test_authorized_spring_penalty_version_rejection():
    from pyMOFL.functions.benchmark.spso_compression_spring import SPSOCompressionSpringFunction

    with pytest.raises(ValueError):
        SPSOCompressionSpringFunction(penalty_version="unsupported")


@pytest.mark.parametrize("year", [2007, 2011])
def test_authorized_source_boundary_captures(year):
    configured = os.environ.get("PYMOFL_SPSO_BOUNDARY_PATH")
    if configured is None:
        pytest.skip("explicit authorized source boundary captures are not provisioned")
    root = Path(configured)
    provenance = next(
        row
        for row in json.loads((root / "boundary-provenance.json").read_text())
        if row["year"] == year
    )
    expected_archive = {
        2007: "f9524f7f9568009b4ab5c76cd32d91c255fef978b4ff64891b460cb520f34bd1",
        2011: "11692f658158b18aafd97d667eeebdc7527cf21147d530d73d4c7eb795af0557",
    }[year]
    assert provenance["archive_sha256"] == expected_archive
    directory = root / str(year)
    controls = (directory / "release.jsonl").read_bytes()
    expected_control_hash = {
        2007: "422812db86a36dcdee25526dfea724f4698bc2a107feb8772a5c6aeef64c73cf",
        2011: "3d338e9d7f4b9e54cbbbd07957b46fbf6ca494ed1aa1a668742bac4de17183ef",
    }[year]
    assert hashlib.sha256(controls).hexdigest() == expected_control_hash
    metadata = {
        row["func_id"]: row for row in map(json.loads, controls.splitlines()) if "case" not in row
    }
    dimensions = {4: 2, 11: 42, 18: 4, 21: 3}
    counts = {4: 9, 11: 228, 18: 24, 21: 12}
    assert set(metadata) == set(dimensions)
    assert len(provenance["boundary_inputs"]) == 4
    assert {item["fid"] for item in provenance["boundary_inputs"]} == set(dimensions)
    release = (directory / "boundary-release.jsonl").read_bytes()
    assert release == (directory / "boundary-ubsan.jsonl").read_bytes()
    assert hashlib.sha256(release).hexdigest() == provenance["boundary_release_sha256"]
    assert (
        hashlib.sha256((directory / "boundary-driver.c").read_bytes()).hexdigest()
        == provenance["boundary_driver_sha256"]
    )
    rows = [json.loads(line) for line in release.splitlines()]
    assert len(rows) == provenance["boundary_cases"] == 273
    assert {row["func_id"] for row in rows} == set(dimensions)
    suite = pyMOFL.get_suite(f"spso{year}")
    for record in provenance["boundary_inputs"]:
        fid = record["fid"]
        payload = (directory / f"boundary-f{fid:02d}.txt").read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        tokens = payload.decode().split()
        source_fid, dimension, count = map(int, tokens[:3])
        assert source_fid == fid
        assert dimension == dimensions[fid]
        assert count == counts[fid]
        cases = [row for row in rows if row["func_id"] == fid]
        assert len(cases) == count
        assert [row["case"] for row in cases] == list(range(count))
        X = np.asarray([row["x"] for row in cases], dtype=np.float64)
        native = metadata[fid]
        if fid == 4:
            prescribed = [
                [a, b]
                for a in (native["low"][0], 0.0, native["high"][0])
                for b in (native["low"][1], 0.0, native["high"][1])
            ]
        else:
            prescribed = []
            active = [i for i, step in enumerate(native["steps"]) if step > 1e-40]
            assert len(active) == {11: 38, 18: 4, 21: 2}[fid]
            for coordinate in active:
                step = native["steps"][coordinate]
                ties = [
                    (np.floor(native["low"][coordinate] / step) + 0.5) * step,
                    (np.floor(native["high"][coordinate] / step) - 0.5) * step,
                ]
                for tie in ties:
                    for point_value in (np.nextafter(tie, -np.inf), tie, np.nextafter(tie, np.inf)):
                        point = list(native["low"])
                        point[coordinate] = point_value
                        prescribed.append(point)
        np.testing.assert_array_equal(X, prescribed)
        np.testing.assert_array_equal(
            X, np.asarray(tokens[3:], dtype=np.float64).reshape(count, dimension)
        )
        before = X.copy()
        function = suite[f"f{fid:02d}"]
        quantized = X.copy()
        for transform in function.input_transforms:
            quantized = transform.transform_batch(quantized)
        np.testing.assert_array_equal(quantized, [row["quantized_x"] for row in cases])
        expected = np.asarray([row["value"] for row in cases], dtype=np.float64)
        np.testing.assert_allclose(function.evaluate_batch(X), expected, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(
            [function.evaluate(x) for x in X], expected, rtol=1e-12, atol=1e-10
        )
        np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("year", [2007, 2011])
def test_selected_suite_public_availability(year):
    try:
        suite = pyMOFL.get_suite(f"spso{year}")
    except ValueError:
        pytest.fail(f"public selected SPSO{year} suite is unavailable")
    assert [function.dimension for function in suite] == [2, 42, 4, 3]
    assert [function.function_id for function in suite] == [
        f"spso{year}_f{fid:02d}" for fid in (4, 11, 18, 21)
    ]


@pytest.fixture(params=[2007, 2011])
def captured_source(request):
    root = os.environ.get("PYMOFL_SPSO_REFERENCE_PATH")
    if root is None:
        pytest.skip("explicit PYMOFL_SPSO_REFERENCE_PATH source captures not provisioned")
    path = Path(root) / str(request.param) / "release.jsonl"
    expected_hash = {
        2007: "422812db86a36dcdee25526dfea724f4698bc2a107feb8772a5c6aeef64c73cf",
        2011: "3d338e9d7f4b9e54cbbbd07957b46fbf6ca494ed1aa1a668742bac4de17183ef",
    }[request.param]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_hash
    rows = [json.loads(line) for line in data.splitlines()]
    return request.param, rows


def test_actual_source_scalar_batch_bounds_quantization(captured_source):
    year, rows = captured_source
    suite = pyMOFL.get_suite(f"spso{year}")
    for metadata in (row for row in rows if "case" not in row):
        fid = metadata["func_id"]
        function = suite[f"f{fid:02d}"]
        loaded = pyMOFL.load(f"spso{year}_f{fid:02d}")
        cases = [row for row in rows if row["func_id"] == fid and "case" in row]
        X = np.asarray([row["x"] for row in cases], dtype=np.float64)
        expected = np.asarray([row["value"] for row in cases], dtype=np.float64)
        original = X.copy()
        quantized = X.copy()
        for transform in function.input_transforms:
            np.testing.assert_array_equal(transform(X[0]), cases[0]["quantized_x"])
            buffer = np.empty_like(quantized)
            assert transform.transform_batch(quantized, out=buffer) is buffer
            quantized = buffer
        np.testing.assert_array_equal(quantized, [row["quantized_x"] for row in cases])
        actual = function.evaluate_batch(X)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(
            [loaded.evaluate(point) for point in X], expected, rtol=1e-12, atol=1e-10
        )
        np.testing.assert_array_equal(X, original)
        assert function.evaluate_batch(X[:0]).shape == (0,)
        BenchmarkValidator.assert_bounds_set(function)
        np.testing.assert_array_equal(function.operational_bounds.low, metadata["low"])
        np.testing.assert_array_equal(function.operational_bounds.high, metadata["high"])
        np.testing.assert_array_equal(function.initialization_bounds.low, metadata["low"])
        np.testing.assert_array_equal(function.initialization_bounds.high, metadata["high"])
        assert loaded.dimension == metadata["dim"]


def test_fixed_factory_keeps_actual_config_owned(captured_source):
    year, rows = captured_source
    config_path = (
        Path(pyMOFL.__file__).parent / "constants" / f"spso{year}" / f"spso{year}_suite.json"
    )
    payload = json.loads(config_path.read_text())
    factory = FunctionFactory()
    before = json.dumps(payload, sort_keys=True)
    for entry in payload["functions"]:
        function = factory.create_function(entry["function"], fixed_dimension=entry["dimension"])
        fid = int(entry["id"].rsplit("f", 1)[1])
        cases = [row for row in rows if row["func_id"] == fid and "case" in row]
        X = np.asarray([row["x"] for row in cases], dtype=np.float64)
        expected = np.asarray([row["value"] for row in cases], dtype=np.float64)
        np.testing.assert_allclose(function.evaluate_batch(X), expected, rtol=1e-12, atol=1e-10)
        assert function.dimension == entry["dimension"]
    assert json.dumps(payload, sort_keys=True) == before
