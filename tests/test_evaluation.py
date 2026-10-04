"""V7.1.1: unchanged DATA01, real array views and observation-only delegation."""

import json
from pathlib import Path

import numpy as np
import pytest

import pyMOFL
from pyMOFL.core.function import OptimizationFunction


@pytest.fixture
def reference_batch():
    path = Path(__file__).parent / "validation_data/cec/2005/f01.json"
    case = next(c for c in json.loads(path.read_text())["cases"] if c["dimension"] == 10)
    X = np.asarray(
        [
            case["optimum"],
            case["random_input"],
            case["operational_bounds"]["low"],
            case["operational_bounds"]["high"],
        ],
        dtype=np.float64,
    )
    expected = np.asarray([case["outputs"][k] for k in ("optimum", "random", "lower", "upper")])
    return X, expected


class ObservedFunction(OptimizationFunction):
    """Observe owned chunks while delegating every numerical value to real F1."""

    def __init__(self, original):
        super().__init__(original.shape[1])
        self.original = original
        self.delegate = pyMOFL.load("cec2005_f01", dimension=self.dimension)
        self.rows = []

    def evaluate(self, x):
        return self.delegate.evaluate(x)

    def evaluate_batch(self, X):
        assert not np.shares_memory(X, self.original)
        assert X.dtype == np.float64
        self.rows.append(len(X))
        return self.delegate.evaluate_batch(X)


def helper():
    candidate = getattr(pyMOFL, "evaluate_chunks", None)
    assert callable(candidate), "public evaluate_chunks is unavailable"
    return candidate


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
@pytest.mark.parametrize("layout", ["original", "reverse", "readonly", "fortran"])
def test_real_values_bounded_rows_and_ownership(reference_batch, chunk_size, layout):
    X, expected = reference_batch
    if layout == "reverse":
        X, expected = X[::-1], expected[::-1]
    elif layout == "readonly":
        X.flags.writeable = False
    elif layout == "fortran":
        X = np.asfortranarray(X)
    before = X.copy()
    function = ObservedFunction(X)
    result = helper()(function, X, chunk_size, deterministic=True, batch_independent=True)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-8)
    np.testing.assert_array_equal(X, before)
    assert result.dtype == np.float64
    assert sum(function.rows) == len(X)
    assert max(function.rows) <= chunk_size
    assert function.rows == [
        min(chunk_size, len(X) - start) for start in range(0, len(X), chunk_size)
    ]


def test_real_output_reuse_and_empty_slice(reference_batch):
    X, expected = reference_batch
    function = ObservedFunction(X)
    output = np.empty(len(X), dtype=np.float64)
    result = helper()(function, X, 3, deterministic=True, batch_independent=True, out=output)
    assert result is output
    np.testing.assert_allclose(output, expected, rtol=1e-12, atol=1e-8)
    calls = list(function.rows)
    result = helper()(
        function, X[:0], 3, deterministic=True, batch_independent=True, out=output[:0]
    )
    assert result.shape == (0,)
    assert function.rows == calls


def test_actual_vector_and_overlapping_view_rejected_before_call(reference_batch):
    X, _ = reference_batch
    function = ObservedFunction(X)
    for output in (X[0], X[:, 0]):
        with pytest.raises(ValueError):
            helper()(function, X, 2, deterministic=True, batch_independent=True, out=output)
        assert function.rows == []


def test_actual_integral_bounds_and_reversed_output_buffer(reference_batch):
    X, expected = reference_batch
    integral = X[2:].astype(np.int64)
    np.testing.assert_array_equal(integral, X[2:])
    function = ObservedFunction(integral)
    output = expected[2:].copy()[::-1]
    result = helper()(function, integral, 1, deterministic=True, batch_independent=True, out=output)
    assert result is output
    np.testing.assert_allclose(result, expected[2:], rtol=1e-12, atol=1e-8)
    assert function.rows == [1, 1]


@pytest.mark.parametrize("size", [0, -1, True, np.bool_(True), 1.5, "2"])
def test_authorized_invalid_chunk_sizes(reference_batch, size):
    X, _ = reference_batch
    function = ObservedFunction(X)
    with pytest.raises((ValueError, TypeError)):
        helper()(function, X, size, deterministic=True, batch_independent=True)
    assert function.rows == []


@pytest.mark.parametrize("declarations", [(False, True), (True, False), (1, True), (True, None)])
def test_authorized_invalid_declarations(reference_batch, declarations):
    X, _ = reference_batch
    function = ObservedFunction(X)
    with pytest.raises(ValueError):
        helper()(function, X, 2, deterministic=declarations[0], batch_independent=declarations[1])
    assert function.rows == []


@pytest.mark.parametrize(
    "kind", ["list", "complex", "bool", "object", "string", "vector", "dimension"]
)
def test_authorized_invalid_inputs(reference_batch, kind):
    X, _ = reference_batch
    function = ObservedFunction(X)
    if kind == "list":
        invalid = X.tolist()
    elif kind == "vector":
        invalid = X[0]
    elif kind == "dimension":
        invalid = X[:, :-1]
    else:
        invalid = X.astype(
            {"complex": complex, "bool": bool, "object": object, "string": str}[kind]
        )
    with pytest.raises((ValueError, TypeError)):
        helper()(function, invalid, 2, deterministic=True, batch_independent=True)
    assert function.rows == []


@pytest.mark.parametrize("kind", ["list", "float32", "shape", "readonly", "self_overlap"])
def test_authorized_invalid_outputs(reference_batch, kind):
    X, expected = reference_batch
    function = ObservedFunction(X)
    output = expected.copy()
    if kind == "list":
        output = output.tolist()
    elif kind == "float32":
        output = output.astype(np.float32)
    elif kind == "shape":
        output = output[:, None]
    elif kind == "readonly":
        output.flags.writeable = False
    else:
        output = np.lib.stride_tricks.as_strided(output, shape=output.shape, strides=(0,))
    with pytest.raises((ValueError, TypeError)):
        helper()(function, X, 2, deterministic=True, batch_independent=True, out=output)
    assert function.rows == []


@pytest.mark.parametrize("kind", ["list", "complex", "bool", "object", "string", "shape"])
def test_authorized_invalid_child_results(reference_batch, kind):
    X, _ = reference_batch

    class InvalidResult(ObservedFunction):
        def evaluate_batch(self, chunk):
            actual = super().evaluate_batch(chunk)
            if kind == "list":
                return actual.tolist()
            if kind == "shape":
                return actual[:, None]
            return actual.astype(
                {"complex": complex, "bool": bool, "object": object, "string": str}[kind]
            )

    function = InvalidResult(X)
    with pytest.raises((ValueError, TypeError)):
        helper()(function, X, 2, deterministic=True, batch_independent=True)
    assert function.rows == [2]


def test_authorized_later_failure_preserves_completed_chunks(reference_batch):
    X, expected = reference_batch
    failure = TypeError("authorized later-chunk failure")

    class LaterFailure(ObservedFunction):
        def evaluate_batch(self, chunk):
            if len(self.rows) == 1:
                self.rows.append(len(chunk))
                raise failure
            return super().evaluate_batch(chunk)

    function = LaterFailure(X)
    output = expected[::-1].copy()
    untouched = output.copy()
    before = X.copy()
    with pytest.raises(TypeError) as observed:
        helper()(function, X, 2, deterministic=True, batch_independent=True, out=output)
    assert observed.value is failure
    assert function.rows == [2, 2]
    np.testing.assert_allclose(output[:2], expected[:2], rtol=1e-12, atol=1e-8)
    np.testing.assert_array_equal(output[2:], untouched[2:])
    np.testing.assert_array_equal(X, before)
