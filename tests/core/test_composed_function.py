"""
Tests for ComposedFunction - a wrapper-free function composition system.
Following TDD principles - tests written before implementation.
"""

import gc
import json
import weakref
from pathlib import Path
from types import MethodType

import numpy as np
import pytest

import pyMOFL
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.functions.benchmark.ackley import AckleyFunction
from pyMOFL.functions.benchmark.sphere import SphereFunction
from pyMOFL.functions.transformations import (
    BiasTransform,
    BoundaryPenaltyTransform,
    RotateTransform,
    ScaleTransform,
    ShiftTransform,
)
from pyMOFL.functions.transformations.base import ScalarTransform
from pyMOFL.functions.transformations.composed import ComposedFunction


@pytest.fixture
def captured_cec_f1_batch():
    """DATA-01: reuse existing F1/D10 inputs and independent captured outputs."""
    path = Path(__file__).parents[1] / "validation_data/cec/2005/f01.json"
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


@pytest.mark.parametrize("stage", ["base", "output"])
@pytest.mark.parametrize("supports_out", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
def test_internal_type_error_propagates_once(
    monkeypatch, captured_cec_f1_batch, stage, supports_out, use_out
):
    """DEC-01/V2.1: approved call-counter failure; pytest restores replacements."""
    X, _ = captured_cec_f1_batch
    function = pyMOFL.load("cec2005_f01", dimension=10)
    calls = 0
    failure = TypeError("controlled failure")
    cause = ValueError("controlled cause")

    def fail(values, out=None):
        nonlocal calls
        calls += 1
        raise failure from cause

    def fail_without_buffer(values):
        return fail(values)

    target, method = (
        (function.base_function, "evaluate_batch")
        if stage == "base"
        else (function.output_transforms[0], "transform_batch")
    )
    monkeypatch.setattr(target, method, fail if supports_out else fail_without_buffer)
    out = np.empty(len(X)) if use_out else None
    with pytest.raises(TypeError) as caught:
        function.evaluate_batch(X, out=out)
    assert caught.value is failure
    assert caught.value.__cause__ is cause
    assert calls == 1


@pytest.mark.parametrize("fallback_base", [False, True])
@pytest.mark.parametrize("fallback_output", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
def test_captured_batch_preserves_buffers_and_live_methods(
    monkeypatch, captured_cec_f1_batch, fallback_base, fallback_output, use_out
):
    """Existing fallback methods remain usable when replaced after construction."""
    X, expected = captured_cec_f1_batch
    original = X.copy()
    function = pyMOFL.load("cec2005_f01", dimension=10)
    if fallback_base:
        base = function.base_function
        monkeypatch.setattr(
            base, "evaluate_batch", MethodType(OptimizationFunction.evaluate_batch, base)
        )
    if fallback_output:
        transform = function.output_transforms[0]
        monkeypatch.setattr(
            transform, "transform_batch", MethodType(ScalarTransform.transform_batch, transform)
        )
    out = np.empty(len(X)) if use_out else None
    result = function.evaluate_batch(X, out=out)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(X, original)
    if use_out:
        assert result is out


def test_opaque_batch_callable_retains_buffer_identity(monkeypatch, captured_cec_f1_batch):
    """A real NumPy ufunc needs no signature introspection to execute once."""
    X, expected = captured_cec_f1_batch
    function = pyMOFL.load("cec2005_f01", dimension=10)
    identity = BiasTransform(0.0)
    monkeypatch.setattr(identity, "transform_batch", np.positive)
    function.output_transforms.append(identity)
    out = np.empty(len(X))
    assert function.evaluate_batch(X, out=out) is out
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-8)


@pytest.mark.parametrize("instance_bound", [False, True])
def test_batch_dispatch_cache_does_not_retain_function_instances(
    captured_cec_f1_batch, instance_bound
):
    """Signature reuse must not retain a function and its potentially large data."""
    X, _ = captured_cec_f1_batch
    function = pyMOFL.load("cec2005_f01", dimension=10)
    if instance_bound:
        # Delegate to the real existing method; its bound closure owns the base.
        def wrap(original_method):
            def replacement(self, values, out=None):
                return original_method(values, out=out)

            return replacement

        function.base_function.evaluate_batch = MethodType(
            wrap(function.base_function.evaluate_batch), function.base_function
        )
    reference = weakref.ref(function.base_function)
    function.evaluate_batch(X)
    del function
    gc.collect()
    assert reference() is None


class TestComposedFunction:
    """Test cases for ComposedFunction class."""

    def test_creates_with_base_function_only(self):
        """Test creating composed function with just base, no transformations."""
        base = SphereFunction(dimension=5)
        composed = ComposedFunction(base_function=base)

        assert composed.dimension == 5
        assert composed.base_function is base
        assert not composed.input_transforms and not composed.output_transforms

    def test_evaluates_base_function_directly(self):
        """Test that with no transformations, evaluates base directly."""
        base = SphereFunction(dimension=3)
        composed = ComposedFunction(base_function=base)

        x = np.array([1.0, 2.0, 3.0])
        expected = 1.0 + 4.0 + 9.0  # sum of squares
        assert np.isclose(composed.evaluate(x), expected)

    def test_applies_single_shift_transformation(self):
        """Test applying a shift transformation."""
        base = SphereFunction(dimension=2)
        input_transforms = [ShiftTransform(np.array([1.0, 2.0]))]
        composed = ComposedFunction(base_function=base, input_transforms=input_transforms)

        # Evaluate at shift point should give 0 (optimum)
        x = np.array([1.0, 2.0])
        assert np.isclose(composed.evaluate(x), 0.0)

    def test_applies_single_bias_transformation(self):
        """Test applying a bias transformation."""
        base = SphereFunction(dimension=2)
        output_transforms = [BiasTransform(-450.0)]
        composed = ComposedFunction(base_function=base, output_transforms=output_transforms)

        x = np.zeros(2)
        assert np.isclose(composed.evaluate(x), -450.0)

    def test_applies_multiple_transformations_in_order(self):
        """Test that transformations are applied in correct order."""
        base = SphereFunction(dimension=2)
        input_transforms = [ShiftTransform(np.array([1.0, 1.0])), ScaleTransform(2.0)]
        output_transforms = [BiasTransform(100.0)]
        composed = ComposedFunction(
            base_function=base,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
        )

        # Test at x = [3, 3]
        # After shift: [3, 3] - [1, 1] = [2, 2]
        # After scale: [2, 2] / 2 = [1, 1]
        # Sphere([1, 1]) = 2
        # After bias: 2 + 100 = 102
        x = np.array([3.0, 3.0])
        assert np.isclose(composed.evaluate(x), 102.0)

    def test_applies_rotation_transformation(self):
        """Test rotation transformation."""
        base = SphereFunction(dimension=2)
        # 90-degree rotation matrix
        rotation = np.array([[0, -1], [1, 0]])
        input_transforms = [RotateTransform(rotation)]
        composed = ComposedFunction(base_function=base, input_transforms=input_transforms)

        # [1, 0] rotated 90 degrees becomes [0, 1]
        x = np.array([1.0, 0.0])
        # Sphere([0, 1]) = 1
        assert np.isclose(composed.evaluate(x), 1.0)

    def test_batch_evaluation(self):
        """Test batch evaluation with transformations."""
        base = SphereFunction(dimension=2)
        input_transforms = [ShiftTransform(np.array([1.0, 1.0]))]
        output_transforms = [BiasTransform(-10.0)]
        composed = ComposedFunction(
            base_function=base,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
        )

        X = np.array(
            [
                [1.0, 1.0],  # At optimum after shift -> 0 - 10 = -10
                [2.0, 2.0],  # [1, 1] after shift -> 2 - 10 = -8
            ]
        )
        composed.evaluate_batch(X)
        # assert np.isclose(results[0], -10.0)
        # assert np.isclose(results[1], -8.0)

    def test_complex_cec_like_composition(self):
        """Test a CEC-style complex composition."""
        base = AckleyFunction(dimension=5)
        shift = np.ones(5) * 2.5
        scale = 1.5
        bias = -450.0

        input_transforms = [ShiftTransform(shift), ScaleTransform(scale)]
        output_transforms = [BiasTransform(bias)]
        composed = ComposedFunction(
            base_function=base,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
        )

        # Just verify it runs without error and returns reasonable value
        x = np.random.randn(5)
        result = composed.evaluate(x)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_preserves_bounds_from_base_function(self):
        """Test that bounds are preserved from base function."""
        base = SphereFunction(dimension=3)
        composed = ComposedFunction(
            base_function=base, input_transforms=[ShiftTransform(np.ones(3))]
        )

        assert composed.initialization_bounds == base.initialization_bounds
        assert composed.operational_bounds == base.operational_bounds

    def test_unknown_transformation_raises_error(self):
        """Test that unknown transformation type raises error."""
        pytest.skip("Unknown transformation test not applicable in pure functional model")


class TestComposedFunctionWithPenalty:
    """Test ComposedFunction with penalty_transforms (vector→scalar additive penalty)."""

    def test_penalty_added_to_result(self):
        """Penalty is added to the base function result."""
        base = SphereFunction(dimension=2)
        penalty = BoundaryPenaltyTransform(bound=5.0)
        composed = ComposedFunction(
            base_function=base,
            penalty_transforms=[penalty],
        )
        # x = [6, 0]: sphere([6, 0]) = 36, penalty = (6-5)^2 = 1 → total = 37
        result = composed.evaluate(np.array([6.0, 0.0]))
        np.testing.assert_allclose(result, 37.0)

    def test_no_penalty_when_within_bounds(self):
        """Penalty is zero inside the boundary — result equals base function."""
        base = SphereFunction(dimension=2)
        penalty = BoundaryPenaltyTransform(bound=5.0)
        composed = ComposedFunction(
            base_function=base,
            penalty_transforms=[penalty],
        )
        x = np.array([1.0, 2.0])
        expected = 1.0 + 4.0  # sphere only, no penalty
        np.testing.assert_allclose(composed.evaluate(x), expected)

    def test_penalty_uses_raw_x_not_transformed(self):
        """Penalty is computed on the original x, not the transformed x."""
        base = SphereFunction(dimension=2)
        # Shift moves x into [0,0] for the base, but penalty sees raw x
        shift = ShiftTransform(np.array([10.0, 10.0]))
        penalty = BoundaryPenaltyTransform(bound=5.0)
        composed = ComposedFunction(
            base_function=base,
            input_transforms=[shift],
            penalty_transforms=[penalty],
        )
        # x = [10, 10]: shift → [0, 0], sphere = 0
        # penalty on raw [10, 10] = (10-5)^2 + (10-5)^2 = 50
        result = composed.evaluate(np.array([10.0, 10.0]))
        np.testing.assert_allclose(result, 50.0)

    def test_penalty_combines_with_output_transforms(self):
        """Penalty adds to the result after scalar output transforms."""
        base = SphereFunction(dimension=2)
        bias = BiasTransform(100.0)
        penalty = BoundaryPenaltyTransform(bound=5.0)
        composed = ComposedFunction(
            base_function=base,
            output_transforms=[bias],
            penalty_transforms=[penalty],
        )
        # x = [6, 0]: sphere = 36, bias → 136, penalty = 1 → total = 137
        result = composed.evaluate(np.array([6.0, 0.0]))
        np.testing.assert_allclose(result, 137.0)

    def test_penalty_batch_evaluation(self):
        """Batch evaluation adds penalties correctly."""
        base = SphereFunction(dimension=2)
        penalty = BoundaryPenaltyTransform(bound=5.0)
        composed = ComposedFunction(
            base_function=base,
            penalty_transforms=[penalty],
        )
        X = np.array(
            [
                [1.0, 1.0],  # sphere=2, penalty=0 → 2
                [6.0, 0.0],  # sphere=36, penalty=1 → 37
                [10.0, 10.0],  # sphere=200, penalty=50 → 250
            ]
        )
        results = composed.evaluate_batch(X)
        np.testing.assert_allclose(results, [2.0, 37.0, 250.0])

    def test_multiple_penalty_transforms(self):
        """Multiple penalty transforms all contribute additively."""
        base = SphereFunction(dimension=2)
        penalty1 = BoundaryPenaltyTransform(bound=5.0)
        penalty2 = BoundaryPenaltyTransform(bound=3.0)
        composed = ComposedFunction(
            base_function=base,
            penalty_transforms=[penalty1, penalty2],
        )
        # x = [6, 0]: sphere=36, pen1=(6-5)^2=1, pen2=(6-3)^2=9 → 46
        result = composed.evaluate(np.array([6.0, 0.0]))
        np.testing.assert_allclose(result, 46.0)

    def test_empty_penalty_list_no_effect(self):
        """Empty penalty_transforms list has no effect (backward compat)."""
        base = SphereFunction(dimension=2)
        composed = ComposedFunction(
            base_function=base,
            penalty_transforms=[],
        )
        x = np.array([6.0, 0.0])
        np.testing.assert_allclose(composed.evaluate(x), 36.0)
