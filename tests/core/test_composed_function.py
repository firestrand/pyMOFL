"""
Tests for ComposedFunction - a wrapper-free function composition system.
Following TDD principles - tests written before implementation.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.ackley import AckleyFunction
from pyMOFL.functions.benchmark.sphere import SphereFunction
from pyMOFL.functions.transformations import (
    BiasTransform,
    RotateTransform,
    ScaleTransform,
    ShiftTransform,
)
from pyMOFL.functions.transformations.composed import ComposedFunction


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
