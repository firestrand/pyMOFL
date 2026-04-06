"""Tests for TransformBuilder — construct transform objects from parsed configs."""

import numpy as np
import pytest

from pyMOFL.factories.data_loader import DataLoader
from pyMOFL.factories.transform_builder import TransformBuilder
from pyMOFL.functions.transformations import (
    AsymmetricTransform,
    BiasTransform,
    BoundaryPenaltyTransform,
    IndexedScaleTransform,
    NoiseTransform,
    NonContinuousTransform,
    NormalizeTransform,
    OffsetTransform,
    OscillationTransform,
    PenaltyTransform,
    RotateTransform,
    ScalarTransform,
    ScaleTransform,
    ShiftTransform,
    VectorTransform,
)


@pytest.fixture
def builder():
    return TransformBuilder(DataLoader())


class TestVectorTransforms:
    """Test building vector (input) transforms."""

    def test_shift_from_literal_vector(self, builder):
        t = builder.build("shift", {"vector": [1.0, 2.0, 3.0]}, dimension=3)
        assert isinstance(t, ShiftTransform)
        np.testing.assert_array_equal(t.shift, [1.0, 2.0, 3.0])

    def test_shift_from_scalar(self, builder):
        t = builder.build("shift", {"vector": 5.0}, dimension=3)
        assert isinstance(t, ShiftTransform)
        np.testing.assert_array_equal(t.shift, [5.0, 5.0, 5.0])

    def test_shift_defaults_to_zero(self, builder):
        t = builder.build("shift", {}, dimension=4)
        assert isinstance(t, ShiftTransform)
        np.testing.assert_array_equal(t.shift, np.zeros(4))

    def test_offset_from_scalar(self, builder):
        t = builder.build("offset", {"value": 1.0}, dimension=3)
        assert isinstance(t, OffsetTransform)
        np.testing.assert_array_equal(t.offset, [1.0, 1.0, 1.0])

    def test_offset_from_vector(self, builder):
        t = builder.build("offset", {"vector": [1.0, 2.0]}, dimension=2)
        assert isinstance(t, OffsetTransform)
        np.testing.assert_array_equal(t.offset, [1.0, 2.0])

    def test_rotate_from_identity(self, builder):
        t = builder.build("rotate", {"matrix": "identity"}, dimension=3)
        assert isinstance(t, RotateTransform)
        np.testing.assert_array_equal(t.matrix, np.eye(3))

    def test_rotate_from_literal(self, builder):
        mat = [[1, 0], [0, 1]]
        t = builder.build("rotate", {"matrix": mat}, dimension=2)
        assert isinstance(t, RotateTransform)
        np.testing.assert_array_equal(t.matrix, np.eye(2))

    def test_scale_from_factor(self, builder):
        t = builder.build("scale", {"factor": 2.5}, dimension=2)
        assert isinstance(t, ScaleTransform)
        assert t.factor == 2.5

    def test_scale_defaults_to_1(self, builder):
        t = builder.build("scale", {}, dimension=2)
        assert isinstance(t, ScaleTransform)
        assert t.factor == 1.0

    def test_scale_from_lambda_alias(self, builder):
        t = builder.build("scale", {"lambda": 3.0}, dimension=2)
        assert isinstance(t, ScaleTransform)
        assert t.factor == 3.0

    def test_indexed_scale(self, builder):
        t = builder.build(
            "indexed_scale",
            {"factors": [1.0, 2.0, 3.0], "component_index": 1},
            dimension=3,
        )
        assert isinstance(t, IndexedScaleTransform)

    def test_non_continuous(self, builder):
        t = builder.build("non_continuous", {}, dimension=5)
        assert isinstance(t, NonContinuousTransform)

    def test_oscillation(self, builder):
        t = builder.build("oscillation", {}, dimension=5)
        assert isinstance(t, OscillationTransform)
        assert isinstance(t, VectorTransform)

    def test_oscillation_t_osz_alias(self, builder):
        t = builder.build("t_osz", {}, dimension=5)
        assert isinstance(t, OscillationTransform)

    def test_asymmetric(self, builder):
        t = builder.build("asymmetric", {"beta": 0.5}, dimension=5)
        assert isinstance(t, AsymmetricTransform)
        assert isinstance(t, VectorTransform)
        assert t.beta == 0.5
        assert t.dimension == 5

    def test_asymmetric_t_asy_alias(self, builder):
        t = builder.build("t_asy", {"beta": 0.3}, dimension=10)
        assert isinstance(t, AsymmetricTransform)
        assert t.beta == 0.3

    def test_asymmetric_default_beta(self, builder):
        t = builder.build("asymmetric", {}, dimension=5)
        assert isinstance(t, AsymmetricTransform)
        assert t.beta == 0.2

    def test_oscillation_produces_correct_output(self, builder):
        """Verify that the built OscillationTransform produces correct values."""
        t = builder.build("oscillation", {}, dimension=3)
        x = np.array([1.0, -1.0, 0.0])
        result = t(x)
        np.testing.assert_allclose(result, [1.0, -1.0, 0.0], rtol=1e-14)

    def test_asymmetric_produces_correct_output(self, builder):
        """Verify that the built AsymmetricTransform produces correct values."""
        t = builder.build("asymmetric", {"beta": 0.5}, dimension=5)
        x = np.array([0.0, 0.0, 0.0, 0.0, 4.0])
        result = t(x)
        # last element: ratio=1, 4^(1+0.5*1*2) = 4^2 = 16
        np.testing.assert_allclose(result[4], 16.0, rtol=1e-14)


class TestPenaltyTransforms:
    """Test building penalty transforms."""

    def test_boundary_penalty(self, builder):
        t = builder.build("boundary_penalty", {}, dimension=5)
        assert isinstance(t, BoundaryPenaltyTransform)
        assert isinstance(t, PenaltyTransform)
        assert t.bound == 5.0

    def test_boundary_penalty_custom_bound(self, builder):
        t = builder.build("boundary_penalty", {"bound": 3.0}, dimension=5)
        assert isinstance(t, BoundaryPenaltyTransform)
        assert t.bound == 3.0

    def test_f_pen_alias(self, builder):
        t = builder.build("f_pen", {}, dimension=5)
        assert isinstance(t, BoundaryPenaltyTransform)

    def test_boundary_penalty_produces_correct_output(self, builder):
        """Verify that the built BoundaryPenaltyTransform produces correct values."""
        t = builder.build("boundary_penalty", {"bound": 5.0}, dimension=3)
        x = np.array([6.0, 0.0, 0.0])
        np.testing.assert_allclose(t(x), 1.0)

    def test_boundary_penalty_not_vector_or_scalar(self, builder):
        """BoundaryPenalty is a PenaltyTransform, not VectorTransform or ScalarTransform."""
        t = builder.build("boundary_penalty", {}, dimension=5)
        assert not isinstance(t, VectorTransform)
        assert not isinstance(t, ScalarTransform)


class TestScalarTransforms:
    """Test building scalar (output) transforms."""

    def test_bias(self, builder):
        t = builder.build("bias", {"value": -450.0}, dimension=10)
        assert isinstance(t, BiasTransform)
        assert t.bias == -450.0

    def test_bias_defaults_to_zero(self, builder):
        t = builder.build("bias", {}, dimension=2)
        assert isinstance(t, BiasTransform)
        assert t.bias == 0.0

    def test_noise(self, builder):
        t = builder.build("noise", {"level": 0.5, "seed": 42}, dimension=2)
        assert isinstance(t, NoiseTransform)

    def test_normalize(self, builder):
        t = builder.build("normalize", {"C": 1000.0}, dimension=2)
        assert isinstance(t, NormalizeTransform)


class TestBuildMany:
    """Test building multiple transforms at once."""

    def test_separates_input_and_output(self, builder):
        transforms = [
            ("shift", {"vector": [1.0, 2.0]}),
            ("bias", {"value": -100.0}),
            ("rotate", {"matrix": "identity"}),
        ]
        input_t, output_t, penalty_t = builder.build_many(transforms, dimension=2)

        assert len(input_t) == 2
        assert all(isinstance(t, VectorTransform) for t in input_t)
        assert len(output_t) == 1
        assert all(isinstance(t, ScalarTransform) for t in output_t)
        assert len(penalty_t) == 0

    def test_preserves_order(self, builder):
        """Transforms maintain the input order within each category."""
        transforms = [
            ("shift", {"vector": [1.0, 2.0]}),
            ("rotate", {"matrix": "identity"}),
            ("bias", {"value": -100.0}),
            ("noise", {"level": 0.4}),
        ]
        input_t, output_t, penalty_t = builder.build_many(transforms, dimension=2)

        assert isinstance(input_t[0], ShiftTransform)
        assert isinstance(input_t[1], RotateTransform)
        assert isinstance(output_t[0], BiasTransform)
        assert isinstance(output_t[1], NoiseTransform)
        assert len(penalty_t) == 0

    def test_separates_penalty_transforms(self, builder):
        """Penalty transforms are separated into their own group."""
        transforms = [
            ("shift", {"vector": [1.0, 2.0]}),
            ("boundary_penalty", {"bound": 5.0}),
            ("bias", {"value": -100.0}),
        ]
        input_t, output_t, penalty_t = builder.build_many(transforms, dimension=2)

        assert len(input_t) == 1
        assert len(output_t) == 1
        assert len(penalty_t) == 1
        assert all(isinstance(t, PenaltyTransform) for t in penalty_t)

    def test_empty_list(self, builder):
        input_t, output_t, penalty_t = builder.build_many([], dimension=2)
        assert input_t == []
        assert output_t == []
        assert penalty_t == []


class TestUnknownType:
    """Test error handling for unknown transform types."""

    def test_raises_on_unknown(self, builder):
        with pytest.raises(ValueError, match="Unknown transform type"):
            builder.build("unknown_transform", {}, dimension=2)
