"""Tests for TransformBuilder — construct transform objects from parsed configs."""

import numpy as np
import pytest

from pyMOFL.factories.data_loader import DataLoader
from pyMOFL.factories.transform_builder import TransformBuilder
from pyMOFL.functions.transformations import (
    BiasTransform,
    IndexedScaleTransform,
    NoiseTransform,
    NonContinuousTransform,
    NormalizeTransform,
    OffsetTransform,
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
        input_t, output_t = builder.build_many(transforms, dimension=2)

        assert len(input_t) == 2
        assert all(isinstance(t, VectorTransform) for t in input_t)
        assert len(output_t) == 1
        assert all(isinstance(t, ScalarTransform) for t in output_t)

    def test_preserves_order(self, builder):
        """Transforms maintain the input order within each category."""
        transforms = [
            ("shift", {"vector": [1.0, 2.0]}),
            ("rotate", {"matrix": "identity"}),
            ("bias", {"value": -100.0}),
            ("noise", {"level": 0.4}),
        ]
        input_t, output_t = builder.build_many(transforms, dimension=2)

        assert isinstance(input_t[0], ShiftTransform)
        assert isinstance(input_t[1], RotateTransform)
        assert isinstance(output_t[0], BiasTransform)
        assert isinstance(output_t[1], NoiseTransform)

    def test_empty_list(self, builder):
        input_t, output_t = builder.build_many([], dimension=2)
        assert input_t == []
        assert output_t == []


class TestUnknownType:
    """Test error handling for unknown transform types."""

    def test_raises_on_unknown(self, builder):
        with pytest.raises(ValueError, match="Unknown transform type"):
            builder.build("unknown_transform", {}, dimension=2)
