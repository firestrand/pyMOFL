"""
Tests for indexed transformation decorators.

These decorators support component-specific parameters in compositions.
"""

import numpy as np

from pyMOFL.functions.benchmark.sphere import SphereFunction
from pyMOFL.functions.transformations import (
    IndexedRotateTransform,
    IndexedScaleTransform,
    IndexedShiftTransform,
)
from pyMOFL.functions.transformations.composed import ComposedFunction


class TestIndexedScale:
    """Test indexed scale transformation."""

    def test_single_factor(self):
        """Test with single scale factor (backwards compatible)."""
        base = SphereFunction(dimension=5)
        scaled = ComposedFunction(
            base_function=base, input_transforms=[IndexedScaleTransform(factors=2.0)]
        )

        x = np.ones(5)
        # Should scale by 1/2 = 0.5
        expected = base.evaluate(x * 0.5)
        actual = scaled.evaluate(x)

        assert np.isclose(actual, expected)

    def test_indexed_factors(self):
        """Test with array of scale factors and component index."""
        base = SphereFunction(dimension=5)
        factors = [1.0, 2.0, 0.5, 4.0]

        # Test component 1 (factor = 2.0)
        scaled = ComposedFunction(
            base_function=base,
            input_transforms=[IndexedScaleTransform(factors=factors, component_index=1)],
        )

        x = np.ones(5)
        # Should scale by 1/2.0 = 0.5
        expected = base.evaluate(x * 0.5)
        actual = scaled.evaluate(x)

        assert np.isclose(actual, expected)

    def test_out_of_bounds_index(self):
        """Test that out of bounds index uses default."""
        base = SphereFunction(dimension=5)
        factors = [1.0, 2.0]

        # Index 5 is out of bounds, should use 1.0
        scaled = ComposedFunction(
            base_function=base,
            input_transforms=[
                IndexedScaleTransform(factors=factors, component_index=5, default_factor=1.0)
            ],
        )

        x = np.ones(5)
        expected = base.evaluate(x)  # No scaling
        actual = scaled.evaluate(x)

        assert np.isclose(actual, expected)


class TestIndexedRotation:
    """Test indexed rotation transformation."""

    def test_single_matrix(self):
        """Test with single rotation matrix (backwards compatible)."""
        base = SphereFunction(dimension=2)
        # 90 degree rotation
        matrix = np.array([[0, -1], [1, 0]], dtype=float)

        rotated = ComposedFunction(
            base_function=base, input_transforms=[IndexedRotateTransform(matrices=matrix)]
        )

        x = np.array([1.0, 0.0])
        # After 90 degree rotation: [0, 1]
        expected = base.evaluate(np.array([0.0, 1.0]))
        actual = rotated.evaluate(x)

        assert np.isclose(actual, expected)

    def test_stacked_matrices(self):
        """Test with stacked matrices and component index."""
        base = SphereFunction(dimension=2)

        # Stack two 2x2 matrices
        # First: identity
        # Second: 90 degree rotation
        matrices = np.array(
            [
                [[1, 0], [0, 1]],  # Identity
                [[0, -1], [1, 0]],  # 90 degree rotation
            ]
        ).reshape(4, 2)  # Stack vertically as in CEC files

        # Test component 1 (90 degree rotation)
        rotated = ComposedFunction(
            base_function=base,
            input_transforms=[
                IndexedRotateTransform(matrices=matrices, component_index=1, matrix_dimension=2)
            ],
        )

        x = np.array([1.0, 0.0])
        # After 90 degree rotation: [0, 1]
        expected = base.evaluate(np.array([0.0, 1.0]))
        actual = rotated.evaluate(x)

        assert np.isclose(actual, expected)

    def test_identity_matrix(self):
        """Test with identity matrix string."""
        base = SphereFunction(dimension=3)
        rotated = ComposedFunction(
            base_function=base, input_transforms=[IndexedRotateTransform(matrices=np.eye(3))]
        )

        x = np.array([1.0, 2.0, 3.0])
        # Identity rotation doesn't change input
        expected = base.evaluate(x)
        actual = rotated.evaluate(x)

        assert np.isclose(actual, expected)

    def test_matrix_from_file(self, tmp_path):
        """Test loading matrix from file."""
        # Create test matrix file
        matrix_file = tmp_path / "test_matrix.txt"
        matrix = np.eye(3) * 2.0  # Scale by 2 (not proper rotation but ok for test)
        np.savetxt(matrix_file, matrix)

        base = SphereFunction(dimension=3)

        loaded = np.loadtxt(matrix_file)
        rotated = ComposedFunction(
            base_function=base, input_transforms=[IndexedRotateTransform(matrices=loaded)]
        )

        x = np.array([1.0, 1.0, 1.0])
        # Matrix scales by 2
        expected = base.evaluate(x * 2.0)
        actual = rotated.evaluate(x)

        assert np.isclose(actual, expected)


class TestIndexedShift:
    """Test indexed shift transformation."""

    def test_single_shift(self):
        """Test with single shift vector (backwards compatible)."""
        base = SphereFunction(dimension=3)
        shift = np.array([1.0, 2.0, 3.0])

        shifted = ComposedFunction(
            base_function=base, input_transforms=[IndexedShiftTransform(shifts=shift)]
        )

        x = np.array([1.0, 2.0, 3.0])
        # After shift: [0, 0, 0]
        expected = base.evaluate(np.zeros(3))
        actual = shifted.evaluate(x)

        assert np.isclose(actual, expected)

    def test_indexed_shifts(self):
        """Test with multiple shift vectors and component index."""
        base = SphereFunction(dimension=2)

        # Multiple shift vectors
        shifts = [np.array([1.0, 1.0]), np.array([2.0, 3.0]), np.array([-1.0, -2.0])]

        # Test component 1 (shift by [2, 3])
        shifted = ComposedFunction(
            base_function=base,
            input_transforms=[IndexedShiftTransform(shifts=shifts, component_index=1)],
        )

        x = np.array([2.0, 3.0])
        # After shift: [0, 0]
        expected = base.evaluate(np.zeros(2))
        actual = shifted.evaluate(x)

        assert np.isclose(actual, expected)
