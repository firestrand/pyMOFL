"""
Integration tests for indexed transformations with factory.

Tests that indexed transformations work correctly through the factory
and configuration system.
"""

import numpy as np

from pyMOFL.factories import FunctionFactory
from pyMOFL.factories.function_factory import DataLoader, FunctionRegistry


class TestIndexedTransformsIntegration:
    """Test indexed transforms through factory."""

    def test_component_with_indexed_scale(self):
        """Test creating component with indexed scale factor."""

        # Create test configuration with indexed scale
        config = {
            "type": "scale",
            "parameters": {
                "factors": [0.5, 1.0, 2.0, 4.0],  # Array of factors
                "component_index": 2,  # Use factor at index 2 (value=2.0)
            },
            "function": {"type": "sphere", "parameters": {"dimension": 5}},
        }

        # Create factory
        loader = DataLoader(base_path="src/pyMOFL/constants")
        registry = FunctionRegistry()
        factory = FunctionFactory(data_loader=loader, registry=registry)

        # Create function
        func = factory.create_function(config)

        # Test evaluation
        x = np.ones(5)
        # With scale factor 2.0, input is scaled by 1/2 = 0.5
        # Sphere([0.5, 0.5, ...]) = 5 * 0.25 = 1.25
        expected = 1.25
        actual = func.evaluate(x)

        assert np.isclose(actual, expected), f"Expected {expected}, got {actual}"

    def test_component_with_indexed_rotation(self, tmp_path):
        """Test creating component with indexed rotation matrix."""

        # Create stacked rotation matrices file
        matrix_file = tmp_path / "stacked_rotations.txt"

        # Stack 3 matrices: identity, 90-degree rotation, 180-degree rotation
        matrices = []
        # Identity (2x2)
        matrices.append([[1, 0], [0, 1]])
        # 90-degree rotation
        matrices.append([[0, -1], [1, 0]])
        # 180-degree rotation
        matrices.append([[-1, 0], [0, -1]])

        stacked = np.vstack(matrices)
        np.savetxt(matrix_file, stacked)

        # Create test configuration
        config = {
            "type": "rotate",
            "parameters": {
                "matrix": str(matrix_file),
                "component_index": 1,  # Use 90-degree rotation
                "matrix_dimension": 2,
            },
            "function": {"type": "sphere", "parameters": {"dimension": 2}},
        }

        # Create factory
        loader = DataLoader(base_path="src/pyMOFL/constants")
        registry = FunctionRegistry()
        factory = FunctionFactory(data_loader=loader, registry=registry)

        # Create function
        func = factory.create_function(config)

        # Test evaluation
        x = np.array([1.0, 0.0])
        # After 90-degree rotation: [0, 1]
        # Sphere([0, 1]) = 0 + 1 = 1.0
        expected = 1.0
        actual = func.evaluate(x)

        assert np.isclose(actual, expected), f"Expected {expected}, got {actual}"

    def test_f20_like_composition(self, tmp_path):
        """Test creating F20-like composition with simplified config format."""

        # Create rotation matrix file (two 2x2 identity matrices stacked)
        matrix_file = tmp_path / "test_rotations.txt"
        matrices = np.vstack([np.eye(2), np.eye(2)])
        np.savetxt(matrix_file, matrices)

        # Create shift vectors file
        shift_file = tmp_path / "test_shifts.txt"
        shifts = np.array([[1.0, 2.0], [3.0, 4.0]])  # Two shift vectors
        np.savetxt(shift_file, shifts)

        config = {
            "type": "composition",
            "parameters": {
                "dimension": 2,
                "dominance_suppression": False,
                "shift_file": str(shift_file),
                "rotation_file": str(matrix_file),
                "lambdas": [0.5, 2.0],
                "sigmas": [1.0, 1.0],
                "biases": [0.0, 0.0],
                "C": 2000.0,
                "global_bias": 10.0,
            },
            "functions": [
                {"type": "sphere"},
                {"type": "sphere"},
            ],
        }

        # Create factory
        loader = DataLoader(base_path="src/pyMOFL/constants")
        registry = FunctionRegistry()
        factory = FunctionFactory(data_loader=loader, registry=registry)

        # Create function
        func = factory.create_function(config)

        # Test at one of the component optima
        x = np.array([1.0, 2.0])  # First component's optimum
        result = func.evaluate(x)

        # At component 0's optimum, its shift makes z=0, sphere=0.
        # Result includes global_bias (10.0) plus weighted component contributions.
        assert np.isfinite(result)
        assert result >= 10.0, f"Result {result} should be >= global_bias (10.0)"
