"""Test that transformations are applied in the correct order."""

import numpy as np
import pytest

from pyMOFL.factories import FunctionFactory
from pyMOFL.factories.function_factory import DataLoader, FunctionRegistry


class TestTransformationOrder:
    """Test transformation order follows mathematical nesting."""

    @pytest.fixture
    def factory(self):
        """Create factory for tests."""
        loader = DataLoader(base_path="src/pyMOFL/constants")
        registry = FunctionRegistry()
        return FunctionFactory(data_loader=loader, registry=registry)

    def test_simple_nested_shift(self, factory):
        """Test bias(sphere(shift(x)))."""
        config = {
            "type": "bias",
            "parameters": {"value": -450},
            "function": {
                "type": "sphere",
                "parameters": {"dimension": 2},
                "function": {"type": "shift", "parameters": {"vector": [1.0, 2.0]}},
            },
        }

        func = factory.create_function(config)

        # At shift point, sphere should be 0
        result = func.evaluate(np.array([1.0, 2.0]))
        assert abs(result - (-450)) < 1e-10

        # Away from shift point
        result = func.evaluate(np.array([0.0, 0.0]))
        assert abs(result - (-450 + 5)) < 1e-10  # sphere([0,0] - [1,2]) = 5

    def test_nested_rotate_shift(self, factory):
        """Test sphere(rotate(shift(x)))."""
        # 90-degree rotation
        rot90 = [[0, -1], [1, 0]]

        config = {
            "type": "sphere",
            "parameters": {"dimension": 2},
            "function": {
                "type": "rotate",
                "parameters": {"matrix": rot90},
                "function": {"type": "shift", "parameters": {"vector": [3.0, 0.0]}},
            },
        }

        func = factory.create_function(config)

        # At shift point [3, 0]:
        # shift([3,0]) = [0,0]
        # rotate([0,0]) = [0,0]
        # sphere([0,0]) = 0
        result = func.evaluate(np.array([3.0, 0.0]))
        assert abs(result) < 1e-10

        # At [4, 0]:
        # shift([4,0]) = [1,0]
        # rotate([1,0]) = [0,1] (90° rotation)
        # sphere([0,1]) = 1
        result = func.evaluate(np.array([4.0, 0.0]))
        assert abs(result - 1.0) < 1e-10

    def test_complex_nesting(self, factory):
        """Test bias(elliptic(rotate(shift(x))))."""
        config = {
            "type": "bias",
            "parameters": {"value": -100},
            "function": {
                "type": "high_conditioned_elliptic",
                "parameters": {"dimension": 2, "condition": 100},
                "function": {
                    "type": "rotate",
                    "parameters": {"matrix": [[1, 0], [0, 1]]},  # Identity
                    "function": {"type": "shift", "parameters": {"vector": [5.0, -3.0]}},
                },
            },
        }

        func = factory.create_function(config)

        # At shift point, all transforms should yield 0
        result = func.evaluate(np.array([5.0, -3.0]))
        assert abs(result - (-100)) < 1e-10

    def test_transformation_list_order(self, factory):
        """Test that transformation list respects nesting."""
        config = {
            "type": "bias",
            "parameters": {"value": 10},
            "function": {
                "type": "sphere",
                "parameters": {"dimension": 3},
                "function": {
                    "type": "scale",
                    "parameters": {"factor": 2.0},
                    "function": {
                        "type": "rotate",
                        "parameters": {"matrix": "identity"},
                        "function": {"type": "shift", "parameters": {"vector": [1, 2, 3]}},
                    },
                },
            },
        }

        func = factory.create_function(config)

        # Check that input transforms are in correct order
        # For nested: scale(rotate(shift(x)))
        # Application order should be: shift, rotate, scale
        # Map transform objects to canonical names
        from pyMOFL.functions.transformations import RotateTransform, ScaleTransform, ShiftTransform

        name_map = {ShiftTransform: "shift", RotateTransform: "rotate", ScaleTransform: "scale"}
        transform_types = [name_map.get(type(t)) for t in func.input_transforms]
        assert transform_types == ["shift", "rotate", "scale"]

        # Verify execution
        # At shift point [1,2,3]:
        # shift([1,2,3]) = [0,0,0]
        # rotate([0,0,0]) = [0,0,0]
        # scale([0,0,0]) = [0,0,0]
        # sphere([0,0,0]) = 0
        # bias(0) = 10
        result = func.evaluate(np.array([1.0, 2.0, 3.0]))
        assert abs(result - 10) < 1e-10
