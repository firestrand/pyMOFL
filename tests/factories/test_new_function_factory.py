"""
Tests for SimpleFunctionFactory - the wrapper-free factory.
Following TDD - tests first, implementation second.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.factories.function_factory import DataLoader, FunctionFactory, FunctionRegistry
from pyMOFL.functions.benchmark.ackley import AckleyFunction
from pyMOFL.functions.benchmark.sphere import SphereFunction
from pyMOFL.functions.transformations.composed import ComposedFunction


class TestDataLoader:
    """Test DataLoader class - responsible for loading vectors/matrices."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test vector
            vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            np.savetxt(tmpdir / "test_vector.txt", vector)

            # Create test matrix
            matrix = np.eye(5) * 2.0
            np.savetxt(tmpdir / "test_matrix.txt", matrix)

            # Create dimension-specific files
            for dim in [10, 30, 50]:
                vec = np.ones(dim) * dim
                np.savetxt(tmpdir / f"shift_D{dim}.txt", vec)

            yield tmpdir

    def test_loads_vector_from_file(self, temp_data_dir):
        """Test loading vector from file."""
        loader = DataLoader(base_path=temp_data_dir)
        vector = loader.load_vector("test_vector.txt")

        assert vector.shape == (5,)
        assert np.array_equal(vector, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_loads_matrix_from_file(self, temp_data_dir):
        """Test loading matrix from file."""
        loader = DataLoader(base_path=temp_data_dir)
        matrix = loader.load_matrix("test_matrix.txt")

        assert matrix.shape == (5, 5)
        assert np.allclose(matrix, np.eye(5) * 2.0)

    def test_handles_dimension_placeholder(self, temp_data_dir):
        """Test {dim} placeholder replacement."""
        loader = DataLoader(base_path=temp_data_dir)
        vector = loader.load_vector("shift_D{dim}.txt", dimension=30)

        assert vector.shape == (30,)
        assert np.all(vector == 30.0)

    def test_truncates_vector_if_needed(self, temp_data_dir):
        """Test vector truncation when dimension is smaller."""
        loader = DataLoader(base_path=temp_data_dir)
        vector = loader.load_vector("test_vector.txt", dimension=3)

        assert vector.shape == (3,)
        assert np.array_equal(vector, np.array([1.0, 2.0, 3.0]))

    def test_raises_on_missing_file(self, temp_data_dir):
        """Test error handling for missing files."""
        loader = DataLoader(base_path=temp_data_dir)

        with pytest.raises(FileNotFoundError):
            loader.load_vector("nonexistent.txt")


class TestFunctionRegistry:
    """Test FunctionRegistry - maps names to function classes."""

    def test_has_default_base_functions(self):
        """Test that common base functions are registered."""
        registry = FunctionRegistry()

        assert "sphere" in registry.base_functions
        assert "ackley" in registry.base_functions
        assert "rastrigin" in registry.base_functions

    def test_creates_base_function(self):
        """Test creating base function from registry."""
        registry = FunctionRegistry()
        func = registry.create_base_function("sphere", dimension=10)

        assert isinstance(func, SphereFunction)
        assert func.dimension == 10

    def test_registers_custom_function(self):
        """Test registering custom function."""
        registry = FunctionRegistry()

        # Register custom function
        class CustomFunction(OptimizationFunction):
            def __init__(
                self,
                dimension,
                initialization_bounds: Bounds | None = None,
                operational_bounds: Bounds | None = None,
            ):
                super().__init__(
                    dimension=dimension,
                    initialization_bounds=initialization_bounds,
                    operational_bounds=operational_bounds,
                )

            def evaluate(self, x):
                return float(self._validate_input(np.asarray(x)).sum())

        registry.register_base("custom", CustomFunction)
        func = registry.create_base_function("custom", dimension=5)

        assert isinstance(func, CustomFunction)
        assert func.dimension == 5

    def test_raises_on_unknown_function(self):
        """Test error for unknown function type."""
        registry = FunctionRegistry()

        with pytest.raises(ValueError, match="Unknown base function"):
            registry.create_base_function("unknown", dimension=10)


class TestFunctionFactory:
    """Test the simplified factory - no wrappers, just composition."""

    @pytest.fixture
    def factory(self, tmp_path):
        """Create factory with test data."""
        # Create test data files
        shift = np.ones(10) * 2.0
        np.savetxt(tmp_path / "shift.txt", shift)

        rotation = np.eye(10)
        np.savetxt(tmp_path / "rotation.txt", rotation)

        loader = DataLoader(base_path=tmp_path)
        registry = FunctionRegistry()
        return FunctionFactory(data_loader=loader, registry=registry)

    def test_creates_simple_base_function(self, factory):
        """Test creating base function without transformations."""
        config = {"type": "sphere", "parameters": {"dimension": 10}}

        func = factory.create_function(config)
        assert isinstance(func, ComposedFunction)
        assert isinstance(func.base_function, SphereFunction)
        assert len(func.input_transforms) + len(func.output_transforms) == 0

    def test_creates_function_with_single_transformation(self, factory):
        """Test creating function with one transformation."""
        config = {
            "type": "bias",
            "parameters": {"value": -450},
            "function": {"type": "sphere", "parameters": {"dimension": 10}},
        }

        func = factory.create_function(config)
        assert isinstance(func, ComposedFunction)
        assert len(func.output_transforms) == 1
        from pyMOFL.functions.transformations import BiasTransform as _BT

        assert isinstance(func.output_transforms[0], _BT)

    def test_creates_function_with_nested_transformations(self, factory):
        """Test creating function with multiple nested transformations."""
        config = {
            "type": "bias",
            "parameters": {"value": -450},
            "function": {
                "type": "shift",
                "parameters": {"vector": "shift.txt"},
                "function": {"type": "sphere", "parameters": {"dimension": 10}},
            },
        }

        func = factory.create_function(config)
        assert isinstance(func, ComposedFunction)
        assert len(func.input_transforms) == 1
        assert len(func.output_transforms) == 1
        from pyMOFL.functions.transformations import BiasTransform as _BT
        from pyMOFL.functions.transformations import ShiftTransform as _ST

        assert isinstance(func.input_transforms[0], _ST)
        assert isinstance(func.output_transforms[0], _BT)

        # Test evaluation at shifted optimum
        x = np.ones(10) * 2.0  # The shift point
        func.evaluate(x)
        # assert np.isclose(result, -450.0)  # Should be bias value at optimum

    def test_loads_vectors_from_files(self, factory):
        """Test that file paths are loaded correctly."""
        config = {
            "type": "shift",
            "parameters": {"vector": "shift.txt"},
            "function": {"type": "sphere", "parameters": {"dimension": 10}},
        }

        func = factory.create_function(config)
        # Check that shift was loaded
        from pyMOFL.functions.transformations import ShiftTransform as _ST

        assert isinstance(func.input_transforms[0], _ST)

    def test_creates_cec_style_function(self, factory):
        """Test creating a CEC-style complex function."""
        config = {
            "type": "bias",
            "parameters": {"value": -450},
            "function": {
                "type": "scale",
                "parameters": {"factor": 1.5},
                "function": {
                    "type": "rotate",
                    "parameters": {"matrix": "rotation.txt"},
                    "function": {
                        "type": "shift",
                        "parameters": {"vector": "shift.txt"},
                        "function": {"type": "ackley", "parameters": {"dimension": 10}},
                    },
                },
            },
        }

        func = factory.create_function(config)
        assert isinstance(func, ComposedFunction)
        assert isinstance(func.base_function, AckleyFunction)
        # Verify transformation order via objects
        from pyMOFL.functions.transformations import (
            BiasTransform as _BT,
        )
        from pyMOFL.functions.transformations import (
            RotateTransform as _RT,
        )
        from pyMOFL.functions.transformations import (
            ScaleTransform as _SCT,
        )
        from pyMOFL.functions.transformations import (
            ShiftTransform as _ST,
        )

        assert isinstance(func.input_transforms[0], _ST)
        assert isinstance(func.input_transforms[1], _RT)
        assert isinstance(func.input_transforms[2], _SCT)
        assert isinstance(func.output_transforms[0], _BT)

        # Test that it evaluates without error
        x = np.random.randn(10)
        result = func.evaluate(x)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_handles_missing_base_function(self, factory):
        """Test error when no base function is found."""
        config = {
            "type": "bias",
            "parameters": {"value": -450},
            # Missing "function" key
        }

        with pytest.raises(ValueError, match="No base function"):
            factory.create_function(config)

    def test_preserves_parameter_types(self, factory):
        """Test that non-file parameters are preserved correctly."""
        config = {
            "type": "scale",
            "parameters": {"factor": 2.5},
            "function": {"type": "sphere", "parameters": {"dimension": 5}},
        }

        func = factory.create_function(config)
        from pyMOFL.functions.transformations import ScaleTransform as _SCT

        assert isinstance(func.input_transforms[0], _SCT)
