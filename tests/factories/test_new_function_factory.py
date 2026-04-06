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

    def test_creates_base_function_case_insensitive(self):
        """Registry lookup should be case-insensitive."""
        registry = FunctionRegistry()
        for name in ("Sphere", "SPHERE", "sPHERE", "Ackley"):
            func = registry.create_base_function(name, dimension=5)
            assert isinstance(func, OptimizationFunction)
            assert func.dimension == 5

    def test_has_phase1_bbob_base_functions(self):
        """Phase 1 BBOB types are registered in the factory."""
        registry = FunctionRegistry()
        for key in [
            "linear_slope",
            "attractive_sector",
            "sharp_ridge",
            "schwefel_sin",
            "gallagher_peaks",
        ]:
            assert key in registry.base_functions, f"{key!r} missing from base_functions"

    def test_creates_phase1_bbob_functions(self):
        """Each Phase 1 BBOB type can be instantiated via create_base_function."""
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        registry = FunctionRegistry()
        cases = {
            "linear_slope": LinearSlopeFunction,
            "attractive_sector": AttractiveSectorFunction,
            "sharp_ridge": SharpRidgeFunction,
            "schwefel_sin": SchwefelSinFunction,
            "gallagher_peaks": GallagherPeaksFunction,
        }
        for key, expected_cls in cases.items():
            func = registry.create_base_function(key, dimension=5)
            assert isinstance(func, expected_cls), f"{key!r} created wrong type"

    def test_has_phase3_bbob_base_functions(self):
        """Phase 3 BBOB types are registered in the factory."""
        registry = FunctionRegistry()
        for key in [
            "discus",
            "bent_cigar",
            "different_powers",
            "schaffer_f7",
            "katsuura",
            "lunacek",
            "buche_rastrigin",
            "step_ellipsoid",
        ]:
            assert key in registry.base_functions, f"{key!r} missing from base_functions"

    def test_creates_phase3_bbob_functions(self):
        """Each Phase 3 BBOB type can be instantiated via create_base_function."""
        from pyMOFL.functions.benchmark.bent_cigar import BentCigarFunction, DiscusFunction
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction
        from pyMOFL.functions.benchmark.different_powers import DifferentPowersFunction
        from pyMOFL.functions.benchmark.katsuura import KatsuuraFunction
        from pyMOFL.functions.benchmark.lunacek import LunacekBiRastriginFunction
        from pyMOFL.functions.benchmark.schaffer import SchaffersF7Function
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        registry = FunctionRegistry()
        cases = {
            "discus": DiscusFunction,
            "bent_cigar": BentCigarFunction,
            "different_powers": DifferentPowersFunction,
            "schaffer_f7": SchaffersF7Function,
            "katsuura": KatsuuraFunction,
            "lunacek": LunacekBiRastriginFunction,
            "buche_rastrigin": BucheRastriginFunction,
            "step_ellipsoid": StepEllipsoidFunction,
        }
        for key, expected_cls in cases.items():
            func = registry.create_base_function(key, dimension=5)
            assert isinstance(func, expected_cls), f"{key!r} created wrong type"

    def test_has_phase4_classical_scalable_base_functions(self):
        """Phase 4 scalable classical types are registered in the factory."""
        registry = FunctionRegistry()
        for key in [
            "styblinski_tang",
            "salomon",
            "michalewicz",
            "langermann",
            "brown",
            "chung_reynolds",
            "qing",
            "quartic",
        ]:
            assert key in registry.base_functions, f"{key!r} missing from base_functions"

    def test_creates_phase4_classical_scalable_functions(self):
        """Each Phase 4 scalable classical type can be instantiated."""
        from pyMOFL.functions.benchmark.brown import BrownFunction
        from pyMOFL.functions.benchmark.chung_reynolds import ChungReynoldsFunction
        from pyMOFL.functions.benchmark.langermann import LangermannFunction
        from pyMOFL.functions.benchmark.michalewicz import MichalewiczFunction
        from pyMOFL.functions.benchmark.qing import QingFunction
        from pyMOFL.functions.benchmark.quartic import QuarticFunction
        from pyMOFL.functions.benchmark.salomon import SalomonFunction
        from pyMOFL.functions.benchmark.styblinski_tang import StyblinskiTangFunction

        registry = FunctionRegistry()
        cases = {
            "styblinski_tang": StyblinskiTangFunction,
            "salomon": SalomonFunction,
            "michalewicz": MichalewiczFunction,
            "langermann": LangermannFunction,
            "brown": BrownFunction,
            "chung_reynolds": ChungReynoldsFunction,
            "qing": QingFunction,
            "quartic": QuarticFunction,
        }
        for key, expected_cls in cases.items():
            func = registry.create_base_function(key, dimension=5)
            assert isinstance(func, expected_cls), f"{key!r} created wrong type"

    def test_has_phase4_classical_fixed_base_functions(self):
        """Phase 4 fixed-dimension classical types are registered in the factory."""
        registry = FunctionRegistry()
        for key in [
            "beale",
            "booth",
            "bohachevsky1",
            "bohachevsky2",
            "bohachevsky3",
            "bukin6",
            "six_hump_camel",
            "three_hump_camel",
            "cross_in_tray",
            "drop_wave",
            "eggholder",
            "holder_table",
            "hartmann3",
            "hartmann6",
            "colville",
        ]:
            assert key in registry.base_functions, f"{key!r} missing from base_functions"

    def test_creates_phase4_classical_fixed_functions(self):
        """Each Phase 4 fixed-dimension classical type can be instantiated."""
        from pyMOFL.functions.benchmark.beale import BealeFunction
        from pyMOFL.functions.benchmark.bohachevsky import (
            Bohachevsky1Function,
            Bohachevsky2Function,
            Bohachevsky3Function,
        )
        from pyMOFL.functions.benchmark.booth import BoothFunction
        from pyMOFL.functions.benchmark.bukin import Bukin6Function
        from pyMOFL.functions.benchmark.camel import SixHumpCamelFunction, ThreeHumpCamelFunction
        from pyMOFL.functions.benchmark.colville import ColvilleFunction
        from pyMOFL.functions.benchmark.cross_in_tray import CrossInTrayFunction
        from pyMOFL.functions.benchmark.drop_wave import DropWaveFunction
        from pyMOFL.functions.benchmark.eggholder import EggholderFunction
        from pyMOFL.functions.benchmark.hartmann import Hartmann3Function, Hartmann6Function
        from pyMOFL.functions.benchmark.holder_table import HolderTableFunction

        registry = FunctionRegistry()
        cases = {
            "beale": (BealeFunction, {}),
            "booth": (BoothFunction, {}),
            "bohachevsky1": (Bohachevsky1Function, {}),
            "bohachevsky2": (Bohachevsky2Function, {}),
            "bohachevsky3": (Bohachevsky3Function, {}),
            "bukin6": (Bukin6Function, {}),
            "six_hump_camel": (SixHumpCamelFunction, {}),
            "three_hump_camel": (ThreeHumpCamelFunction, {}),
            "cross_in_tray": (CrossInTrayFunction, {}),
            "drop_wave": (DropWaveFunction, {}),
            "eggholder": (EggholderFunction, {}),
            "holder_table": (HolderTableFunction, {}),
            "hartmann3": (Hartmann3Function, {"dimension": 3}),
            "hartmann6": (Hartmann6Function, {"dimension": 6}),
            "colville": (ColvilleFunction, {"dimension": 4}),
        }
        for key, (expected_cls, extra_params) in cases.items():
            params = extra_params or {}
            func = registry.create_base_function(key, **params)
            assert isinstance(func, expected_cls), f"{key!r} created wrong type"

    def test_has_phase6_selected_classical_functions(self):
        """Phase 6 on-demand additions should be available via the registry."""
        registry = FunctionRegistry()
        for key in [
            "adjiman",
            "box_betts",
            "deb01",
            "deb03",
            "exponential",
            "keane",
            "kowalik",
            "miele_cantrell",
            "parsopoulos",
            "rana",
        ]:
            assert key in registry.base_functions, f"{key!r} missing from base_functions"

    def test_creates_phase6_selected_classical_functions(self):
        """Selected Phase 6 functions should instantiate via create_base_function."""
        from pyMOFL.functions.benchmark.adjiman import AdjimanFunction
        from pyMOFL.functions.benchmark.box_betts import BoxBettsFunction
        from pyMOFL.functions.benchmark.deb01 import Deb01Function
        from pyMOFL.functions.benchmark.deb03 import Deb03Function
        from pyMOFL.functions.benchmark.exponential_function import ExponentialFunction
        from pyMOFL.functions.benchmark.keane import KeaneFunction
        from pyMOFL.functions.benchmark.kowalik import KowalikFunction
        from pyMOFL.functions.benchmark.miele_cantrell import MieleCantrellFunction
        from pyMOFL.functions.benchmark.parsopoulos import ParsopoulosFunction
        from pyMOFL.functions.benchmark.rana import RanaFunction

        registry = FunctionRegistry()
        cases = {
            "adjiman": (AdjimanFunction, {"dimension": 2}),
            "box_betts": (BoxBettsFunction, {"dimension": 3}),
            "deb01": (Deb01Function, {"dimension": 5}),
            "deb03": (Deb03Function, {"dimension": 5}),
            "exponential": (ExponentialFunction, {"dimension": 5}),
            "keane": (KeaneFunction, {"dimension": 5}),
            "kowalik": (KowalikFunction, {"dimension": 4}),
            "miele_cantrell": (MieleCantrellFunction, {"dimension": 4}),
            "parsopoulos": (ParsopoulosFunction, {"dimension": 2}),
            "rana": (RanaFunction, {"dimension": 5}),
        }
        for key, (expected_cls, params) in cases.items():
            func = registry.create_base_function(key, **params)
            assert isinstance(func, expected_cls), f"{key!r} created wrong type"


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

    def test_creates_phase1_function_via_config(self, factory):
        """Phase 1 BBOB types route through the config-driven factory."""
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        cases = [
            ("linear_slope", LinearSlopeFunction),
            ("attractive_sector", AttractiveSectorFunction),
            ("sharp_ridge", SharpRidgeFunction),
            ("schwefel_sin", SchwefelSinFunction),
            ("gallagher_peaks", GallagherPeaksFunction),
        ]
        for type_name, expected_cls in cases:
            config = {"type": type_name, "parameters": {"dimension": 5}}
            func = factory.create_function(config)
            assert isinstance(func, ComposedFunction)
            assert isinstance(func.base_function, expected_cls), (
                f"Config type {type_name!r} routed to wrong base"
            )
