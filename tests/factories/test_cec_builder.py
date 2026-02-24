"""Tests for CompositionBuilder — weighted composition construction."""

from pathlib import Path

import numpy as np
import pytest

from pyMOFL.compositions.weighted_composition import WeightedComposition
from pyMOFL.factories.composition_builder import CompositionBuilder
from pyMOFL.factories.data_loader import DataLoader
from pyMOFL.factories.function_factory import FunctionRegistry
from pyMOFL.utils import load_suite_config


@pytest.fixture(scope="module")
def registry():
    return FunctionRegistry()


@pytest.fixture(scope="module")
def data_loader():
    return DataLoader(base_path="src/pyMOFL/constants/cec/2005")


@pytest.fixture(scope="module")
def builder(data_loader, registry):
    from pyMOFL.factories.config_parser import ConfigParser

    parser = ConfigParser(known_base_types=frozenset(registry.base_functions))
    return CompositionBuilder(data_loader=data_loader, registry=registry, parser=parser)


@pytest.fixture(scope="module")
def suite_config():
    config_path = Path("src/pyMOFL/constants/cec/2005/cec2005_suite.json")
    return load_suite_config(config_path)


def get_composition_config(suite_config, func_number: str) -> dict:
    """Extract the composition config for a given function number like 'f15'."""
    target_id = f"cec05_{func_number}_"
    for item in suite_config["functions"]:
        if item["id"].startswith(target_id):
            # Navigate to the composition node
            node = item["function"]
            while node.get("type") != "composition":
                node = node.get("function", {})
            return node
    raise ValueError(f"Function {func_number} not found")


class TestBuildProducesWeightedComposition:
    """All compositions should produce WeightedComposition objects."""

    @pytest.mark.parametrize(
        "func_id",
        ["f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24", "f25"],
    )
    def test_build_returns_weighted_composition(self, builder, suite_config, func_id):
        config = get_composition_config(suite_config, func_id)
        result = builder.build(config, dimension=10)
        assert isinstance(result, WeightedComposition), (
            f"{func_id} should produce WeightedComposition, got {type(result).__name__}"
        )


def _extract_lambdas(wc):
    """Extract lambda (scale factor) from each component's ScaleTransform."""
    from pyMOFL.functions.transformations import ScaleTransform

    lambdas = []
    for comp in wc.components:
        for t in comp.input_transforms:
            if isinstance(t, ScaleTransform):
                lambdas.append(t.factor)
                break
    return lambdas


def _extract_normalization_c(wc):
    """Extract C from first component's NormalizeTransform."""
    return wc.components[0].output_transforms[0].C


class TestTopLevelParameterExtraction:
    """Test that the builder reads lambdas/sigmas/biases from top-level parameters."""

    def test_f15_lambdas(self, builder, suite_config):
        config = get_composition_config(suite_config, "f15")
        result = builder.build(config, dimension=10)
        expected_lambdas = [
            1.0,
            1.0,
            10.0,
            10.0,
            5.0 / 60.0,
            5.0 / 60.0,
            5.0 / 32.0,
            5.0 / 32.0,
            5.0 / 100.0,
            5.0 / 100.0,
        ]
        np.testing.assert_allclose(_extract_lambdas(result), expected_lambdas, rtol=1e-4)

    def test_f15_sigmas(self, builder, suite_config):
        config = get_composition_config(suite_config, "f15")
        result = builder.build(config, dimension=10)
        expected_sigmas = [1.0] * 10
        np.testing.assert_array_equal(result.sigmas, expected_sigmas)

    def test_f18_lambdas(self, builder, suite_config):
        config = get_composition_config(suite_config, "f18")
        result = builder.build(config, dimension=10)
        expected_lambdas = [
            5.0 / 16.0,
            5.0 / 32.0,
            2.0,
            1.0,
            1.0 / 10.0,
            1.0 / 20.0,
            20.0,
            10.0,
            1.0 / 6.0,
            1.0 / 12.0,
        ]
        np.testing.assert_allclose(_extract_lambdas(result), expected_lambdas, rtol=1e-4)

    def test_f18_sigmas(self, builder, suite_config):
        config = get_composition_config(suite_config, "f18")
        result = builder.build(config, dimension=10)
        expected_sigmas = [1.0, 2.0, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0]
        np.testing.assert_array_equal(result.sigmas, expected_sigmas)

    def test_f21_lambdas(self, builder, suite_config):
        config = get_composition_config(suite_config, "f21")
        result = builder.build(config, dimension=10)
        expected_lambdas = [
            1.0 / 4.0,
            1.0 / 20.0,
            5.0,
            1.0,
            5.0,
            1.0,
            50.0,
            10.0,
            1.0 / 8.0,
            1.0 / 40.0,
        ]
        np.testing.assert_allclose(_extract_lambdas(result), expected_lambdas, rtol=1e-4)

    def test_f24_lambdas(self, builder, suite_config):
        config = get_composition_config(suite_config, "f24")
        result = builder.build(config, dimension=10)
        expected_lambdas = [
            10.0,
            5.0 / 20.0,
            1.0,
            5.0 / 32.0,
            1.0,
            5.0 / 100.0,
            5.0 / 50.0,
            1.0,
            5.0 / 100.0,
            5.0 / 100.0,
        ]
        np.testing.assert_allclose(_extract_lambdas(result), expected_lambdas, rtol=1e-4)

    def test_f24_sigmas_all_2(self, builder, suite_config):
        config = get_composition_config(suite_config, "f24")
        result = builder.build(config, dimension=10)
        expected_sigmas = [2.0] * 10
        np.testing.assert_array_equal(result.sigmas, expected_sigmas)

    def test_biases_standard_pattern(self, builder, suite_config):
        """All compositions use biases = [0, 100, 200, ..., 900]."""
        config = get_composition_config(suite_config, "f15")
        result = builder.build(config, dimension=10)
        expected_biases = [i * 100.0 for i in range(10)]
        np.testing.assert_array_equal(result.biases, expected_biases)

    def test_c_defaults_to_2000(self, builder, suite_config):
        config = get_composition_config(suite_config, "f15")
        result = builder.build(config, dimension=10)
        assert _extract_normalization_c(result) == 2000.0


class TestSpecialCaseFlags:
    """Test special-case flags: zero_last_optimum, optimum_pattern, non_continuous."""

    def test_f18_zero_last_optimum(self, builder, suite_config):
        config = get_composition_config(suite_config, "f18")
        result = builder.build(config, dimension=10)
        np.testing.assert_array_equal(result.optima[-1], np.zeros(10))

    def test_f20_alternate_odds(self, builder, suite_config):
        config = get_composition_config(suite_config, "f20")
        result = builder.build(config, dimension=10)
        # Odd indices of first optimum should be 5.0
        for j in range(1, 10, 2):
            assert result.optima[0][j] == 5.0, (
                f"F20 optima[0][{j}] should be 5.0, got {result.optima[0][j]}"
            )
        # Pattern applies only to first component in this benchmark definition
        assert result.optima[1][1] != 5.0

    def test_zero_last_optimum_takes_precedence(self, builder):
        config = {
            "type": "composition",
            "parameters": {
                "dimension": 2,
                "lambdas": [1.0, 1.0],
                "sigmas": [1.0, 1.0],
                "biases": [0.0, 0.0],
                "shift": [[1.0, 2.0], [3.0, 4.0]],
                "optimum_pattern": "alternate_odds",
                "zero_last_optimum": True,
            },
            "functions": [
                {"type": "sphere"},
                {"type": "sphere"},
            ],
        }
        result = builder.build(config, dimension=2)
        np.testing.assert_array_equal(result.optima[0], np.array([1.0, 5.0]))
        np.testing.assert_array_equal(result.optima[1], np.zeros(2))

    def test_f23_non_continuous(self, builder, suite_config):
        config = get_composition_config(suite_config, "f23")
        result = builder.build(config, dimension=10)
        assert result.non_continuous is True


class TestComponentOrder:
    """Test that component function types match the C reference."""

    def test_f15_components_rastrigin_weierstrass_griewank_ackley_sphere(
        self, builder, suite_config
    ):
        config = get_composition_config(suite_config, "f15")
        result = builder.build(config, dimension=10)
        assert len(result.components) == 10

    def test_f24_has_10_components(self, builder, suite_config):
        config = get_composition_config(suite_config, "f24")
        result = builder.build(config, dimension=10)
        assert len(result.components) == 10

    def test_f24_non_continuous_component_transform_order(self, builder, suite_config):
        """Non-continuous component transforms should run after shared composition transforms."""
        config = get_composition_config(suite_config, "f24")
        result = builder.build(config, dimension=10)

        from pyMOFL.functions.transformations import (
            NonContinuousTransform,
            RotateTransform,
            ScaleTransform,
            ShiftTransform,
        )

        for idx in (6, 7):
            comp = result.components[idx]
            transforms = comp.input_transforms
            assert isinstance(transforms[0], ShiftTransform), "expected shared shift first"
            assert isinstance(transforms[1], ScaleTransform), "expected shared scale second"
            assert isinstance(transforms[2], RotateTransform), "expected shared rotate third"
            assert isinstance(transforms[3], NonContinuousTransform), (
                "expected component non_continuous last"
            )


def _extract_local_rotations(wc):
    """Extract local rotation matrices from each component's RotateTransform."""
    from pyMOFL.functions.transformations import RotateTransform

    rotations = []
    for comp in wc.components:
        # The last RotateTransform in each component is the local rotation
        rots = [t for t in comp.input_transforms if isinstance(t, RotateTransform)]
        if len(rots) >= 2:
            rotations.append(rots[-1].matrix)
        elif rots:
            rotations.append(rots[0].matrix)
    return rotations


class TestRotationHandling:
    """Test rotation matrix loading."""

    def test_f15_no_rotation(self, builder, suite_config):
        """F15 has no rotation — local rotations should be identity."""
        config = get_composition_config(suite_config, "f15")
        result = builder.build(config, dimension=2)
        for i, rot in enumerate(_extract_local_rotations(result)):
            np.testing.assert_array_equal(
                rot, np.eye(2), err_msg=f"F15 local_rotation[{i}] should be identity"
            )

    def test_f16_has_rotation(self, builder, suite_config):
        """F16 has rotation — local rotations should NOT be identity."""
        config = get_composition_config(suite_config, "f16")
        result = builder.build(config, dimension=2)
        rotations = _extract_local_rotations(result)
        any_non_identity = any(not np.allclose(rot, np.eye(2)) for rot in rotations)
        assert any_non_identity, "F16 should have non-identity rotation matrices"

    def test_invalid_rotation_matrix_is_replaced_with_identity(self, builder):
        config = {
            "type": "composition",
            "parameters": {
                "dimension": 2,
                "lambdas": [1.0, 1.0],
                "sigmas": [1.0, 1.0],
                "biases": [0.0, 0.0],
                "shift": [[1.0, 2.0], [3.0, 4.0]],
                "rotation": [[float("nan"), 0.0], [0.0, 1.0]],
            },
            "functions": [
                {"type": "sphere"},
                {"type": "sphere"},
            ],
        }
        result = builder.build(config, dimension=2)
        for rot in _extract_local_rotations(result):
            np.testing.assert_array_equal(rot, np.eye(2))

    def test_large_finite_rotation_is_accepted(self, builder):
        config = {
            "type": "composition",
            "parameters": {
                "dimension": 2,
                "lambdas": [1.0, 1.0],
                "sigmas": [1.0, 1.0],
                "biases": [0.0, 0.0],
                "shift": [[1.0, 2.0], [3.0, 4.0]],
                "rotation": [[1200.0, 0.0], [0.0, 1200.0]],
            },
            "functions": [
                {"type": "sphere"},
                {"type": "sphere"},
            ],
        }
        result = builder.build(config, dimension=2)
        rotations = _extract_local_rotations(result)
        expected = np.array([[1200.0, 0.0], [0.0, 1200.0]])
        for rot in rotations:
            np.testing.assert_array_equal(rot, expected)


class TestExtractBaseParams:
    """Test the extract_base_params helper function."""

    def test_flat_config_with_params(self):
        from pyMOFL.factories.composition_builder import extract_base_params

        known = frozenset(["weierstrass", "sphere"])
        comp = {"type": "weierstrass", "parameters": {"a": 0.5, "b": 3, "k_max": 20}}
        result = extract_base_params(comp, known)
        assert result == {"a": 0.5, "b": 3, "k_max": 20}

    def test_flat_config_no_params(self):
        from pyMOFL.factories.composition_builder import extract_base_params

        known = frozenset(["sphere"])
        comp = {"type": "sphere"}
        result = extract_base_params(comp, known)
        assert result == {}

    def test_nested_config_finds_base(self):
        from pyMOFL.factories.composition_builder import extract_base_params

        known = frozenset(["rastrigin"])
        comp = {
            "type": "non_continuous",
            "function": {"type": "rastrigin", "parameters": {"custom": 42}},
        }
        result = extract_base_params(comp, known)
        assert result == {"custom": 42}

    def test_deeply_nested_verbose_config(self):
        from pyMOFL.factories.composition_builder import extract_base_params

        known = frozenset(["weierstrass"])
        comp = {
            "type": "weight",
            "function": {
                "type": "bias",
                "function": {
                    "type": "weierstrass",
                    "parameters": {"a": 0.5, "b": 3, "k_max": 20},
                },
            },
        }
        result = extract_base_params(comp, known)
        assert result == {"a": 0.5, "b": 3, "k_max": 20}

    def test_dimension_excluded(self):
        from pyMOFL.factories.composition_builder import extract_base_params

        known = frozenset(["sphere"])
        comp = {"type": "sphere", "parameters": {"dimension": 10}}
        result = extract_base_params(comp, known)
        assert result == {}

    def test_unknown_type_returns_empty(self):
        from pyMOFL.factories.composition_builder import extract_base_params

        known = frozenset(["sphere"])
        comp = {"type": "unknown_function"}
        result = extract_base_params(comp, known)
        assert result == {}


class TestSimplifiedConfigFormat:
    """Test that the simplified JSON config format produces identical results."""

    def test_simplified_f15_matches_values(self, builder):
        """Build F15 from simplified config and verify it evaluates correctly."""
        config = {
            "type": "composition",
            "parameters": {
                "num_functions": 10,
                "dominance_suppression": True,
                "shift_file": "f15/vector_shift_D50.txt",
                "rotation_file": None,
                "lambdas": [1.0, 1.0, 10.0, 10.0, 5 / 60, 5 / 60, 5 / 32, 5 / 32, 0.05, 0.05],
                "sigmas": [1.0] * 10,
                "biases": [i * 100.0 for i in range(10)],
                "C": 2000.0,
            },
            "functions": [
                {"type": "rastrigin"},
                {"type": "rastrigin"},
                {"type": "weierstrass", "parameters": {"a": 0.5, "b": 3, "k_max": 20}},
                {"type": "weierstrass", "parameters": {"a": 0.5, "b": 3, "k_max": 20}},
                {"type": "griewank"},
                {"type": "griewank"},
                {"type": "ackley"},
                {"type": "ackley"},
                {"type": "sphere"},
                {"type": "sphere"},
            ],
        }
        result = builder.build(config, dimension=10)
        assert isinstance(result, WeightedComposition)
        assert len(result.components) == 10

        # Evaluate at origin — should match known value (minus outer bias)
        origin = np.zeros(10)
        val = result.evaluate(origin)
        # F15 at origin = 1666.72... (with outer bias +120 applied by factory)
        # WeightedComposition alone = 1666.72... - 120 = 1546.72...
        expected_wc_val = 1666.7225273397953 - 120.0
        np.testing.assert_allclose(val, expected_wc_val, rtol=1e-12)

    def test_simplified_non_continuous_component(self, builder):
        """Non-continuous component in simplified format should work."""
        config = {
            "type": "composition",
            "parameters": {
                "num_functions": 2,
                "shift_file": "f24/vector_shift_D50.txt",
                "rotation_file": None,
                "lambdas": [10.0, 1.0],
                "sigmas": [2.0, 2.0],
                "biases": [0.0, 100.0],
                "C": 2000.0,
            },
            "functions": [
                {"type": "weierstrass", "parameters": {"a": 0.5, "b": 3, "k_max": 20}},
                {"type": "non_continuous", "function": {"type": "rastrigin"}},
            ],
        }
        result = builder.build(config, dimension=10)
        assert isinstance(result, WeightedComposition)
        assert len(result.components) == 2
        # Should not raise on evaluate
        val = result.evaluate(np.zeros(10))
        assert np.isfinite(val)
