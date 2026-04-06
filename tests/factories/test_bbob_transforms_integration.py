"""
End-to-end integration tests for BBOB transforms through the full factory pipeline.

Verifies that JSON config → ConfigParser → TransformBuilder → ComposedFunction → evaluate
works correctly for oscillation (T_osz), asymmetric (T_asy), and boundary_penalty (f_pen).
"""

import numpy as np
import pytest

from pyMOFL.factories.function_factory import FunctionFactory
from pyMOFL.functions.transformations.asymmetric import AsymmetricTransform
from pyMOFL.functions.transformations.boundary_penalty import BoundaryPenaltyTransform
from pyMOFL.functions.transformations.composed import ComposedFunction
from pyMOFL.functions.transformations.oscillation import OscillationTransform


@pytest.fixture
def factory():
    return FunctionFactory()


class TestOscillationThroughFactory:
    """Test oscillation transform through the full factory pipeline."""

    def test_oscillation_through_factory(self, factory):
        config = {
            "type": "oscillation",
            "function": {"type": "sphere", "parameters": {"dimension": 5}},
        }

        func = factory.create_function(config)

        assert isinstance(func, ComposedFunction)
        assert len(func.input_transforms) == 1
        assert isinstance(func.input_transforms[0], OscillationTransform)
        assert len(func.output_transforms) == 0
        assert len(func.penalty_transforms) == 0

        # Evaluate at origin — oscillation of zeros is zeros, sphere(zeros)=0
        origin = np.zeros(5)
        assert func(origin) == pytest.approx(0.0)

        # Evaluate at a random point — should produce a finite float
        rng = np.random.RandomState(42)
        x = rng.randn(5)
        result = func(x)
        assert np.isfinite(result)
        assert isinstance(result, float)


class TestAsymmetricThroughFactory:
    """Test asymmetric transform through the full factory pipeline."""

    def test_asymmetric_through_factory(self, factory):
        config = {
            "type": "asymmetric",
            "parameters": {"beta": 0.5},
            "function": {"type": "sphere", "parameters": {"dimension": 5}},
        }

        func = factory.create_function(config)

        assert isinstance(func, ComposedFunction)
        assert len(func.input_transforms) == 1
        assert isinstance(func.input_transforms[0], AsymmetricTransform)
        assert func.input_transforms[0].beta == pytest.approx(0.5)
        assert len(func.output_transforms) == 0
        assert len(func.penalty_transforms) == 0

        # Evaluate at origin — asymmetric of zeros is zeros, sphere(zeros)=0
        origin = np.zeros(5)
        assert func(origin) == pytest.approx(0.0)

        # Evaluate at a positive point — asymmetric changes positive values
        # Use 2.0 instead of 1.0 since 1^anything = 1 (no visible effect)
        x = np.full(5, 2.0)
        result = func(x)
        assert np.isfinite(result)
        assert isinstance(result, float)
        # sphere(2,2,2,2,2)=20.0, but asymmetric raises values > 1 to power > 1
        assert result > np.sum(x**2)


class TestBoundaryPenaltyThroughFactory:
    """Test boundary_penalty transform through the full factory pipeline."""

    def test_boundary_penalty_through_factory(self, factory):
        config = {
            "type": "boundary_penalty",
            "parameters": {"bound": 5.0},
            "function": {"type": "sphere", "parameters": {"dimension": 5}},
        }

        func = factory.create_function(config)

        assert isinstance(func, ComposedFunction)
        assert len(func.input_transforms) == 0
        assert len(func.output_transforms) == 0
        assert len(func.penalty_transforms) == 1
        assert isinstance(func.penalty_transforms[0], BoundaryPenaltyTransform)

    def test_no_penalty_within_bounds(self, factory):
        config = {
            "type": "boundary_penalty",
            "parameters": {"bound": 5.0},
            "function": {"type": "sphere", "parameters": {"dimension": 5}},
        }

        func = factory.create_function(config)

        # Point within bounds — penalty is 0, result equals sphere
        x = np.array([1.0, 2.0, 3.0, 4.0, 0.0])
        sphere_val = float(np.sum(x**2))
        assert func(x) == pytest.approx(sphere_val)

    def test_penalty_added_outside_bounds(self, factory):
        config = {
            "type": "boundary_penalty",
            "parameters": {"bound": 5.0},
            "function": {"type": "sphere", "parameters": {"dimension": 5}},
        }

        func = factory.create_function(config)

        # Point outside bounds — penalty should be added
        x = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        sphere_val = float(np.sum(x**2))  # 100.0
        penalty_val = (10.0 - 5.0) ** 2  # 25.0
        expected = sphere_val + penalty_val
        assert func(x) == pytest.approx(expected)


class TestAllThreeBBOBTransformsComposed:
    """Test all three BBOB transforms composed together."""

    def test_all_three_bbob_transforms_composed(self, factory):
        # Config: boundary_penalty(oscillation(asymmetric(sphere)))
        # Nesting = composition order: asymmetric applied first, then oscillation
        config = {
            "type": "boundary_penalty",
            "parameters": {"bound": 5.0},
            "function": {
                "type": "oscillation",
                "function": {
                    "type": "asymmetric",
                    "parameters": {"beta": 0.5},
                    "function": {"type": "sphere", "parameters": {"dimension": 5}},
                },
            },
        }

        func = factory.create_function(config)

        assert isinstance(func, ComposedFunction)
        # Input transforms: asymmetric first, then oscillation (application order)
        assert len(func.input_transforms) == 2
        assert isinstance(func.input_transforms[0], AsymmetricTransform)
        assert isinstance(func.input_transforms[1], OscillationTransform)
        # No output transforms
        assert len(func.output_transforms) == 0
        # One penalty transform
        assert len(func.penalty_transforms) == 1
        assert isinstance(func.penalty_transforms[0], BoundaryPenaltyTransform)

        # Evaluate at origin — all transforms preserve zero
        origin = np.zeros(5)
        assert func(origin) == pytest.approx(0.0)

        # Evaluate at an out-of-bounds point
        x = np.array([10.0, 0.0, 0.0, 0.0, 0.0])
        result = func(x)
        assert np.isfinite(result)
        # Must be > sphere(origin) since penalty is added
        assert result > 0.0


class TestBBOBWithCECStyleTransforms:
    """Test mixing BBOB + CEC-style transforms in a single pipeline."""

    def test_bbob_transforms_with_cec_style_transforms(self, factory):
        # Config: bias(boundary_penalty(oscillation(asymmetric(shift(sphere)))))
        config = {
            "type": "bias",
            "parameters": {"value": -450.0},
            "function": {
                "type": "boundary_penalty",
                "parameters": {"bound": 5.0},
                "function": {
                    "type": "oscillation",
                    "function": {
                        "type": "asymmetric",
                        "parameters": {"beta": 0.2},
                        "function": {
                            "type": "shift",
                            "parameters": {"vector": [1.0, 2.0, 3.0]},
                            "function": {
                                "type": "sphere",
                                "parameters": {"dimension": 3},
                            },
                        },
                    },
                },
            },
        }

        func = factory.create_function(config)

        assert isinstance(func, ComposedFunction)

        # Input transforms: shift, asymmetric, oscillation (application order)
        from pyMOFL.functions.transformations.shift import ShiftTransform

        assert len(func.input_transforms) == 3
        assert isinstance(func.input_transforms[0], ShiftTransform)
        assert isinstance(func.input_transforms[1], AsymmetricTransform)
        assert isinstance(func.input_transforms[2], OscillationTransform)

        # Output transforms: bias
        from pyMOFL.functions.transformations.bias import BiasTransform

        assert len(func.output_transforms) == 1
        assert isinstance(func.output_transforms[0], BiasTransform)

        # Penalty transforms: boundary_penalty
        assert len(func.penalty_transforms) == 1
        assert isinstance(func.penalty_transforms[0], BoundaryPenaltyTransform)

        # Evaluate at a point — should produce a finite float
        x = np.array([1.0, 2.0, 3.0])
        result = func(x)
        assert np.isfinite(result)
        assert isinstance(result, float)

        # At the shift point, shifted input is zero → asymmetric(0)=0 → osc(0)=0
        # → sphere(0)=0 → bias(-450) → -450, no penalty (within bounds)
        assert result == pytest.approx(-450.0)


class TestPenaltyUsesRawInput:
    """Verify penalty is computed on raw x, not on transformed x."""

    def test_penalty_uses_raw_input_in_composed_chain(self, factory):
        # Use a point where oscillation would change the penalty result.
        # oscillation(6.0) != 6.0, so if penalty were computed on transformed x,
        # the result would differ.
        config = {
            "type": "boundary_penalty",
            "parameters": {"bound": 5.0},
            "function": {
                "type": "oscillation",
                "function": {"type": "sphere", "parameters": {"dimension": 1}},
            },
        }

        func = factory.create_function(config)

        # x = [6.0] is outside bound=5.0
        x = np.array([6.0])

        # Penalty should be on raw x: (|6| - 5)^2 = 1.0
        expected_penalty = (6.0 - 5.0) ** 2  # 1.0

        # oscillation(6.0) → some value, sphere of that → base_val
        osc = OscillationTransform()
        osc_x = osc(x)
        base_val = float(osc_x[0] ** 2)  # sphere of oscillated value

        expected_total = base_val + expected_penalty
        assert func(x) == pytest.approx(expected_total)

        # Also verify that if penalty were on oscillated x, result would differ
        osc_penalty = float(np.sum(np.maximum(0.0, np.abs(osc_x) - 5.0) ** 2))
        # oscillation(6.0) != 6.0, so penalties differ
        assert osc_penalty != pytest.approx(expected_penalty, abs=1e-10)


class TestBBOBTransformAliases:
    """Verify BBOB alias names work through the full factory pipeline."""

    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("t_osz", OscillationTransform),
            ("t_asy", AsymmetricTransform),
            ("f_pen", BoundaryPenaltyTransform),
        ],
    )
    def test_bbob_transform_aliases(self, factory, alias, canonical):
        params = {}
        if alias == "t_asy":
            params = {"beta": 0.5}
        if alias == "f_pen":
            params = {"bound": 5.0}

        config = {
            "type": alias,
            "parameters": params,
            "function": {"type": "sphere", "parameters": {"dimension": 3}},
        }

        func = factory.create_function(config)

        assert isinstance(func, ComposedFunction)

        if canonical is BoundaryPenaltyTransform:
            assert len(func.penalty_transforms) == 1
            assert isinstance(func.penalty_transforms[0], canonical)
        else:
            assert len(func.input_transforms) == 1
            assert isinstance(func.input_transforms[0], canonical)

        # Evaluates without error
        x = np.zeros(3)
        result = func(x)
        assert np.isfinite(result)


class TestCompositionRejectsOuterVectorTransforms:
    """Verify that wrapping a composition in vector transforms raises an error."""

    @pytest.fixture
    def composition_factory(self, tmp_path):
        """Factory with data files needed for a minimal composition."""
        from pyMOFL.factories.function_factory import DataLoader, FunctionRegistry

        # Create shift data: 2 components × dim=3 → 6 values
        shift_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.savetxt(tmp_path / "shift.txt", shift_data)
        loader = DataLoader(base_path=tmp_path)
        return FunctionFactory(data_loader=loader, registry=FunctionRegistry())

    def _minimal_composition_config(self):
        return {
            "type": "composition",
            "parameters": {
                "dimension": 3,
                "num_functions": 2,
                "shift_file": "shift.txt",
                "sigmas": [1.0, 1.0],
                "lambdas": [1.0, 1.0],
                "biases": [0.0, 100.0],
            },
            "functions": [
                {"type": "sphere", "parameters": {"dimension": 3}},
                {"type": "sphere", "parameters": {"dimension": 3}},
            ],
        }

    def test_oscillation_wrapping_composition_raises(self, composition_factory):
        config = {
            "type": "oscillation",
            "function": self._minimal_composition_config(),
        }
        with pytest.raises(ValueError, match=r"[Vv]ector transform.*composition"):
            composition_factory.create_function(config)

    def test_asymmetric_wrapping_composition_raises(self, composition_factory):
        config = {
            "type": "asymmetric",
            "parameters": {"beta": 0.5},
            "function": self._minimal_composition_config(),
        }
        with pytest.raises(ValueError, match=r"[Vv]ector transform.*composition"):
            composition_factory.create_function(config)

    def test_shift_wrapping_composition_raises(self, composition_factory):
        config = {
            "type": "shift",
            "parameters": {"vector": [1.0, 2.0, 3.0]},
            "function": self._minimal_composition_config(),
        }
        with pytest.raises(ValueError, match=r"[Vv]ector transform.*composition"):
            composition_factory.create_function(config)

    def test_scalar_transform_wrapping_composition_allowed(self, composition_factory):
        """Scalar transforms (bias) around compositions should still work."""
        config = {
            "type": "bias",
            "parameters": {"value": -450.0},
            "function": self._minimal_composition_config(),
        }
        func = composition_factory.create_function(config)
        assert isinstance(func, ComposedFunction)
        assert len(func.output_transforms) == 1

    def test_penalty_wrapping_composition_allowed(self, composition_factory):
        """Penalty transforms around compositions should still work."""
        config = {
            "type": "boundary_penalty",
            "parameters": {"bound": 5.0},
            "function": self._minimal_composition_config(),
        }
        func = composition_factory.create_function(config)
        assert isinstance(func, ComposedFunction)
        assert len(func.penalty_transforms) == 1


class TestCompositionComponentPenaltyPropagation:
    """Verify that penalty transforms on individual composition components are propagated."""

    @pytest.fixture
    def composition_factory(self, tmp_path):
        from pyMOFL.factories.function_factory import DataLoader, FunctionRegistry

        # 2 components × dim=3 → 6 shift values
        shift_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.savetxt(tmp_path / "shift.txt", shift_data)
        loader = DataLoader(base_path=tmp_path)
        return FunctionFactory(data_loader=loader, registry=FunctionRegistry())

    def test_component_penalties_propagated(self, composition_factory):
        """Penalty transforms on components should be included in component ComposedFunctions."""
        config = {
            "type": "composition",
            "parameters": {
                "dimension": 3,
                "num_functions": 2,
                "shift_file": "shift.txt",
                "sigmas": [1.0, 1.0],
                "lambdas": [1.0, 1.0],
                "biases": [0.0, 0.0],
            },
            "functions": [
                {
                    "type": "boundary_penalty",
                    "parameters": {"bound": 5.0},
                    "function": {"type": "sphere", "parameters": {"dimension": 3}},
                },
                {"type": "sphere", "parameters": {"dimension": 3}},
            ],
        }

        func = composition_factory.create_function(config)

        # The composed base is a WeightedComposition
        from pyMOFL.compositions.weighted_composition import WeightedComposition

        assert isinstance(func.base_function, WeightedComposition)
        composition = func.base_function

        # First component should have a penalty transform
        comp0 = composition.components[0]
        assert isinstance(comp0, ComposedFunction)
        assert len(comp0.penalty_transforms) == 1
        assert isinstance(comp0.penalty_transforms[0], BoundaryPenaltyTransform)

        # Second component should have no penalty transforms
        comp1 = composition.components[1]
        assert isinstance(comp1, ComposedFunction)
        assert len(comp1.penalty_transforms) == 0

    def test_component_penalty_affects_evaluation(self, composition_factory):
        """Penalty on a component should affect evaluation at out-of-bounds points."""
        # Two identical compositions, one with penalty on component, one without
        config_with_penalty = {
            "type": "composition",
            "parameters": {
                "dimension": 3,
                "num_functions": 1,
                "shift_file": "shift.txt",
                "sigmas": [1.0],
                "lambdas": [1.0],
                "biases": [0.0],
            },
            "functions": [
                {
                    "type": "boundary_penalty",
                    "parameters": {"bound": 5.0},
                    "function": {"type": "sphere", "parameters": {"dimension": 3}},
                },
            ],
        }
        config_without_penalty = {
            "type": "composition",
            "parameters": {
                "dimension": 3,
                "num_functions": 1,
                "shift_file": "shift.txt",
                "sigmas": [1.0],
                "lambdas": [1.0],
                "biases": [0.0],
            },
            "functions": [
                {"type": "sphere", "parameters": {"dimension": 3}},
            ],
        }

        func_with = composition_factory.create_function(config_with_penalty)
        func_without = composition_factory.create_function(config_without_penalty)

        # Within bounds: results should be close (normalization C may differ due to f_max)
        x_in = np.array([1.0, 1.0, 1.0])
        # Both should evaluate without error
        result_with = func_with(x_in)
        result_without = func_without(x_in)
        assert np.isfinite(result_with)
        assert np.isfinite(result_without)

        # Out of bounds: penalty version should return higher value
        x_out = np.array([10.0, 10.0, 10.0])
        result_with_out = func_with(x_out)
        result_without_out = func_without(x_out)
        assert result_with_out > result_without_out
