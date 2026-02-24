"""
Tests for the WeightedComposition class (migrated from CompositeFunction).
"""

import numpy as np
import pytest

from pyMOFL.compositions import WeightedComposition
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.functions import RastriginFunction, SphereFunction


class TestWeightedComposition:
    """Tests for the WeightedComposition class."""

    def test_initialization(self):
        """Test initialization with default and custom bounds."""
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)

        components = [sphere, rastrigin]
        optima = [np.zeros(2), np.zeros(2)]
        sigmas = [1.0, 2.0]
        biases = [0.0, 100.0]
        wc = WeightedComposition(
            dimension=2,
            components=components,
            optima=optima,
            sigmas=sigmas,
            biases=biases,
        )

        assert wc.dimension == 2
        assert len(wc.components) == 2
        assert wc.sigmas == [1.0, 2.0]
        assert wc.biases == [0.0, 100.0]
        np.testing.assert_array_equal(wc.optima[0], np.zeros(2))
        np.testing.assert_array_equal(wc.optima[1], np.zeros(2))

        # Create with custom bounds
        custom_init_bounds = Bounds(
            low=np.array([-10, -5]),
            high=np.array([10, 5]),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        custom_oper_bounds = Bounds(
            low=np.array([-10, -5]),
            high=np.array([10, 5]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        wc = WeightedComposition(
            dimension=2,
            components=components,
            optima=optima,
            sigmas=sigmas,
            biases=biases,
            initialization_bounds=custom_init_bounds,
            operational_bounds=custom_oper_bounds,
        )

        assert wc.dimension == 2
        np.testing.assert_array_equal(wc.initialization_bounds.low, custom_init_bounds.low)
        np.testing.assert_array_equal(wc.operational_bounds.high, custom_oper_bounds.high)

    def test_parameter_validation(self):
        """Test that parameters are validated correctly."""
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)
        components = [sphere, rastrigin]
        optima = [np.zeros(2), np.zeros(2)]

        # Mismatched sigmas length
        with pytest.raises(ValueError):
            WeightedComposition(
                dimension=2,
                components=components,
                optima=optima,
                sigmas=[1.0],
                biases=[0.0, 100.0],
            )

        # Mismatched biases length
        with pytest.raises(ValueError):
            WeightedComposition(
                dimension=2,
                components=components,
                optima=optima,
                sigmas=[1.0, 2.0],
                biases=[0.0],
            )

        # Mismatched optima length
        with pytest.raises(ValueError):
            WeightedComposition(
                dimension=2,
                components=components,
                optima=[np.zeros(2)],
                sigmas=[1.0, 2.0],
                biases=[0.0, 100.0],
            )

    def test_evaluate_simple(self):
        """Test the evaluate method with a single component."""
        sphere = SphereFunction(dimension=2)

        wc = WeightedComposition(
            dimension=2,
            components=[sphere],
            optima=[np.zeros(2)],
            sigmas=[1.0],
            biases=[0.0],
        )

        # At origin: sphere(0,0) = 0
        assert wc.evaluate(np.array([0.0, 0.0])) == 0.0

        # At [2, 3]: sphere = 4 + 9 = 13
        x = np.array([2.0, 3.0])
        assert wc.evaluate(x) == 13.0

    def test_evaluate_multiple_components(self):
        """Test the evaluate method with multiple components."""
        sphere = SphereFunction(dimension=2)
        rastrigin = RastriginFunction(dimension=2)

        wc = WeightedComposition(
            dimension=2,
            components=[sphere, rastrigin],
            optima=[np.zeros(2), np.zeros(2)],
            sigmas=[1.0, 2.0],
            biases=[0.0, 100.0],
        )

        # At origin: both optima at origin, equal weights (0.5 each)
        # 0.5*(sphere(0,0) + 0) + 0.5*(rastrigin(0,0) + 100)
        # = 0.5*0 + 0.5*100 = 50.0
        assert wc.evaluate(np.array([0.0, 0.0])) == 50.0

        # At [1,1]: weights differ due to different sigmas, result > 0
        x = np.array([1.0, 1.0])
        result = wc.evaluate(x)
        assert result > 0.0

    def test_evaluate_with_shifted_optima(self):
        """Test evaluation with components having different optima."""
        sphere1 = SphereFunction(dimension=2)
        sphere2 = SphereFunction(dimension=2)

        opt1 = np.array([0.0, 0.0])
        opt2 = np.array([3.0, 4.0])

        wc = WeightedComposition(
            dimension=2,
            components=[sphere1, sphere2],
            optima=[opt1, opt2],
            sigmas=[1.0, 1.0],
            biases=[0.0, 0.0],
        )

        # At [0,0]: close to opt1, weight heavily on component 0
        # sphere1(0,0) = 0, sphere2(0,0) = 0 — both evaluate at same x
        val_at_origin = wc.evaluate(np.zeros(2))
        assert np.isfinite(val_at_origin)

        # At [3,4]: close to opt2, weight heavily on component 1
        val_at_opt2 = wc.evaluate(np.array([3.0, 4.0]))
        assert np.isfinite(val_at_opt2)

    def test_evaluate_batch(self):
        """Test the evaluate_batch method."""
        sphere = SphereFunction(dimension=2)

        wc = WeightedComposition(
            dimension=2,
            components=[sphere],
            optima=[np.zeros(2)],
            sigmas=[1.0],
            biases=[0.0],
        )

        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
        expected = np.array([0.0, 2.0, 13.0])
        np.testing.assert_allclose(wc.evaluate_batch(X), expected)

    def test_dimension_validation(self):
        """Test that input dimension is validated correctly."""
        sphere = SphereFunction(dimension=2)

        wc = WeightedComposition(
            dimension=2,
            components=[sphere],
            optima=[np.zeros(2)],
            sigmas=[1.0],
            biases=[0.0],
        )

        # Wrong dimension for evaluate
        with pytest.raises(ValueError):
            wc.evaluate(np.array([1.0, 2.0, 3.0]))

        # Wrong dimension for evaluate_batch
        with pytest.raises(ValueError):
            wc.evaluate_batch(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_global_bias(self):
        """Test that global_bias is added to the result."""
        sphere = SphereFunction(dimension=2)

        wc = WeightedComposition(
            dimension=2,
            components=[sphere],
            optima=[np.zeros(2)],
            sigmas=[1.0],
            biases=[0.0],
            global_bias=42.0,
        )

        # At origin: sphere(0,0) + global_bias = 0 + 42 = 42
        assert wc.evaluate(np.zeros(2)) == 42.0

    def test_biases_default_to_zeros(self):
        """Test that biases default to zeros when not provided."""
        sphere = SphereFunction(dimension=2)

        wc = WeightedComposition(
            dimension=2,
            components=[sphere],
            optima=[np.zeros(2)],
            sigmas=[1.0],
        )

        assert wc.biases == [0.0]
        assert wc.evaluate(np.zeros(2)) == 0.0
