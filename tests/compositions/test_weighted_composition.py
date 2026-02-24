"""Tests for the WeightedComposition class."""

import numpy as np
import pytest

from pyMOFL.compositions import WeightedComposition
from pyMOFL.functions.benchmark.sphere import SphereFunction


class TestWeightedCompositionInit:
    """Test initialization and validation."""

    def test_basic_initialization(self):
        s1 = SphereFunction(dimension=2)
        s2 = SphereFunction(dimension=2)
        wc = WeightedComposition(
            dimension=2,
            components=[s1, s2],
            optima=[np.zeros(2), np.ones(2)],
            sigmas=[1.0, 1.0],
        )
        assert wc.dimension == 2
        assert len(wc.components) == 2
        assert wc.biases == [0.0, 0.0]
        assert wc.global_bias == 0.0
        assert wc.dominance_suppression is False
        assert wc.non_continuous is False

    def test_mismatched_optima_raises(self):
        s = SphereFunction(dimension=2)
        with pytest.raises(ValueError, match="optima"):
            WeightedComposition(
                dimension=2,
                components=[s, s],
                optima=[np.zeros(2)],
                sigmas=[1.0, 1.0],
            )

    def test_mismatched_sigmas_raises(self):
        s = SphereFunction(dimension=2)
        with pytest.raises(ValueError, match="sigmas"):
            WeightedComposition(
                dimension=2,
                components=[s, s],
                optima=[np.zeros(2), np.ones(2)],
                sigmas=[1.0],
            )

    def test_mismatched_biases_raises(self):
        s = SphereFunction(dimension=2)
        with pytest.raises(ValueError, match="biases"):
            WeightedComposition(
                dimension=2,
                components=[s, s],
                optima=[np.zeros(2), np.ones(2)],
                sigmas=[1.0, 1.0],
                biases=[0.0],
            )


class TestGaussianWeighting:
    """Test Gaussian weight computation."""

    def test_equal_distance_equal_weights(self):
        """At the midpoint between two optima with same sigma, weights should be equal."""
        s1 = SphereFunction(dimension=2)
        s2 = SphereFunction(dimension=2)
        wc = WeightedComposition(
            dimension=2,
            components=[s1, s2],
            optima=[np.array([-1.0, 0.0]), np.array([1.0, 0.0])],
            sigmas=[1.0, 1.0],
        )
        w = wc._compute_weights(np.array([0.0, 0.0]))
        np.testing.assert_allclose(w[0], w[1], rtol=1e-12)
        np.testing.assert_allclose(np.sum(w), 1.0)

    def test_closer_to_first_gets_more_weight(self):
        """Point closer to first optimum should give it more weight."""
        s1 = SphereFunction(dimension=2)
        s2 = SphereFunction(dimension=2)
        wc = WeightedComposition(
            dimension=2,
            components=[s1, s2],
            optima=[np.array([0.0, 0.0]), np.array([10.0, 0.0])],
            sigmas=[1.0, 1.0],
        )
        w = wc._compute_weights(np.array([1.0, 0.0]))
        assert w[0] > w[1]

    def test_weights_sum_to_one(self):
        s = SphereFunction(dimension=3)
        wc = WeightedComposition(
            dimension=3,
            components=[s, s, s],
            optima=[np.zeros(3), np.ones(3), -np.ones(3)],
            sigmas=[1.0, 2.0, 0.5],
        )
        w = wc._compute_weights(np.array([0.5, 0.5, 0.5]))
        np.testing.assert_allclose(np.sum(w), 1.0)


class TestDominanceSuppression:
    """Test CEC-style winner-takes-most suppression."""

    def test_suppression_amplifies_dominant_weight(self):
        """With dominance suppression, the dominant component should get even more weight."""
        s1 = SphereFunction(dimension=2)
        s2 = SphereFunction(dimension=2)
        optima = [np.array([0.0, 0.0]), np.array([10.0, 0.0])]

        wc_no_supp = WeightedComposition(
            dimension=2,
            components=[s1, s2],
            optima=optima,
            sigmas=[1.0, 1.0],
            dominance_suppression=False,
        )
        wc_supp = WeightedComposition(
            dimension=2,
            components=[s1, s2],
            optima=optima,
            sigmas=[1.0, 1.0],
            dominance_suppression=True,
        )

        x = np.array([1.0, 0.0])
        w_no = wc_no_supp._compute_weights(x)
        w_yes = wc_supp._compute_weights(x)

        # With suppression, the dominant weight should be larger
        assert w_yes[0] > w_no[0]
        np.testing.assert_allclose(np.sum(w_yes), 1.0)


class TestEvaluate:
    """Test evaluate method."""

    def test_single_component_at_origin(self):
        """Single sphere component with no bias/normalization should return sphere value."""
        sphere = SphereFunction(dimension=2)
        wc = WeightedComposition(
            dimension=2,
            components=[sphere],
            optima=[np.zeros(2)],
            sigmas=[1.0],
        )
        assert wc.evaluate(np.zeros(2)) == 0.0
        assert wc.evaluate(np.array([1.0, 1.0])) == 2.0

    def test_per_component_bias(self):
        """Biases should be added inside weight multiplication."""
        sphere = SphereFunction(dimension=2)
        wc = WeightedComposition(
            dimension=2,
            components=[sphere],
            optima=[np.zeros(2)],
            sigmas=[1.0],
            biases=[100.0],
        )
        # At origin: f=0, bias=100 => weight*100 = 1.0*100 = 100
        assert wc.evaluate(np.zeros(2)) == 100.0

    def test_global_bias(self):
        """Global bias should be added to total."""
        sphere = SphereFunction(dimension=2)
        wc = WeightedComposition(
            dimension=2,
            components=[sphere],
            optima=[np.zeros(2)],
            sigmas=[1.0],
            global_bias=42.0,
        )
        assert wc.evaluate(np.zeros(2)) == 42.0

    def test_two_components_equal_optima(self):
        """Two identical components at same optimum should average."""
        sphere = SphereFunction(dimension=2)
        wc = WeightedComposition(
            dimension=2,
            components=[sphere, sphere],
            optima=[np.zeros(2), np.zeros(2)],
            sigmas=[1.0, 1.0],
            biases=[0.0, 100.0],
        )
        # Equal weights (both at same optimum) => 0.5*(0+0) + 0.5*(0+100) = 50
        result = wc.evaluate(np.zeros(2))
        np.testing.assert_allclose(result, 50.0)

    def test_evaluate_batch(self):
        """Batch evaluation should match individual evaluations."""
        sphere = SphereFunction(dimension=2)
        wc = WeightedComposition(
            dimension=2,
            components=[sphere],
            optima=[np.zeros(2)],
            sigmas=[1.0],
        )
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
        expected = np.array([wc.evaluate(X[i]) for i in range(3)])
        np.testing.assert_allclose(wc.evaluate_batch(X), expected)

    def test_dimension_validation(self):
        """Should reject wrong-dimension input."""
        sphere = SphereFunction(dimension=2)
        wc = WeightedComposition(
            dimension=2,
            components=[sphere],
            optima=[np.zeros(2)],
            sigmas=[1.0],
        )
        with pytest.raises(ValueError):
            wc.evaluate(np.array([1.0, 2.0, 3.0]))


class TestNonContinuousFirstComponent:
    """Test F23-style non-continuous preprocessing."""

    def test_noncontinuous_map_leaves_close_values(self):
        """Values close to first optimum (|x-o1| < 0.5) should be unchanged."""
        sphere = SphereFunction(dimension=2)
        o1 = np.array([1.0, 2.0])
        wc = WeightedComposition(
            dimension=2,
            components=[sphere],
            optima=[o1],
            sigmas=[1.0],
            non_continuous=True,
        )
        x = o1 + np.array([0.1, -0.2])
        y = wc._noncontinuous_map(x)
        np.testing.assert_array_equal(y, x)

    def test_noncontinuous_map_rounds_far_values(self):
        """Values far from first optimum (|x-o1| >= 0.5) should be rounded."""
        sphere = SphereFunction(dimension=2)
        o1 = np.array([0.0, 0.0])
        wc = WeightedComposition(
            dimension=2,
            components=[sphere],
            optima=[o1],
            sigmas=[1.0],
            non_continuous=True,
        )
        x = np.array([1.3, -0.8])
        y = wc._noncontinuous_map(x)
        # 2*1.3=2.6 => floor=2, b=0.6>=0.5 and v>0 => res=3 => 3/2=1.5
        # 2*(-0.8)=-1.6 => floor=-2, b=0.4<0.5 => res=-2 => -2/2=-1.0
        np.testing.assert_allclose(y, [1.5, -1.0])

    def test_flag_only_affects_first_component(self):
        """Non-continuous flag should only apply to component 0."""
        s1 = SphereFunction(dimension=2)
        s2 = SphereFunction(dimension=2)
        o1 = np.array([0.0, 0.0])
        o2 = np.array([5.0, 5.0])
        wc = WeightedComposition(
            dimension=2,
            components=[s1, s2],
            optima=[o1, o2],
            sigmas=[1.0, 1.0],
            non_continuous=True,
        )
        # Evaluate — should not raise
        result = wc.evaluate(np.array([1.3, -0.8]))
        assert np.isfinite(result)


class TestWithComposedFunction:
    """Test WeightedComposition with ComposedFunction components (the target use case)."""

    def test_with_shift_transform(self):
        """Components with ShiftTransform should evaluate correctly."""
        from pyMOFL.functions.transformations import ComposedFunction, ShiftTransform

        sphere = SphereFunction(dimension=2)
        shift_vec = np.array([1.0, 2.0])
        composed = ComposedFunction(
            base_function=sphere,
            input_transforms=[ShiftTransform(shift_vec)],
        )
        wc = WeightedComposition(
            dimension=2,
            components=[composed],
            optima=[shift_vec],
            sigmas=[1.0],
        )
        # At the optimum, shift makes x=[1,2] -> [0,0], sphere=0
        np.testing.assert_allclose(wc.evaluate(shift_vec), 0.0)

    def test_with_normalize_transform(self):
        """Components with NormalizeTransform should scale output by C/f_max."""
        from pyMOFL.functions.transformations import ComposedFunction, NormalizeTransform

        sphere = SphereFunction(dimension=2)
        f_max = 50.0
        C = 2000.0
        composed = ComposedFunction(
            base_function=sphere,
            output_transforms=[NormalizeTransform(C=C, f_max=f_max)],
        )
        wc = WeightedComposition(
            dimension=2,
            components=[composed],
            optima=[np.zeros(2)],
            sigmas=[1.0],
        )
        x = np.array([1.0, 1.0])
        # sphere([1,1]) = 2, normalized = 2 * 2000/50 = 80
        np.testing.assert_allclose(wc.evaluate(x), 80.0)
