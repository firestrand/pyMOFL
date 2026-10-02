"""Tests for vectorized batch hotpaths and pre-allocated buffer (out=) support."""

from __future__ import annotations

import numpy as np

from pyMOFL.compositions import HybridFunction, WeightedComposition
from pyMOFL.functions.benchmark.alpine import Alpine1Function, Alpine2Function
from pyMOFL.functions.benchmark.chebyshev import ChebyshevFunction
from pyMOFL.functions.benchmark.cola import ColaFunction
from pyMOFL.functions.benchmark.deceptive import DeceptiveFunction
from pyMOFL.functions.benchmark.lennard_jones import LennardJonesFunction
from pyMOFL.functions.benchmark.network import NetworkFunction
from pyMOFL.functions.benchmark.rastrigin import RastriginFunction
from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction
from pyMOFL.functions.benchmark.sphere import SphereFunction
from pyMOFL.functions.transformations import (
    BiasTransform,
    BoundaryPenaltyTransform,
    ComposedFunction,
    IndexedRotateTransform,
    IndexedScaleTransform,
    IndexedShiftTransform,
    NoiseTransform,
    RotateTransform,
    ScaleTransform,
    ShiftTransform,
)


class TestWeightedCompositionBatch:
    """Test vectorized batch evaluation in WeightedComposition."""

    def test_batch_matches_single_evaluate(self):
        s1 = SphereFunction(dimension=3)
        s2 = SphereFunction(dimension=3)
        wc = WeightedComposition(
            dimension=3,
            components=[s1, s2],
            optima=[np.zeros(3), np.ones(3)],
            sigmas=[1.0, 2.0],
            biases=[10.0, 20.0],
            global_bias=5.0,
        )

        np.random.seed(42)
        X = np.random.uniform(-5.0, 5.0, size=(25, 3))
        expected = np.array([wc.evaluate(row) for row in X])
        actual = wc.evaluate_batch(X)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_batch_with_inverse_distance_and_dominance(self):
        s1 = SphereFunction(dimension=2)
        s2 = SphereFunction(dimension=2)
        o1 = np.array([0.0, 0.0])
        o2 = np.array([3.0, 3.0])
        wc = WeightedComposition(
            dimension=2,
            components=[s1, s2],
            optima=[o1, o2],
            sigmas=[1.0, 1.0],
            biases=[0.0, 100.0],
            inverse_distance_weight=True,
            dominance_suppression=True,
        )

        # Include exact optima in the batch to test distance = 0 handling
        X = np.array(
            [
                [0.0, 0.0],
                [3.0, 3.0],
                [1.5, 1.5],
                [10.0, 10.0],
                [-2.0, 1.0],
            ]
        )
        expected = np.array([wc.evaluate(row) for row in X])
        actual = wc.evaluate_batch(X)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_batch_non_continuous(self):
        s1 = SphereFunction(dimension=2)
        s2 = SphereFunction(dimension=2)
        wc = WeightedComposition(
            dimension=2,
            components=[s1, s2],
            optima=[np.array([1.0, 2.0]), np.array([4.0, 5.0])],
            sigmas=[1.0, 1.0],
            non_continuous=True,
        )

        X = np.array(
            [
                [1.2, 2.3],
                [1.8, 2.7],
                [3.1, 4.4],
                [-0.5, 0.6],
            ]
        )
        expected = np.array([wc.evaluate(row) for row in X])
        actual = wc.evaluate_batch(X)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_batch_with_out_buffer(self):
        s1 = SphereFunction(dimension=2)
        s2 = SphereFunction(dimension=2)
        wc = WeightedComposition(
            dimension=2,
            components=[s1, s2],
            optima=[np.zeros(2), np.ones(2)],
            sigmas=[1.0, 1.0],
        )

        X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        buf = np.zeros(3)
        res = wc.evaluate_batch(X, out=buf)

        assert res is buf
        expected = np.array([wc.evaluate(row) for row in X])
        np.testing.assert_allclose(buf, expected, rtol=1e-12, atol=1e-12)


class TestHybridFunctionBatch:
    """Test vectorized batch evaluation in HybridFunction."""

    def test_batch_matches_single_evaluate(self):
        s = SphereFunction(dimension=2)
        r = RastriginFunction(dimension=2)
        hf = HybridFunction(
            components=[s, r],
            partitions=[(0, 2), (2, 4)],
            weights=[0.4, 0.6],
        )

        np.random.seed(42)
        X = np.random.uniform(-4.0, 4.0, size=(20, 4))
        expected = np.array([hf.evaluate(row) for row in X])
        actual = hf.evaluate_batch(X)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_batch_dimension_padding_and_truncation(self):
        s = SphereFunction(dimension=3)
        r = RastriginFunction(dimension=1)
        # s needs 3 dims, slice has 2 (needs padding). r needs 1 dim, slice has 2 (needs truncation).
        hf = HybridFunction(
            components=[s, r],
            partitions=[(0, 2), (2, 4)],
        )

        X = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [-1.0, -2.0, 0.5, 1.5],
            ]
        )
        expected = np.array([hf.evaluate(row) for row in X])
        actual = hf.evaluate_batch(X)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_batch_with_out_buffer(self):
        s = SphereFunction(dimension=2)
        hf = HybridFunction(components=[s], partitions=[(0, 2)])
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        buf = np.zeros(2)
        res = hf.evaluate_batch(X, out=buf)

        assert res is buf
        expected = np.array([hf.evaluate(row) for row in X])
        np.testing.assert_allclose(buf, expected, rtol=1e-12, atol=1e-12)


class TestIndexedTransformsBatch:
    """Test transform_batch and out= buffer support on indexed transformations."""

    def test_indexed_shift_batch(self):
        shifts = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        t = IndexedShiftTransform(shifts=shifts, component_index=1)

        X = np.array([[10.0, 20.0], [30.0, 40.0]])
        expected = np.array([t(row) for row in X])
        actual = t.transform_batch(X)
        np.testing.assert_allclose(actual, expected)

        buf = np.zeros_like(X)
        res = t.transform_batch(X, out=buf)
        assert res is buf
        np.testing.assert_allclose(buf, expected)

    def test_indexed_scale_batch(self):
        factors = [2.0, 5.0]
        t = IndexedScaleTransform(factors=factors, component_index=1)

        X = np.array([[10.0, 20.0], [30.0, 40.0]])
        expected = np.array([t(row) for row in X])
        actual = t.transform_batch(X)
        np.testing.assert_allclose(actual, expected)

        buf = np.zeros_like(X)
        res = t.transform_batch(X, out=buf)
        assert res is buf
        np.testing.assert_allclose(buf, expected)

    def test_indexed_rotate_batch(self):
        M = np.array([[0.0, 1.0], [-1.0, 0.0]])
        t = IndexedRotateTransform(matrices=M)

        X = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 5.0]])
        expected = np.array([t(row) for row in X])
        actual = t.transform_batch(X)
        np.testing.assert_allclose(actual, expected)

        buf = np.zeros_like(X)
        res = t.transform_batch(X, out=buf)
        assert res is buf
        np.testing.assert_allclose(buf, expected)

    def test_noise_batch(self):
        t = NoiseTransform(noise_level=0.4, seed=123)
        Y = np.array([10.0, 20.0, 30.0])
        # Calling transform_batch should return an array of matching shape
        out = t.transform_batch(Y)
        assert out.shape == Y.shape
        assert np.all(out >= Y)  # Since 1 + 0.4*abs(randn) >= 1.0


class TestBenchmarkFunctionsBatchVectorization:
    """Test batch evaluation matches row evaluation for optimized benchmark functions."""

    def test_schwefel_sin_batch_with_boundary_handling(self):
        f = SchwefelSinFunction(dimension=3, boundary_handling=True)
        # Mix of inside bounds, > 500, < -500
        X = np.array(
            [
                [420.9687, 420.9687, 420.9687],
                [550.0, 100.0, -600.0],
                [-510.0, -520.0, 501.0],
                [0.0, 0.0, 0.0],
            ]
        )
        expected = np.array([f.evaluate(row) for row in X])
        actual = f.evaluate_batch(X)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_alpine1_batch(self):
        f = Alpine1Function(dimension=2)
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [-3.0, 4.0],
                [5.5, -6.2],
            ]
        )
        expected = np.array([f.evaluate(row) for row in X])
        actual = f.evaluate_batch(X)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_alpine2_batch(self):
        f = Alpine2Function(dimension=3)
        # Include zeros and negative numbers
        X = np.array(
            [
                [7.917, 7.917, 7.917],
                [1.0, 2.0, 3.0],
                [0.0, 2.0, 3.0],
                [-1.0, 2.0, 3.0],
                [4.0, -0.5, 6.0],
            ]
        )
        expected = np.array([f.evaluate(row) for row in X])
        actual = f.evaluate_batch(X)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_deceptive_batch(self):
        f = DeceptiveFunction(dimension=4)
        np.random.seed(42)
        X = np.random.uniform(0.0, 1.0, size=(25, 4))
        expected = np.array([f.evaluate(row) for row in X])
        actual = f.evaluate_batch(X)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_cola_batch(self):
        f = ColaFunction()
        np.random.seed(42)
        X = np.random.uniform(-4.0, 4.0, size=(15, 17))
        expected = np.array([f.evaluate(row) for row in X])
        actual = f.evaluate_batch(X)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_network_batch(self):
        f = NetworkFunction()
        np.random.seed(42)
        X = np.random.uniform(0.0, 20.0, size=(10, f.dimension))
        expected = np.array([f.evaluate(row) for row in X])
        actual = f.evaluate_batch(X)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_lennard_jones_batch(self):
        f = LennardJonesFunction(n_atoms=4)
        np.random.seed(42)
        X = np.random.uniform(-2.0, 2.0, size=(10, 12))
        expected = np.array([f.evaluate(row) for row in X])
        actual = f.evaluate_batch(X)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_chebyshev_batch(self):
        f = ChebyshevFunction(dimension=9)
        np.random.seed(42)
        X = np.random.uniform(-10.0, 10.0, size=(12, 9))
        expected = np.array([f.evaluate(row) for row in X])
        actual = f.evaluate_batch(X)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


class TestComposedFunctionAndCoreTransformsBuffers:
    """Test out= pre-allocated buffer propagation across ComposedFunction and transforms."""

    def test_shift_and_scale_out_buffer(self):
        shift = ShiftTransform(np.array([1.0, 2.0]))
        scale = ScaleTransform(2.0)
        X = np.array([[3.0, 4.0], [5.0, 6.0]])

        buf = np.empty_like(X)
        r1 = shift.transform_batch(X, out=buf)
        assert r1 is buf
        np.testing.assert_allclose(buf, [[2.0, 2.0], [4.0, 4.0]])

        r2 = scale.transform_batch(buf, out=buf)
        assert r2 is buf
        np.testing.assert_allclose(buf, [[1.0, 1.0], [2.0, 2.0]])

    def test_rotate_out_buffer(self):
        M = np.array([[0.0, 1.0], [-1.0, 0.0]])
        rot = RotateTransform(M)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        buf = np.empty_like(X)
        res = rot.transform_batch(X, out=buf)
        assert res is buf
        expected = np.array([rot(row) for row in X])
        np.testing.assert_allclose(buf, expected)

    def test_bias_and_boundary_penalty_out_buffer(self):
        bias = BiasTransform(10.0)
        Y = np.array([1.0, 2.0, 3.0])
        buf_y = np.empty_like(Y)
        res_y = bias.transform_batch(Y, out=buf_y)
        assert res_y is buf_y
        np.testing.assert_allclose(buf_y, [11.0, 12.0, 13.0])

        penalty = BoundaryPenaltyTransform(bound=5.0)
        X = np.array([[4.0, 4.0], [6.0, 4.0], [6.0, 7.0]])
        buf_p = np.empty(3)
        res_p = penalty.compute_batch(X, out=buf_p)
        assert res_p is buf_p
        np.testing.assert_allclose(buf_p, [0.0, 1.0, 5.0])

    def test_composed_function_out_buffer(self):
        sphere = SphereFunction(dimension=2)
        shift = ShiftTransform(np.array([1.0, 1.0]))
        bias = BiasTransform(10.0)
        pen = BoundaryPenaltyTransform(bound=5.0)

        cf = ComposedFunction(
            base_function=sphere,
            input_transforms=[shift],
            output_transforms=[bias],
            penalty_transforms=[pen],
        )

        X = np.array([[1.0, 1.0], [3.0, 1.0], [6.0, 1.0]])
        buf = np.empty(3)
        res = cf.evaluate_batch(X, out=buf)
        assert res is buf

        expected = np.array([cf.evaluate(row) for row in X])
        np.testing.assert_allclose(buf, expected, rtol=1e-12, atol=1e-12)
