"""
Tests for Schwefel 2.36 function.

f(x) = Σ_{i=1}^{D} -x_i * sin(sqrt(|x_i|))

Same as Schwefel 2.26 but with domain restricted to [0, 500].

References:
    Schwefel, H.P. (1981). "Numerical Optimization of Computer Models".
"""

import numpy as np

from pyMOFL.functions.benchmark.schwefel import Schwefel_2_36


class TestSchwefel_2_36:
    """Tests for Schwefel 2.36 — sine-root sum on non-negative domain."""

    def test_initialization(self):
        func = Schwefel_2_36(dimension=2)
        assert func.dimension == 2
        np.testing.assert_allclose(func.operational_bounds.low, [0, 0])
        np.testing.assert_allclose(func.operational_bounds.high, [500, 500])

    def test_value_at_origin(self):
        func = Schwefel_2_36(dimension=2)
        # f(0, 0) = -0*sin(0) + -0*sin(0) = 0
        assert np.isclose(func.evaluate(np.zeros(2)), 0.0, atol=1e-15)

    def test_global_minimum(self):
        """Global minimum at (420.9687, ..., 420.9687) ≈ -418.98 × D."""
        func = Schwefel_2_36(dimension=2)
        x_opt = np.full(2, 420.9687)
        val = func.evaluate(x_opt)
        assert np.isclose(val, -418.9829 * 2, atol=0.1)

    def test_get_global_minimum(self):
        point, value = Schwefel_2_36.get_global_minimum(3)
        np.testing.assert_allclose(point, np.full(3, 420.9687), atol=0.001)
        assert np.isclose(value, -418.9829 * 3, atol=0.1)

    def test_known_value_1d_component(self):
        """Each term: -x_i * sin(sqrt(|x_i|)). Check one component."""
        func = Schwefel_2_36(dimension=2)
        # f(420.9687, 0) = -420.9687*sin(sqrt(420.9687)) + 0 ≈ -418.98
        val = func.evaluate(np.array([420.9687, 0.0]))
        assert np.isclose(val, -418.9829, atol=0.01)

    def test_separable(self):
        """Function is separable — f(x) = Σ g(x_i)."""
        func = Schwefel_2_36(dimension=2)
        a, b = 100.0, 200.0
        f_ab = func.evaluate(np.array([a, b]))
        f_a0 = func.evaluate(np.array([a, 0.0]))
        f_0b = func.evaluate(np.array([0.0, b]))
        # f(a,b) = g(a) + g(b) = f(a,0) + f(0,b) since g(0) = 0
        assert np.isclose(f_ab, f_a0 + f_0b, atol=1e-10)

    def test_multimodal(self):
        """Multiple local minima due to sin oscillations."""
        func = Schwefel_2_36(dimension=2)
        rng = np.random.default_rng(42)
        vals = [func.evaluate(rng.uniform(0, 500, 2)) for _ in range(200)]
        assert max(vals) - min(vals) > 50

    def test_scalable(self):
        for d in [2, 3, 5]:
            func = Schwefel_2_36(dimension=d)
            assert np.isfinite(func.evaluate(np.zeros(d)))

    def test_batch_evaluation(self):
        func = Schwefel_2_36(dimension=2)
        X = np.array([[0, 0], [420.9687, 0], [100, 200]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-10)

    def test_batch_shape(self):
        func = Schwefel_2_36(dimension=3)
        assert func.evaluate_batch(np.zeros((4, 3))).shape == (4,)

    def test_higher_dim_additive(self):
        """With more dimensions, minimum scales linearly: f* ≈ -418.98 × D."""
        func = Schwefel_2_36(dimension=5)
        x_opt = np.full(5, 420.9687)
        val = func.evaluate(x_opt)
        assert np.isclose(val, -418.9829 * 5, atol=0.5)
