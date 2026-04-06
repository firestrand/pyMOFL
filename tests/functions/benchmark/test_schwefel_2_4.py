"""
Tests for Schwefel 2.4 (Extended Rosenbrock) function.

f(x) = Σ_{i=2}^{D} [(x_1 - x_i²)² + (x_i - 1)²]

References:
    Schwefel, H.P. (1981). "Numerical Optimization of Computer Models".
"""

import numpy as np

from pyMOFL.functions.benchmark.schwefel import Schwefel_2_4


class TestSchwefel_2_4:
    """Tests for Schwefel 2.4 — Extended Rosenbrock with star dependency on x_1."""

    def test_initialization(self):
        func = Schwefel_2_4(dimension=3)
        assert func.dimension == 3
        np.testing.assert_allclose(func.operational_bounds.low, [-10, -10, -10])
        np.testing.assert_allclose(func.operational_bounds.high, [10, 10, 10])

    def test_global_minimum(self):
        for d in [2, 3, 5]:
            func = Schwefel_2_4(dimension=d)
            x_opt = np.ones(d)
            assert np.isclose(func.evaluate(x_opt), 0.0, atol=1e-15)

    def test_get_global_minimum(self):
        func = Schwefel_2_4(dimension=4)
        point, value = func.get_global_minimum()
        np.testing.assert_array_equal(point, np.ones(4))
        assert value == 0.0

    def test_known_values(self):
        func = Schwefel_2_4(dimension=3)
        # f(0,0,0): (0-0)^2+(0-1)^2 + (0-0)^2+(0-1)^2 = 2
        assert np.isclose(func.evaluate(np.array([0.0, 0.0, 0.0])), 2.0)
        # f(2,1,1): (2-1)^2+(1-1)^2 + (2-1)^2+(1-1)^2 = 2
        assert np.isclose(func.evaluate(np.array([2.0, 1.0, 1.0])), 2.0)
        # f(4,2,2): (4-4)^2+(2-1)^2 + (4-4)^2+(2-1)^2 = 2
        assert np.isclose(func.evaluate(np.array([4.0, 2.0, 2.0])), 2.0)

    def test_star_dependency(self):
        """x_1 appears in every term — changing x_1 affects all terms."""
        func = Schwefel_2_4(dimension=3)
        base = func.evaluate(np.array([1.0, 1.0, 1.0]))
        perturbed = func.evaluate(np.array([2.0, 1.0, 1.0]))
        assert base == 0.0
        assert perturbed > 0.0  # Changing x_1 from optimum increases value

    def test_non_negative(self):
        func = Schwefel_2_4(dimension=4)
        rng = np.random.default_rng(42)
        for _ in range(50):
            assert func.evaluate(rng.uniform(-10, 10, 4)) >= 0.0

    def test_batch_evaluation(self):
        func = Schwefel_2_4(dimension=3)
        X = np.array([[1, 1, 1], [0, 0, 0], [4, 2, 2]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-12)

    def test_batch_shape(self):
        func = Schwefel_2_4(dimension=4)
        assert func.evaluate_batch(np.ones((3, 4))).shape == (3,)

    def test_dimension_2(self):
        """Minimum case: only one term in the sum."""
        func = Schwefel_2_4(dimension=2)
        # f(1, 1) = (1-1)^2 + (1-1)^2 = 0
        assert np.isclose(func.evaluate(np.array([1.0, 1.0])), 0.0)
        # f(0, 0) = (0-0)^2 + (0-1)^2 = 1
        assert np.isclose(func.evaluate(np.array([0.0, 0.0])), 1.0)
