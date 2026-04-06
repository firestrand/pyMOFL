"""
Tests for the Büche-Rastrigin function (BBOB f4 base).

f(x) = sum(s_i * (x_i^2 - 10*cos(2*pi*x_i)) + 10)

Where s_i applies asymmetric scaling:
- For even indices: s_i = 1 if x_i > 0, s_i = 10^(0.5 * i/(D-1)) otherwise
  (but BBOB spec applies s_i as multiplier to the whole term)

The Büche-Rastrigin is a variant with per-element asymmetric scaling that
breaks the symmetry of the standard Rastrigin.

Global minimum at x_opt = 0 with value 0 (before external transforms).
"""

import numpy as np
import pytest

from tests.utils.benchmark_validation import BenchmarkValidator


class TestBucheRastriginFunction:
    """Tests for BucheRastriginFunction."""

    def test_contract_dim2(self):
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction

        func = BucheRastriginFunction(dimension=2)
        BenchmarkValidator.assert_contract(func)

    def test_contract_multiple_dimensions(self):
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction

        BenchmarkValidator.assert_contract_multiple_dimensions(
            BucheRastriginFunction, dimensions=[2, 5, 10, 30]
        )

    def test_evaluate_at_origin(self):
        """f(0) = 0."""
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction

        func = BucheRastriginFunction(dimension=5)
        np.testing.assert_allclose(func.evaluate(np.zeros(5)), 0.0, atol=1e-12)

    def test_positive_away_from_origin(self):
        """Function value should be positive away from origin."""
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction

        func = BucheRastriginFunction(dimension=5)
        rng = np.random.default_rng(42)
        x = rng.uniform(-3.0, 3.0, size=5)
        assert func.evaluate(x) > 0.0

    def test_asymmetry(self):
        """The function should NOT be symmetric due to per-element scaling."""
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction

        func = BucheRastriginFunction(dimension=4)
        x = np.array([1.0, 1.0, 1.0, 1.0])
        f_pos = func.evaluate(x)
        f_neg = func.evaluate(-x)
        # Due to asymmetric scaling, f(x) != f(-x)
        assert f_pos != pytest.approx(f_neg, rel=1e-6)

    def test_dimension_1(self):
        """Should work with dimension 1."""
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction

        func = BucheRastriginFunction(dimension=1)
        np.testing.assert_allclose(func.evaluate(np.zeros(1)), 0.0, atol=1e-12)

    def test_at_integer_points(self):
        """At integer points, cos(2*pi*x_i) = 1, so the cosine term vanishes partially."""
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction

        func = BucheRastriginFunction(dimension=2)
        # x = [1, 1]: cos(2*pi) = 1, so each term is s_i*(1 - 10*1) + 10 = s_i*(-9) + 10
        # This depends on s_i which depends on sign and index
        result = func.evaluate(np.array([1.0, 1.0]))
        assert np.isfinite(result)

    def test_evaluate_batch(self):
        """Batch evaluation should match individual evaluations."""
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction

        func = BucheRastriginFunction(dimension=3)
        X = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, -1.0, 0.5],
                [-2.0, 3.0, -1.0],
            ]
        )
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(X[i]) for i in range(3)])
        np.testing.assert_allclose(results, expected, rtol=1e-12)

    def test_global_minimum(self):
        """Global minimum should be at origin with value 0."""
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction

        func = BucheRastriginFunction(dimension=5)
        point, value = func.get_global_minimum()
        np.testing.assert_array_equal(point, np.zeros(5))
        assert value == 0.0

    def test_registry_aliases(self):
        from pyMOFL.registry import get

        cls = get("buche_rastrigin")
        assert cls is not None

    def test_bounds(self):
        """Default bounds should be [-5, 5]."""
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction

        func = BucheRastriginFunction(dimension=3)
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(3, -5.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(3, 5.0))
