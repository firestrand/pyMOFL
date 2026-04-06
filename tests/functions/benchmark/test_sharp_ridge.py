"""
Tests for the Sharp Ridge function (BBOB f13 base).

f(x) = x_1^2 + 100 * sqrt(sum(x_i^2, i=2..D))

Global minimum at the origin with value 0.
The ridge along x_2=...=x_D=0 is non-differentiable.
"""

import numpy as np

from tests.utils.benchmark_validation import BenchmarkValidator


class TestSharpRidgeFunction:
    """Tests for SharpRidgeFunction."""

    def test_contract_dim2(self):
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        func = SharpRidgeFunction(dimension=2)
        BenchmarkValidator.assert_contract(func)

    def test_contract_multiple_dimensions(self):
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        BenchmarkValidator.assert_contract_multiple_dimensions(
            SharpRidgeFunction, dimensions=[2, 5, 10, 30]
        )

    def test_evaluate_at_origin(self):
        """f(0) = 0."""
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        func = SharpRidgeFunction(dimension=5)
        np.testing.assert_allclose(func.evaluate(np.zeros(5)), 0.0, atol=1e-12)

    def test_along_ridge(self):
        """Along the ridge (x_2=...=x_D=0), f(x) = x_1^2."""
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        func = SharpRidgeFunction(dimension=5)
        x = np.array([3.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(func.evaluate(x), 9.0, rtol=1e-12)

    def test_off_ridge(self):
        """Off the ridge, the sqrt term dominates."""
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        func = SharpRidgeFunction(dimension=3)
        x = np.array([0.0, 3.0, 4.0])
        # f = 0 + 100 * sqrt(9 + 16) = 100 * 5 = 500
        np.testing.assert_allclose(func.evaluate(x), 500.0, rtol=1e-12)

    def test_combined(self):
        """Test with both ridge and off-ridge components."""
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        func = SharpRidgeFunction(dimension=3)
        x = np.array([2.0, 3.0, 4.0])
        # f = 4 + 100 * sqrt(9 + 16) = 4 + 500 = 504
        np.testing.assert_allclose(func.evaluate(x), 504.0, rtol=1e-12)

    def test_symmetry_off_ridge(self):
        """The off-ridge part is symmetric in x_2..x_D."""
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        func = SharpRidgeFunction(dimension=4)
        x1 = np.array([1.0, 2.0, 0.0, 0.0])
        x2 = np.array([1.0, 0.0, 2.0, 0.0])
        x3 = np.array([1.0, 0.0, 0.0, 2.0])
        np.testing.assert_allclose(func.evaluate(x1), func.evaluate(x2), rtol=1e-12)
        np.testing.assert_allclose(func.evaluate(x2), func.evaluate(x3), rtol=1e-12)

    def test_evaluate_batch(self):
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        func = SharpRidgeFunction(dimension=3)
        X = np.array([[0.0, 0.0, 0.0], [2.0, 3.0, 4.0], [1.0, 0.0, 0.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(X[i]) for i in range(3)])
        np.testing.assert_allclose(results, expected, rtol=1e-12)

    def test_non_differentiable_at_ridge(self):
        """The sqrt term creates a V-shape (non-differentiable) at the ridge.

        As epsilon -> 0, the one-sided finite-difference slopes in the off-ridge
        direction should remain bounded away from zero (linear, not quadratic),
        confirming the sharp ridge characteristic.
        """
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        func = SharpRidgeFunction(dimension=3)
        eps = 1e-8
        # Point on the ridge
        x_ridge = np.array([1.0, 0.0, 0.0])
        f_ridge = func.evaluate(x_ridge)

        # Step off the ridge in +x_2 and -x_2
        f_plus = func.evaluate(x_ridge + np.array([0.0, eps, 0.0]))
        f_minus = func.evaluate(x_ridge + np.array([0.0, -eps, 0.0]))

        # Both sides increase equally (symmetric V-shape)
        np.testing.assert_allclose(f_plus, f_minus, rtol=1e-10)

        # The slope |df/dx_2| ~ 100 (from 100*sqrt(eps^2)/eps = 100),
        # not vanishing as it would for a smooth (quadratic) minimum
        slope = (f_plus - f_ridge) / eps
        np.testing.assert_allclose(slope, 100.0, rtol=1e-4)

    def test_ridge_linear_vs_quadratic_growth(self):
        """Off-ridge growth is linear (sqrt), not quadratic.

        For a smooth function, f(ridge + h*e_2) - f(ridge) ~ O(h^2).
        For the sharp ridge, it should be O(h) due to sqrt.
        """
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        func = SharpRidgeFunction(dimension=2)
        x_ridge = np.array([1.0, 0.0])
        f_ridge = func.evaluate(x_ridge)

        h1 = 0.01
        h2 = 0.02
        delta1 = func.evaluate(np.array([1.0, h1])) - f_ridge
        delta2 = func.evaluate(np.array([1.0, h2])) - f_ridge

        # Linear growth: delta2/delta1 ≈ h2/h1 = 2.0 (not 4.0 as quadratic would give)
        ratio = delta2 / delta1
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-10)

    def test_dimension_1_raises(self):
        """SharpRidge requires dimension >= 2."""
        import pytest

        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction

        with pytest.raises(ValueError, match="dimension >= 2"):
            SharpRidgeFunction(dimension=1)

    def test_registry_aliases(self):
        from pyMOFL.registry import get

        cls = get("sharp_ridge")
        assert cls is not None
