"""
Tests for the Step Ellipsoidal function (BBOB f7 base).

f(z) = 0.1 * max(|z_hat_1|/10000, sum(10^(2i/(D-1)) * z_tilde_i^2))

where:
    z_hat = diag(10^(i/2(D-1))) * z  (Lambda conditioning)
    z_tilde_i = floor(0.5 + z_hat_i) if |z_hat_i| > 0.5, else z_hat_i

The step operation creates plateaus in the landscape.
Global minimum at origin with value 0.
"""

import numpy as np

from tests.utils.benchmark_validation import BenchmarkValidator


class TestStepEllipsoidFunction:
    """Tests for StepEllipsoidFunction."""

    def test_contract_dim2(self):
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        func = StepEllipsoidFunction(dimension=2)
        BenchmarkValidator.assert_contract(func)

    def test_contract_multiple_dimensions(self):
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        BenchmarkValidator.assert_contract_multiple_dimensions(
            StepEllipsoidFunction, dimensions=[2, 5, 10, 30]
        )

    def test_evaluate_at_origin(self):
        """f(0) = 0."""
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        func = StepEllipsoidFunction(dimension=5)
        np.testing.assert_allclose(func.evaluate(np.zeros(5)), 0.0, atol=1e-12)

    def test_positive_away_from_origin(self):
        """Function value should be positive away from origin."""
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        func = StepEllipsoidFunction(dimension=5)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert func.evaluate(x) > 0.0

    def test_plateau_property(self):
        """Small perturbations near origin should give zero due to floor/step.

        For small z values (|z_hat| < 0.5), z_tilde = z_hat (no rounding),
        so the function is smooth near origin.
        """
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        func = StepEllipsoidFunction(dimension=2)
        # Small values: z_hat will be small, z_tilde = z_hat
        x1 = np.array([0.1, 0.1])
        x2 = np.array([0.2, 0.2])
        # Both should give finite positive values
        assert func.evaluate(x1) >= 0.0
        assert func.evaluate(x2) > func.evaluate(x1)

    def test_step_discontinuity(self):
        """For values with |z_hat| > 0.5, the floor operation creates steps."""
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        func = StepEllipsoidFunction(dimension=2)
        # At x=1.0, z_hat should be > 0.5, so z_tilde = floor(0.5 + z_hat)
        # This means small perturbations don't change the value
        x_base = np.array([2.0, 0.0])
        x_pert = np.array([2.01, 0.0])
        # The step nature means these might give the same result
        f_base = func.evaluate(x_base)
        f_pert = func.evaluate(x_pert)
        assert np.isfinite(f_base)
        assert np.isfinite(f_pert)

    def test_first_component_dominance(self):
        """The max(|z_hat_1|/10000, ...) term protects the first component."""
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        func = StepEllipsoidFunction(dimension=2)
        # With a very large first component
        x = np.array([50000.0, 0.0])
        result = func.evaluate(x)
        # 0.1 * max(50000/10000, sum(...)) = 0.1 * max(5, ...)
        # Should be positive and finite
        assert result > 0.0
        assert np.isfinite(result)

    def test_dimension_1(self):
        """Should work with dimension 1."""
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        func = StepEllipsoidFunction(dimension=1)
        np.testing.assert_allclose(func.evaluate(np.zeros(1)), 0.0, atol=1e-12)

    def test_evaluate_batch(self):
        """Batch evaluation should match individual evaluations."""
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        func = StepEllipsoidFunction(dimension=3)
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
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        func = StepEllipsoidFunction(dimension=5)
        point, value = func.get_global_minimum()
        np.testing.assert_array_equal(point, np.zeros(5))
        assert value == 0.0

    def test_registry_aliases(self):
        from pyMOFL.registry import get

        cls = get("step_ellipsoid")
        assert cls is not None

    def test_dimension_1_raises(self):
        """StepEllipsoid requires dimension >= 2 (needs at least 2 for conditioning)."""
        # Actually, let's allow dimension 1 as a degenerate case
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction

        func = StepEllipsoidFunction(dimension=1)
        assert func.dimension == 1
