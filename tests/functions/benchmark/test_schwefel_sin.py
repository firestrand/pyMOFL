"""
Tests for the Schwefel x*sin(sqrt(|x|)) function (BBOB f20 base).

f(x) = 418.9829 * D - sum(x_i * sin(sqrt(|x_i|)))

Traditional Schwefel function with domain [-500, 500]^D.
Global minimum at x_i ≈ 420.9687... with value ≈ 0.
"""

import numpy as np

from tests.utils.benchmark_validation import BenchmarkValidator


class TestSchwefelSinFunction:
    """Tests for SchwefelSinFunction."""

    def test_contract_dim2(self):
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction

        func = SchwefelSinFunction(dimension=2)
        BenchmarkValidator.assert_contract(func, global_min_atol=1e-3)

    def test_contract_multiple_dimensions(self):
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction

        BenchmarkValidator.assert_contract_multiple_dimensions(
            SchwefelSinFunction, dimensions=[2, 5, 10], global_min_atol=1e-3
        )

    def test_evaluate_at_origin(self):
        """f(0) = 418.9829 * D since sin(0) = 0."""
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction

        func = SchwefelSinFunction(dimension=3)
        expected = 418.9829 * 3
        np.testing.assert_allclose(func.evaluate(np.zeros(3)), expected, rtol=1e-4)

    def test_evaluate_near_optimum(self):
        """At x_i ≈ 420.9687, the function should be near 0."""
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction

        func = SchwefelSinFunction(dimension=2)
        x_opt = np.full(2, 420.9687)
        result = func.evaluate(x_opt)
        assert abs(result) < 1.0  # Should be very close to 0

    def test_separability(self):
        """The function is separable: f(x) = sum of 1D contributions + constant."""
        from pyMOFL.functions.benchmark.schwefel_sin import (
            _SCHWEFEL_CONSTANT,
            SchwefelSinFunction,
        )

        func = SchwefelSinFunction(dimension=3)
        x = np.array([100.0, 200.0, 300.0])
        expected = _SCHWEFEL_CONSTANT * 3 - np.sum(x * np.sin(np.sqrt(np.abs(x))))
        np.testing.assert_allclose(func.evaluate(x), expected, rtol=1e-10)

    def test_symmetry_property(self):
        """f is not symmetric: f(x) != f(-x) in general."""
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction

        func = SchwefelSinFunction(dimension=2)
        x = np.array([100.0, 200.0])
        f_pos = func.evaluate(x)
        f_neg = func.evaluate(-x)
        # These should differ due to x*sin(sqrt(|x|))
        assert not np.isclose(f_pos, f_neg)

    def test_deceptive_structure(self):
        """The second-best local minimum should be far from the global minimum."""
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction

        func = SchwefelSinFunction(dimension=1)
        # The global minimum is near x=420.9687
        # A prominent local minimum is near x=-302.52
        f_global = func.evaluate(np.array([420.9687]))
        f_local = func.evaluate(np.array([-302.52]))
        assert f_global < f_local

    def test_evaluate_batch(self):
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction

        func = SchwefelSinFunction(dimension=2)
        X = np.array([[0.0, 0.0], [420.9687, 420.9687], [100.0, -200.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(X[i]) for i in range(3)])
        np.testing.assert_allclose(results, expected, rtol=1e-12)

    def test_registry_aliases(self):
        from pyMOFL.registry import get

        cls = get("schwefel_sin")
        assert cls is not None


class TestSchwefelSinBoundaryHandling:
    """Tests for CEC 2014 boundary handling mode."""

    def test_within_bounds_matches_standard(self):
        """For inputs within [-500, 500], boundary handling matches standard."""
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction

        func_std = SchwefelSinFunction(dimension=3)
        func_bh = SchwefelSinFunction(dimension=3, boundary_handling=True)
        x = np.array([100.0, -200.0, 300.0])
        np.testing.assert_allclose(func_bh.evaluate(x), func_std.evaluate(x), rtol=1e-12)

    def test_positive_boundary_clamping(self):
        """For z > 500, uses fmod clamping + penalty."""
        from pyMOFL.functions.benchmark.schwefel_sin import (
            _SCHWEFEL_CONSTANT,
            SchwefelSinFunction,
        )

        func = SchwefelSinFunction(dimension=1, boundary_handling=True)
        z = np.array([600.0])
        # fmod(600, 500) = 100, clamped = 500 - 100 = 400
        # contribution: -(400 * sin(sqrt(400)))
        # penalty: ((600 - 500) / 100)^2 / 1 = 1.0
        clamped = 400.0
        expected = -clamped * np.sin(np.sqrt(clamped)) + 1.0 + _SCHWEFEL_CONSTANT
        np.testing.assert_allclose(func.evaluate(z), expected, rtol=1e-12)

    def test_negative_boundary_clamping(self):
        """For z < -500, uses fmod clamping + penalty."""
        from pyMOFL.functions.benchmark.schwefel_sin import (
            _SCHWEFEL_CONSTANT,
            SchwefelSinFunction,
        )

        func = SchwefelSinFunction(dimension=1, boundary_handling=True)
        z = np.array([-600.0])
        # fmod(600, 500) = 100, clamped = 500 - 100 = 400
        # contribution: -(-500 + 100) * sin(sqrt(400)) = 400 * sin(sqrt(400))
        # penalty: ((-600 + 500) / 100)^2 / 1 = 1.0
        mod_val = 100.0
        clamped = 400.0
        expected = -(-500.0 + mod_val) * np.sin(np.sqrt(clamped)) + 1.0 + _SCHWEFEL_CONSTANT
        np.testing.assert_allclose(func.evaluate(z), expected, rtol=1e-12)

    def test_mixed_boundary_handling(self):
        """Test with mix of in-bounds and out-of-bounds components."""
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction

        func_bh = SchwefelSinFunction(dimension=3, boundary_handling=True)
        func_std = SchwefelSinFunction(dimension=3)
        # All within bounds — should match
        x_in = np.array([100.0, -200.0, 300.0])
        np.testing.assert_allclose(func_bh.evaluate(x_in), func_std.evaluate(x_in), rtol=1e-12)
        # With out-of-bounds — should differ
        x_out = np.array([600.0, -700.0, 300.0])
        assert func_bh.evaluate(x_out) != func_std.evaluate(x_out)

    def test_batch_boundary_handling(self):
        """Batch evaluation matches single evaluation with boundary handling."""
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction

        func = SchwefelSinFunction(dimension=2, boundary_handling=True)
        X = np.array([[100.0, 200.0], [600.0, -700.0], [-500.0, 500.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(X[i]) for i in range(3)])
        np.testing.assert_allclose(results, expected, rtol=1e-12)
