"""
Tests for the Linear Slope function (BBOB f5).

f(x) = sum(5 * |s_i| - s_i * x_i)
where s_i = sign_i * 10^((i-1)/(D-1))

Global minimum is at the domain boundary x_i = sign_i * 5.
"""

import numpy as np
import pytest

from tests.utils.benchmark_validation import BenchmarkValidator


class TestLinearSlopeFunction:
    """Tests for LinearSlopeFunction."""

    def test_contract_dim2(self):
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction

        func = LinearSlopeFunction(dimension=2)
        BenchmarkValidator.assert_contract(func)

    def test_contract_multiple_dimensions(self):
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction

        BenchmarkValidator.assert_contract_multiple_dimensions(
            LinearSlopeFunction, dimensions=[2, 5, 10, 30]
        )

    def test_evaluate_at_origin(self):
        """At the origin, f(0) = sum(5 * |s_i|) since -s_i * 0 = 0."""
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction

        func = LinearSlopeFunction(dimension=2)
        result = func.evaluate(np.zeros(2))
        # s_0 = 1*sqrt(10)^0 = 1, s_1 = 1*sqrt(10)^1 = sqrt(10) (default sign_vector = ones)
        # f(0) = 5*1 + 5*sqrt(10) ≈ 20.8114
        expected = 5.0 * (1.0 + np.sqrt(10.0))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_evaluate_at_optimum(self):
        """At the optimum x_i = sign_i * 5, f = 0."""
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction

        func = LinearSlopeFunction(dimension=3)
        point, value = func.get_global_minimum()
        np.testing.assert_allclose(func.evaluate(point), value, atol=1e-12)
        np.testing.assert_allclose(value, 0.0, atol=1e-12)

    def test_global_minimum_at_boundary(self):
        """The optimum should be at the domain boundary (+/-5 per dimension)."""
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction

        func = LinearSlopeFunction(dimension=4)
        point, _ = func.get_global_minimum()
        np.testing.assert_allclose(np.abs(point), 5.0)

    def test_custom_sign_vector(self):
        """Test with a custom sign vector (mixed +/-)."""
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction

        sign_vec = np.array([1.0, -1.0, 1.0])
        func = LinearSlopeFunction(dimension=3, sign_vector=sign_vec)
        point, value = func.get_global_minimum()
        # Optimum should be at [5, -5, 5]
        np.testing.assert_allclose(point, [5.0, -5.0, 5.0])
        np.testing.assert_allclose(value, 0.0, atol=1e-12)
        np.testing.assert_allclose(func.evaluate(point), 0.0, atol=1e-12)

    def test_slope_scaling(self):
        """Verify slope coefficients scale as sqrt(10)^(i/(D-1))."""
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction

        # In 2D: s_0 = sqrt(10)^0 = 1, s_1 = sqrt(10)^1 = sqrt(10)
        func = LinearSlopeFunction(dimension=2)
        s10 = np.sqrt(10.0)
        # f([1, 0]) = (5*1 - 1*1) + (5*s10 - s10*0) = 4 + 5*s10
        np.testing.assert_allclose(func.evaluate(np.array([1.0, 0.0])), 4.0 + 5.0 * s10, rtol=1e-10)

    def test_linearity(self):
        """The function is purely linear, so f(a*x) should scale linearly from f(0)."""
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction

        func = LinearSlopeFunction(dimension=3)
        x = np.array([1.0, 2.0, -1.0])
        f_x = func.evaluate(x)
        f_0 = func.evaluate(np.zeros(3))
        f_2x = func.evaluate(2.0 * x)
        # f(2x) - f(0) = 2 * (f(x) - f(0)) due to linearity
        np.testing.assert_allclose(f_2x - f_0, 2.0 * (f_x - f_0), rtol=1e-10)

    def test_evaluate_batch(self):
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction

        func = LinearSlopeFunction(dimension=2)
        X = np.array([[0.0, 0.0], [5.0, 5.0], [1.0, 1.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(X[i]) for i in range(3)])
        np.testing.assert_allclose(results, expected, rtol=1e-12)

    def test_dimension_1(self):
        """Test with dimension=1 (special case: exponent = 0)."""
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction

        func = LinearSlopeFunction(dimension=1)
        # s_1 = 1*10^0 = 1, f(0) = 5*1 = 5
        np.testing.assert_allclose(func.evaluate(np.zeros(1)), 5.0, rtol=1e-10)
        point, value = func.get_global_minimum()
        np.testing.assert_allclose(func.evaluate(point), value, atol=1e-12)

    def test_sign_vector_wrong_shape_raises(self):
        """sign_vector with wrong shape must raise ValueError."""
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction

        with pytest.raises(ValueError, match="sign_vector"):
            LinearSlopeFunction(dimension=3, sign_vector=np.array([1.0, -1.0]))

    def test_sign_vector_with_zero_raises(self):
        """sign_vector containing zero (np.sign(0)=0) must raise ValueError."""
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction

        with pytest.raises(ValueError, match="sign_vector"):
            LinearSlopeFunction(dimension=3, sign_vector=np.array([1.0, 0.0, -1.0]))

    def test_registry_aliases(self):
        from pyMOFL.registry import get

        cls = get("linear_slope")
        assert cls is not None
