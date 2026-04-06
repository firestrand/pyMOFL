"""
Tests for LunacekBiRastriginFunction.

Validates mathematical correctness, bounds handling, batch evaluation,
and structural properties of the Lunacek Bi-Rastrigin benchmark function.

f(x) = min(sum((x_i - mu0)^2), d*D + s*sum((x_i - mu1)^2)) + 10*(D - sum(cos(2*pi*(x_i - mu0))))
mu0 = 2.5, d = 1, s = 1 - 1/(2*sqrt(D+20) - 8.2), mu1 = -sqrt((mu0^2 - d) / s)

The Rastrigin cosine is centered at mu0 (per CEC source code), so f(mu0) = 0.
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.lunacek import LunacekBiRastriginFunction


class TestLunacekBiRastriginFunction:
    """Tests for LunacekBiRastriginFunction."""

    def test_initialization(self):
        """Test proper initialization with default bounds."""
        func = LunacekBiRastriginFunction(dimension=10)
        assert func.dimension == 10
        np.testing.assert_array_equal(func.initialization_bounds.low, np.full(10, -100.0))
        np.testing.assert_array_equal(func.initialization_bounds.high, np.full(10, 100.0))
        np.testing.assert_array_equal(func.operational_bounds.low, np.full(10, -100.0))
        np.testing.assert_array_equal(func.operational_bounds.high, np.full(10, 100.0))

    def test_initialization_various_dimensions(self):
        """Test initialization with various dimensions."""
        for dim in [2, 5, 10, 30]:
            func = LunacekBiRastriginFunction(dimension=dim)
            assert func.dimension == dim
            assert len(func.initialization_bounds.low) == dim

    def test_parameters_computed_correctly(self):
        """Test that dimension-dependent parameters are computed correctly."""
        dim = 10
        func = LunacekBiRastriginFunction(dimension=dim)

        assert func._mu0 == 2.5
        assert func._d == 1.0

        expected_s = 1.0 - 1.0 / (2.0 * np.sqrt(dim + 20.0) - 8.2)
        assert func._s == pytest.approx(expected_s)

        expected_mu1 = -np.sqrt((2.5**2 - 1.0) / expected_s)
        assert func._mu1 == pytest.approx(expected_mu1)

    def test_parameters_vary_with_dimension(self):
        """Test that s and mu1 change with dimension."""
        func5 = LunacekBiRastriginFunction(dimension=5)
        func30 = LunacekBiRastriginFunction(dimension=30)

        # s should differ for different dimensions
        assert func5._s != pytest.approx(func30._s)
        # mu1 also depends on s
        assert func5._mu1 != pytest.approx(func30._mu1)

    def test_global_minimum_value(self):
        """Test that f(mu0) = 0 (global minimum).

        At x = mu0: sum1 = 0 (basin center), cos(2π(x-μ₀)) = cos(0) = 1,
        so rastrigin = 10*(D - D) = 0. Therefore f(μ₀) = 0.
        """
        for dim in [2, 5, 10]:
            func = LunacekBiRastriginFunction(dimension=dim)
            x = np.full(dim, 2.5)
            result = func.evaluate(x)
            assert result == pytest.approx(0.0, abs=1e-10), (
                f"Expected f*=0 at mu0 for dim={dim}, got {result}"
            )

    def test_get_global_minimum(self):
        """Test get_global_minimum method returns correct point and value."""
        for dim in [2, 5, 10, 30]:
            func = LunacekBiRastriginFunction(dimension=dim)
            min_point, min_value = func.get_global_minimum()
            np.testing.assert_array_equal(min_point, np.full(dim, 2.5))
            assert min_value == 0.0

    def test_get_global_minimum_consistency(self):
        """Test that evaluate at get_global_minimum point matches declared value."""
        for dim in [2, 5, 10]:
            func = LunacekBiRastriginFunction(dimension=dim)
            min_point, min_value = func.get_global_minimum()
            result = func.evaluate(min_point)
            assert result == pytest.approx(min_value, abs=1e-10)

    def test_two_basin_structure(self):
        """Test the two-basin structure by evaluating near both basin centers."""
        dim = 10
        func = LunacekBiRastriginFunction(dimension=dim)

        # At mu0: f = 0 (global minimum)
        x_mu0 = np.full(dim, func._mu0)
        f_mu0 = func.evaluate(x_mu0)
        assert f_mu0 == pytest.approx(0.0, abs=1e-10)

        # At mu1: sum2 = d*D (basin 2 center), rastrigin > 0 in general
        x_mu1 = np.full(dim, func._mu1)
        f_mu1 = func.evaluate(x_mu1)
        assert np.isfinite(f_mu1)
        # Basin 2 is higher than basin 1 optimum
        assert f_mu1 > f_mu0

    def test_known_value_at_origin(self):
        """Test function value at origin."""
        dim = 5
        func = LunacekBiRastriginFunction(dimension=dim)
        x = np.zeros(dim)

        sum1 = np.sum((x - func._mu0) ** 2)  # D * 2.5^2 = D * 6.25
        sum2 = func._d * dim + func._s * np.sum((x - func._mu1) ** 2)
        rastrigin = 10.0 * (dim - np.sum(np.cos(2.0 * np.pi * (x - func._mu0))))
        expected = min(sum1, sum2) + rastrigin

        result = func.evaluate(x)
        assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"

    def test_rastrigin_modulation(self):
        """Test that the Rastrigin term adds multimodal structure.

        At x = mu0 + integer offsets, cos(2π(x-mu0)) = cos(2π*k) = 1, rastrigin = 0.
        At x = mu0 + half-integer offsets, cos(2π(x-mu0)) = cos(π) = -1, rastrigin = 20*D.
        """
        dim = 5
        func = LunacekBiRastriginFunction(dimension=dim)

        # mu0 + 1 (integer offset from mu0): rastrigin = 0
        x_int = np.full(dim, func._mu0 + 1.0)  # = 3.5
        sum1 = np.sum((x_int - func._mu0) ** 2)
        f_int = func.evaluate(x_int)
        # rastrigin is 0 at integer offsets from mu0
        assert f_int == pytest.approx(sum1, abs=1e-10)

        # mu0 + 0.5 (half-integer offset from mu0): rastrigin = 20*D
        x_half = np.full(dim, func._mu0 + 0.5)  # = 3.0
        f_half = func.evaluate(x_half)
        # cos(2π*0.5) = cos(π) = -1, rastrigin = 10*(D+D) = 20D
        sum1_half = np.sum((x_half - func._mu0) ** 2)
        expected_half = sum1_half + 20.0 * dim
        assert f_half == pytest.approx(expected_half, abs=1e-10)

    def test_evaluate_batch(self):
        """Test that batch evaluation matches individual evaluation."""
        func = LunacekBiRastriginFunction(dimension=5)
        rng = np.random.default_rng(42)
        X = rng.uniform(-5, 5, size=(10, 5))
        batch_results = func.evaluate_batch(X)
        individual_results = np.array([func.evaluate(x) for x in X])
        np.testing.assert_array_almost_equal(batch_results, individual_results)

    def test_batch_shape(self):
        """Test that batch output has correct shape."""
        func = LunacekBiRastriginFunction(dimension=5)
        X = np.zeros((7, 5))
        results = func.evaluate_batch(X)
        assert results.shape == (7,), f"Expected shape (7,), got {results.shape}"

    def test_dimension_validation(self):
        """Test that wrong input shape raises an error."""
        func = LunacekBiRastriginFunction(dimension=5)
        with pytest.raises(ValueError):
            func.evaluate(np.array([1.0, 2.0, 3.0]))  # Wrong dimension

        with pytest.raises(ValueError):
            func.evaluate_batch(np.array([[1.0, 2.0, 3.0]]))  # Wrong dimension

    def test_function_components_decomposition(self):
        """Test the three components of the function independently."""
        dim = 5
        func = LunacekBiRastriginFunction(dimension=dim)
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, size=dim)

        sum1 = np.sum((x - func._mu0) ** 2)
        sum2 = func._d * dim + func._s * np.sum((x - func._mu1) ** 2)
        rastrigin = 10.0 * (dim - np.sum(np.cos(2.0 * np.pi * (x - func._mu0))))
        expected = min(sum1, sum2) + rastrigin

        result = func.evaluate(x)
        assert abs(result - expected) < 1e-10

    def test_min_selects_smaller_basin(self):
        """Test that the min() operation correctly selects the smaller basin."""
        dim = 5
        func = LunacekBiRastriginFunction(dimension=dim)

        # At mu0: sum1 = 0, sum2 > 0, min selects sum1 = 0
        x = np.full(dim, func._mu0)
        sum1 = np.sum((x - func._mu0) ** 2)
        sum2 = func._d * dim + func._s * np.sum((x - func._mu1) ** 2)
        assert sum1 == pytest.approx(0.0)
        assert sum2 > 0.0

        # At mu1: sum2 = d*D, sum1 = D*(mu1-mu0)^2
        x = np.full(dim, func._mu1)
        sum1 = np.sum((x - func._mu0) ** 2)
        sum2 = func._d * dim + func._s * np.sum((x - func._mu1) ** 2)
        assert sum2 == pytest.approx(func._d * dim)

    def test_non_negative(self):
        """Test that function values are non-negative near the optimum."""
        func = LunacekBiRastriginFunction(dimension=5)
        rng = np.random.default_rng(42)
        X = rng.uniform(1.5, 3.5, size=(20, 5))
        for x in X:
            assert func.evaluate(x) >= -1e-10

    def test_registry_names(self):
        """Test that all registry names work."""
        from pyMOFL.registry import get

        func1 = get("lunacek_bi_rastrigin")(dimension=3)
        func2 = get("LunacekBiRastrigin")(dimension=3)
        assert isinstance(func1, LunacekBiRastriginFunction)
        assert isinstance(func2, LunacekBiRastriginFunction)

        x = np.array([1.0, 2.0, 3.0])
        assert func1.evaluate(x) == func2.evaluate(x)
