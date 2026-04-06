"""
Tests for Gallagher's Gaussian Peaks function (BBOB f21/f22 base).

f(x) = 10 - max_i(w_i * exp(-1/(2D) * (x - y_i)^T C_i (x - y_i)))

Parametrized by n_peaks (101 for BBOB f21, 21 for BBOB f22).
"""

import numpy as np
import pytest

from tests.utils.benchmark_validation import BenchmarkValidator


class TestGallagherPeaks101:
    """Tests for GallagherPeaksFunction with 101 peaks (BBOB f21)."""

    def test_contract_dim2(self):
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func = GallagherPeaksFunction(dimension=2, n_peaks=101, seed=42)
        BenchmarkValidator.assert_contract(func)

    def test_contract_multiple_dimensions(self):
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        for dim in [2, 5, 10]:
            func = GallagherPeaksFunction(dimension=dim, n_peaks=101, seed=42)
            BenchmarkValidator.assert_contract(func)

    def test_evaluate_at_global_optimum(self):
        """At the global optimum (peak 0), the function value should be 0."""
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func = GallagherPeaksFunction(dimension=2, n_peaks=101, seed=42)
        point, value = func.get_global_minimum()
        np.testing.assert_allclose(func.evaluate(point), value, atol=1e-8)
        np.testing.assert_allclose(value, 0.0, atol=1e-8)

    def test_value_far_from_peaks(self):
        """Far from all peaks, the function value should be close to 10."""
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func = GallagherPeaksFunction(dimension=2, n_peaks=101, seed=42)
        # At a point very far from all peaks (if peaks are in [-5,5])
        x_far = np.full(2, 1000.0)
        result = func.evaluate(x_far)
        np.testing.assert_allclose(result, 10.0, atol=0.1)


class TestGallagherPeaks21:
    """Tests for GallagherPeaksFunction with 21 peaks (BBOB f22)."""

    def test_contract_dim2(self):
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func = GallagherPeaksFunction(dimension=2, n_peaks=21, seed=42)
        BenchmarkValidator.assert_contract(func)

    def test_evaluate_at_global_optimum(self):
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func = GallagherPeaksFunction(dimension=3, n_peaks=21, seed=42)
        point, value = func.get_global_minimum()
        np.testing.assert_allclose(func.evaluate(point), value, atol=1e-8)

    def test_fewer_peaks_than_101(self):
        """21 peaks should have a simpler landscape than 101 peaks."""
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func_21 = GallagherPeaksFunction(dimension=2, n_peaks=21, seed=42)
        func_101 = GallagherPeaksFunction(dimension=2, n_peaks=101, seed=42)
        assert func_21._n_peaks == 21
        assert func_101._n_peaks == 101


class TestGallagherPeaksReproducibility:
    """Tests for seeded reproducibility."""

    def test_same_seed_same_results(self):
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func1 = GallagherPeaksFunction(dimension=3, n_peaks=101, seed=123)
        func2 = GallagherPeaksFunction(dimension=3, n_peaks=101, seed=123)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(func1.evaluate(x), func2.evaluate(x), rtol=1e-14)

    def test_different_seed_different_results(self):
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func1 = GallagherPeaksFunction(dimension=3, n_peaks=101, seed=123)
        func2 = GallagherPeaksFunction(dimension=3, n_peaks=101, seed=456)
        x = np.array([1.0, 2.0, 3.0])
        assert not np.isclose(func1.evaluate(x), func2.evaluate(x))

    def test_different_optima_different_seeds(self):
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func1 = GallagherPeaksFunction(dimension=2, n_peaks=101, seed=1)
        func2 = GallagherPeaksFunction(dimension=2, n_peaks=101, seed=2)
        p1, _ = func1.get_global_minimum()
        p2, _ = func2.get_global_minimum()
        assert not np.allclose(p1, p2)


class TestGallagherPeaksEdgeCases:
    """Tests for edge-case n_peaks and dimension values."""

    def test_n_peaks_2(self):
        """n_peaks=2 exercises the special-case branch."""
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func = GallagherPeaksFunction(dimension=2, n_peaks=2, seed=42)
        BenchmarkValidator.assert_contract(func)
        assert func._weights[0] == 10.0
        assert func._weights[1] == 1.1

    def test_n_peaks_1(self):
        """n_peaks=1 has only the global peak."""
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func = GallagherPeaksFunction(dimension=2, n_peaks=1, seed=42)
        point, value = func.get_global_minimum()
        np.testing.assert_allclose(func.evaluate(point), value, atol=1e-8)

    def test_n_peaks_zero_raises(self):
        """n_peaks=0 must raise ValueError."""
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        with pytest.raises(ValueError, match="n_peaks"):
            GallagherPeaksFunction(dimension=2, n_peaks=0, seed=42)

    def test_n_peaks_negative_raises(self):
        """Negative n_peaks must raise ValueError."""
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        with pytest.raises(ValueError, match="n_peaks"):
            GallagherPeaksFunction(dimension=2, n_peaks=-5, seed=42)

    def test_dimension_1(self):
        """dimension=1 exercises the diag special case."""
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func = GallagherPeaksFunction(dimension=1, n_peaks=5, seed=42)
        BenchmarkValidator.assert_contract(func)


class TestGallagherPeaksBatch:
    """Tests for batch evaluation."""

    def test_evaluate_batch(self):
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction

        func = GallagherPeaksFunction(dimension=2, n_peaks=101, seed=42)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-2.0, 3.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(X[i]) for i in range(3)])
        np.testing.assert_allclose(results, expected, rtol=1e-12)


class TestGallagherPeaksRegistry:
    """Tests for registry aliases."""

    def test_registry_aliases(self):
        from pyMOFL.registry import get

        cls = get("gallagher_peaks")
        assert cls is not None
