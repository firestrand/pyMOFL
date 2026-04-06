"""
Tests for Schaffer function variants (N.1, N.2, N.4).

References:
    Schaffer, J.D., et al. (1989). "A study of control parameters for genetic algorithms."
    Jamil, M., & Yang, X.S. (2013). arXiv:1308.4008
"""

import numpy as np
import pytest

from pyMOFL.functions.benchmark.schaffer import (
    Schaffer1Function,
    Schaffer2Function,
    Schaffer4Function,
)


# ---------------------------------------------------------------------------
# Schaffer N.1: f(x,y) = 0.5 + (sin²(sqrt(x²+y²)) - 0.5) / (1 + 0.001(x²+y²))²
# ---------------------------------------------------------------------------
class TestSchaffer1:
    """Tests for Schaffer N.1 — concentric ring landscape."""

    def test_initialization(self):
        func = Schaffer1Function()
        assert func.dimension == 2

    def test_dimension_must_be_2(self):
        with pytest.raises(ValueError, match="dimension=2"):
            Schaffer1Function(dimension=3)

    def test_global_minimum(self):
        func = Schaffer1Function()
        assert np.isclose(func.evaluate(np.array([0.0, 0.0])), 0.0, atol=1e-12)

    def test_get_global_minimum(self):
        func = Schaffer1Function()
        point, value = func.get_global_minimum()
        np.testing.assert_array_equal(point, [0.0, 0.0])
        assert value == 0.0

    def test_circular_symmetry(self):
        """f depends only on r² = x²+y², so rotations around origin give same value."""
        func = Schaffer1Function()
        v1 = func.evaluate(np.array([3.0, 4.0]))  # r=5
        v2 = func.evaluate(np.array([5.0, 0.0]))  # r=5
        v3 = func.evaluate(np.array([0.0, 5.0]))  # r=5
        assert np.isclose(v1, v2, atol=1e-10)
        assert np.isclose(v1, v3, atol=1e-10)

    def test_value_range(self):
        """Values should be in [0, 1)."""
        func = Schaffer1Function()
        rng = np.random.default_rng(42)
        for _ in range(100):
            val = func.evaluate(rng.uniform(-100, 100, 2))
            assert 0.0 <= val < 1.0

    def test_known_value(self):
        """Hand-computed: f(1,0) = 0.5 + (sin²(1) - 0.5)/(1.001)²."""
        func = Schaffer1Function()
        expected = 0.5 + (np.sin(1.0) ** 2 - 0.5) / (1.001) ** 2
        assert np.isclose(func.evaluate(np.array([1.0, 0.0])), expected, atol=1e-10)

    def test_batch_evaluation(self):
        func = Schaffer1Function()
        X = np.array([[0, 0], [1, 0], [3, 4]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-12)

    def test_batch_shape(self):
        func = Schaffer1Function()
        assert func.evaluate_batch(np.zeros((5, 2))).shape == (5,)


# ---------------------------------------------------------------------------
# Schaffer N.2: f(x,y) = 0.5 + (sin²(x²-y²) - 0.5) / (1 + 0.001(x²+y²))²
# ---------------------------------------------------------------------------
class TestSchaffer2:
    """Tests for Schaffer N.2 — deceptive oscillation variant."""

    def test_initialization(self):
        func = Schaffer2Function()
        assert func.dimension == 2

    def test_dimension_must_be_2(self):
        with pytest.raises(ValueError, match="dimension=2"):
            Schaffer2Function(dimension=5)

    def test_global_minimum(self):
        func = Schaffer2Function()
        assert np.isclose(func.evaluate(np.array([0.0, 0.0])), 0.0, atol=1e-12)

    def test_get_global_minimum(self):
        func = Schaffer2Function()
        point, value = func.get_global_minimum()
        np.testing.assert_array_equal(point, [0.0, 0.0])
        assert value == 0.0

    def test_not_circularly_symmetric(self):
        """Unlike N.1, this uses x²-y² inside sin, so it's NOT circularly symmetric."""
        func = Schaffer2Function()
        # Swap symmetry: sin²(x²-y²) = sin²(y²-x²), so f(a,b) = f(b,a)
        v1 = func.evaluate(np.array([3.0, 4.0]))
        v2 = func.evaluate(np.array([4.0, 3.0]))
        assert np.isclose(v1, v2)
        # NOT circularly symmetric: same r but different x²-y²
        v7 = func.evaluate(np.array([3.0, 0.0]))  # r²=9, sin²(9)
        v9 = func.evaluate(np.array([2.12, 2.12]))  # r²≈8.99, x²-y²≈0
        assert not np.isclose(v7, v9)  # sin²(9) vs sin²(0) — different

    def test_value_range(self):
        func = Schaffer2Function()
        rng = np.random.default_rng(42)
        for _ in range(100):
            val = func.evaluate(rng.uniform(-100, 100, 2))
            assert 0.0 <= val < 1.0

    def test_known_value(self):
        func = Schaffer2Function()
        x, y = 1.0, 0.0
        r2 = x**2 + y**2
        expected = 0.5 + (np.sin(x**2 - y**2) ** 2 - 0.5) / (1 + 0.001 * r2) ** 2
        assert np.isclose(func.evaluate(np.array([x, y])), expected, atol=1e-12)

    def test_batch_evaluation(self):
        func = Schaffer2Function()
        X = np.array([[0, 0], [1, 0], [2, 1]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-12)

    def test_batch_shape(self):
        func = Schaffer2Function()
        assert func.evaluate_batch(np.zeros((4, 2))).shape == (4,)


# ---------------------------------------------------------------------------
# Schaffer N.4: f(x,y) = 0.5 + (cos²(sin(|x²-y²|)) - 0.5) / (1 + 0.001(x²+y²))²
# ---------------------------------------------------------------------------
class TestSchaffer4:
    """Tests for Schaffer N.4 — cos-of-sin variant, hardest in the family."""

    def test_initialization(self):
        func = Schaffer4Function()
        assert func.dimension == 2

    def test_dimension_must_be_2(self):
        with pytest.raises(ValueError, match="dimension=2"):
            Schaffer4Function(dimension=4)

    def test_value_at_origin(self):
        """f(0,0) = 0.5 + (cos²(sin(0)) - 0.5) / 1 = 0.5 + (1 - 0.5) = 1.0."""
        func = Schaffer4Function()
        assert np.isclose(func.evaluate(np.array([0.0, 0.0])), 1.0, atol=1e-12)

    def test_global_minimum_value(self):
        """Global minimum ≈ 0.29257."""
        func = Schaffer4Function()
        point, value = func.get_global_minimum()
        actual = func.evaluate(point)
        assert np.isclose(actual, value, atol=1e-4)
        assert np.isclose(value, 0.29258, atol=0.001)

    def test_value_range(self):
        """Values should be in [~0.29, 1]."""
        func = Schaffer4Function()
        rng = np.random.default_rng(42)
        for _ in range(100):
            val = func.evaluate(rng.uniform(-100, 100, 2))
            assert 0.0 <= val <= 1.0 + 1e-10

    def test_known_value(self):
        func = Schaffer4Function()
        x, y = 2.0, 1.0
        r2 = x**2 + y**2
        expected = 0.5 + (np.cos(np.sin(np.abs(x**2 - y**2))) ** 2 - 0.5) / (1 + 0.001 * r2) ** 2
        assert np.isclose(func.evaluate(np.array([x, y])), expected, atol=1e-12)

    def test_multimodal(self):
        """Many local minima exist."""
        func = Schaffer4Function()
        rng = np.random.default_rng(42)
        vals = sorted(func.evaluate(rng.uniform(-100, 100, 2)) for _ in range(500))
        # There should be distinct value clusters
        assert vals[-1] - vals[0] > 0.5

    def test_batch_evaluation(self):
        func = Schaffer4Function()
        X = np.array([[0, 0], [2, 1], [-1.256, -0.075]])
        batch = func.evaluate_batch(X)
        individual = np.array([func.evaluate(x) for x in X])
        np.testing.assert_allclose(batch, individual, atol=1e-10)

    def test_batch_shape(self):
        func = Schaffer4Function()
        assert func.evaluate_batch(np.zeros((3, 2))).shape == (3,)
