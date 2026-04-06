"""
Tests for the Attractive Sector function (BBOB f6 base).

Base formula: f(x) = sum((s_i * x_i)^2)^0.9
where s_i = 100 if x_i * x_opt_i > 0, else 1.

The T_osz transformation and Q*Lambda*R rotations are applied externally
via ComposedFunction in the BBOB suite config (Phase 3).
"""

import numpy as np
import pytest

from tests.utils.benchmark_validation import BenchmarkValidator


class TestAttractiveSectorFunction:
    """Tests for AttractiveSectorFunction."""

    def test_contract_dim2(self):
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction

        func = AttractiveSectorFunction(dimension=2)
        BenchmarkValidator.assert_contract(func)

    def test_contract_multiple_dimensions(self):
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction

        BenchmarkValidator.assert_contract_multiple_dimensions(
            AttractiveSectorFunction, dimensions=[2, 5, 10]
        )

    def test_evaluate_at_origin(self):
        """At origin, f(0) = 0."""
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction

        func = AttractiveSectorFunction(dimension=3)
        np.testing.assert_allclose(func.evaluate(np.zeros(3)), 0.0, atol=1e-12)

    def test_asymmetric_weighting_same_sign(self):
        """When x_i and x_opt_i have the same sign, s_i = 100."""
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction

        # Default x_opt = ones(D), so positive x_i gets s_i = 100
        func = AttractiveSectorFunction(dimension=2)
        x = np.array([1.0, 0.0])
        # s_1 = 100 (same sign as x_opt_1=1), s_2 = 1 (x_2=0, not > 0)
        # f = ((100*1)^2 + (1*0)^2)^0.9 = (10000)^0.9
        expected = 10000.0**0.9
        np.testing.assert_allclose(func.evaluate(x), expected, rtol=1e-10)

    def test_asymmetric_weighting_opposite_sign(self):
        """When x_i and x_opt_i have opposite signs, s_i = 1."""
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction

        # Default x_opt = ones(D), so negative x_i gets s_i = 1
        func = AttractiveSectorFunction(dimension=2)
        x = np.array([-1.0, 0.0])
        # s_1 = 1 (opposite sign), s_2 = 1 (x_2=0)
        # f = ((1*(-1))^2 + (1*0)^2)^0.9 = (1)^0.9 = 1
        expected = 1.0**0.9
        np.testing.assert_allclose(func.evaluate(x), expected, rtol=1e-10)

    def test_asymmetry_ratio(self):
        """The attractive sector should penalize opposite-sign directions much less."""
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction

        func = AttractiveSectorFunction(dimension=2)
        x_pos = np.array([1.0, 1.0])
        x_neg = np.array([-1.0, -1.0])
        # Same magnitude, but very different function values due to 100 vs 1 weighting
        f_pos = func.evaluate(x_pos)
        f_neg = func.evaluate(x_neg)
        assert f_pos > f_neg * 100  # 100^2 scaling squared

    def test_custom_x_opt(self):
        """Test with a custom reference direction."""
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction

        x_opt = np.array([-1.0, 1.0])
        func = AttractiveSectorFunction(dimension=2, x_opt=x_opt)
        # x = [-1, 1] should be in the "attractive" sector (same sign as x_opt)
        x = np.array([-1.0, 1.0])
        # s_1 = 100 ((-1)*(-1) > 0), s_2 = 100 ((1)*(1) > 0)
        expected = (100.0**2 + 100.0**2) ** 0.9
        np.testing.assert_allclose(func.evaluate(x), expected, rtol=1e-10)

    def test_evaluate_batch(self):
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction

        func = AttractiveSectorFunction(dimension=2)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
        results = func.evaluate_batch(X)
        expected = np.array([func.evaluate(X[i]) for i in range(3)])
        np.testing.assert_allclose(results, expected, rtol=1e-12)

    def test_x_opt_wrong_shape_raises(self):
        """x_opt with wrong shape must raise ValueError."""
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction

        with pytest.raises(ValueError, match="x_opt"):
            AttractiveSectorFunction(dimension=3, x_opt=np.array([1.0, -1.0]))

    def test_registry_aliases(self):
        from pyMOFL.registry import get

        cls = get("attractive_sector")
        assert cls is not None
