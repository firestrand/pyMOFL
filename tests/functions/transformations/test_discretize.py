"""Tests for DiscretizeTransform (mixed-integer variable discretization)."""

import numpy as np
import pytest

from pyMOFL.functions.transformations.discretize import DiscretizeTransform


class TestDiscretizeTransform:
    """Tests for DiscretizeTransform.

    COCO discretization: 80% discrete, 20% continuous.
    D must be divisible by 5. Variables split into 5 groups of D/5:
    arities = [2, 4, 8, 16, continuous(0)].

    For arity n, inner bounds [-4, 4]:
      inner_l = -4 + 8/(n+1)
      inner_u = 4 - 8/(n+1)
      x_inner = inner_l + (inner_u - inner_l) * round(x_outer) / (n-1)
    Continuous variables (arity=0) pass through unchanged.
    """

    def test_continuous_passthrough(self):
        """Variables with arity=0 (continuous) should pass through unchanged."""
        # D=5: arities = [2, 4, 8, 16, 0]. Variable 4 is continuous.
        t = DiscretizeTransform(dimension=5)
        x = np.array([0.0, 0.0, 0.0, 0.0, 3.14159])
        result = t(x)
        assert result[4] == pytest.approx(3.14159)

    def test_binary_discretization(self):
        """Arity=2: 2 levels mapping to 2 inner points."""
        t = DiscretizeTransform(dimension=5)
        # First variable has arity 2: levels are 0 and 1
        # inner_l = -4 + 8/3 = -4 + 2.667 = -1.333
        # inner_u = 4 - 8/3 = 4 - 2.667 = 1.333
        # x_outer=0 -> inner_l = -1.333
        # x_outer=1 -> inner_u = 1.333
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        x1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        r0 = t(x0)
        r1 = t(x1)
        assert r0[0] == pytest.approx(-4.0 + 8.0 / 3.0)
        assert r1[0] == pytest.approx(4.0 - 8.0 / 3.0)

    def test_four_level_discretization(self):
        """Arity=4: 4 levels mapping to 4 inner points."""
        t = DiscretizeTransform(dimension=5)
        # Second variable (index 1) has arity 4
        # inner_l = -4 + 8/5 = -2.4
        # inner_u = 4 - 8/5 = 2.4
        # x_outer=0 -> -2.4
        # x_outer=1 -> -2.4 + 4.8/3 = -2.4 + 1.6 = -0.8
        # x_outer=2 -> -2.4 + 4.8*2/3 = -2.4 + 3.2 = 0.8
        # x_outer=3 -> 2.4
        x = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
        result = t(x)
        assert result[1] == pytest.approx(0.8)

    def test_clamping(self):
        """Out-of-range outer values should be clamped to [0, n-1]."""
        t = DiscretizeTransform(dimension=5)
        # Binary: clamp -1 to 0, clamp 5 to 1
        x_low = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
        x_high = np.array([5.0, 0.0, 0.0, 0.0, 0.0])
        r_low = t(x_low)
        r_high = t(x_high)
        # Clamped to 0 -> inner_l
        assert r_low[0] == pytest.approx(-4.0 + 8.0 / 3.0)
        # Clamped to 1 -> inner_u
        assert r_high[0] == pytest.approx(4.0 - 8.0 / 3.0)

    def test_five_group_arities_dim10(self):
        """D=10: each group of 2 variables gets arity [2,4,8,16,0]."""
        t = DiscretizeTransform(dimension=10)
        assert t.arities == [2, 2, 4, 4, 8, 8, 16, 16, 0, 0]

    def test_five_group_arities_dim20(self):
        """D=20: each group of 4 variables."""
        t = DiscretizeTransform(dimension=20)
        expected = [2] * 4 + [4] * 4 + [8] * 4 + [16] * 4 + [0] * 4
        assert t.arities == expected

    def test_affine_mapping_bounds(self):
        """All discrete values should map within [-4, 4]."""
        for arity in [2, 4, 8, 16]:
            inner_l = -4.0 + 8.0 / (arity + 1)
            inner_u = 4.0 - 8.0 / (arity + 1)
            assert inner_l >= -4.0
            assert inner_u <= 4.0
            assert inner_l < inner_u

    def test_batch_consistency(self):
        """Batch transform should match individual transforms."""
        t = DiscretizeTransform(dimension=5)
        X = np.array(
            [
                [0.0, 1.0, 3.0, 7.0, 2.5],
                [1.0, 3.0, 7.0, 15.0, -1.0],
                [0.5, 2.0, 4.0, 8.0, 0.0],
            ]
        )
        batch_result = t.transform_batch(X)
        for i in range(X.shape[0]):
            single_result = t(X[i])
            np.testing.assert_array_almost_equal(batch_result[i], single_result)

    def test_dimension_not_divisible_by_5(self):
        """D not divisible by 5 should raise ValueError."""
        with pytest.raises(ValueError, match="divisible by 5"):
            DiscretizeTransform(dimension=7)

    def test_rounding(self):
        """Fractional outer values should be rounded to nearest integer."""
        t = DiscretizeTransform(dimension=5)
        # Binary arity: 0.3 rounds to 0, 0.7 rounds to 1
        x03 = np.array([0.3, 0.0, 0.0, 0.0, 0.0])
        x07 = np.array([0.7, 0.0, 0.0, 0.0, 0.0])
        r03 = t(x03)
        r07 = t(x07)
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        x1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        assert r03[0] == pytest.approx(t(x0)[0])  # rounds to 0
        assert r07[0] == pytest.approx(t(x1)[0])  # rounds to 1

    def test_sixteen_level(self):
        """Arity=16: all 16 levels should be distinct."""
        t = DiscretizeTransform(dimension=5)
        values = set()
        for k in range(16):
            x = np.array([0.0, 0.0, 0.0, float(k), 0.0])
            result = t(x)
            values.add(round(result[3], 10))
        assert len(values) == 16

    def test_offset_alignment(self):
        """Transform with xopt offset should snap optimum to grid."""
        xopt = np.array([0.5, 1.5, 3.5, 7.5, 2.0])
        t = DiscretizeTransform(dimension=5, xopt=xopt)
        # At xopt, discrete variables should map to specific grid points
        result = t(xopt)
        # Continuous variable should be unchanged
        assert result[4] == pytest.approx(xopt[4])
        # Result should be finite
        assert np.all(np.isfinite(result))
