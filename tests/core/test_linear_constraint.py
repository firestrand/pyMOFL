"""Tests for LinearConstraint."""

import numpy as np
import pytest

from pyMOFL.core.linear_constraint import LinearConstraint


class TestLinearConstraint:
    """Tests for LinearConstraint: g(x) = a^T (x - shift * a/||a||) <= 0."""

    def test_evaluate_at_origin(self):
        """Basic evaluation at origin."""
        a = np.array([1.0, 0.0])
        c = LinearConstraint(normal=a, shift=0.0)
        # g(0) = a^T @ (0 - 0) = 0
        assert c.evaluate(np.array([0.0, 0.0])) == pytest.approx(0.0)

    def test_feasible_point(self):
        """Point on feasible side should have g(x) <= 0."""
        a = np.array([1.0, 0.0])
        c = LinearConstraint(normal=a, shift=0.0)
        # g([-1, 0]) = 1*(-1) + 0*0 = -1 <= 0
        assert c.evaluate(np.array([-1.0, 0.0])) == pytest.approx(-1.0)
        assert c.is_feasible(np.array([-1.0, 0.0]))

    def test_infeasible_point(self):
        """Point on infeasible side should have g(x) > 0."""
        a = np.array([1.0, 0.0])
        c = LinearConstraint(normal=a, shift=0.0)
        # g([1, 0]) = 1*1 + 0*0 = 1 > 0
        assert c.evaluate(np.array([1.0, 0.0])) == pytest.approx(1.0)
        assert not c.is_feasible(np.array([1.0, 0.0]))

    def test_shift_behavior(self):
        """Shift moves the constraint boundary along the normal direction."""
        a = np.array([1.0, 0.0])
        c = LinearConstraint(normal=a, shift=2.0)
        # g(x) = a^T (x - 2 * a/||a||) = a^T x - 2
        # g([2, 0]) = 2 - 2 = 0 (on boundary)
        assert c.evaluate(np.array([2.0, 0.0])) == pytest.approx(0.0)
        # g([0, 0]) = 0 - 2 = -2 (feasible)
        assert c.evaluate(np.array([0.0, 0.0])) == pytest.approx(-2.0)

    def test_is_active_binding(self):
        """Active constraint (shift=0) should bind at origin."""
        a = np.array([1.0, 0.0])
        c = LinearConstraint(normal=a, shift=0.0, is_active=True)
        assert c.is_active

    def test_is_active_inactive(self):
        """Inactive constraint."""
        a = np.array([1.0, 0.0])
        c = LinearConstraint(normal=a, shift=1.0, is_active=False)
        assert not c.is_active

    def test_multidimensional(self):
        """Should work with higher dimensions."""
        a = np.array([1.0, 2.0, 3.0])
        c = LinearConstraint(normal=a, shift=0.0)
        x = np.array([1.0, 1.0, 1.0])
        # g(x) = (1*1 + 2*1 + 3*1) = 6
        assert c.evaluate(x) == pytest.approx(6.0)

    def test_non_unit_normal(self):
        """Should handle non-unit normals correctly."""
        a = np.array([3.0, 4.0])
        c = LinearConstraint(normal=a, shift=5.0)
        # ||a|| = 5, a/||a|| = [0.6, 0.8]
        # shift_vec = 5 * [0.6, 0.8] = [3, 4]
        # g(x) = a^T (x - [3, 4]) = 3(x1-3) + 4(x2-4)
        # g([3, 4]) = 0
        assert c.evaluate(np.array([3.0, 4.0])) == pytest.approx(0.0)

    def test_batch_evaluate(self):
        """Batch evaluation should match individual evaluations."""
        a = np.array([1.0, -1.0])
        c = LinearConstraint(normal=a, shift=0.0)
        X = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
            ]
        )
        batch = c.evaluate_batch(X)
        for i in range(X.shape[0]):
            assert batch[i] == pytest.approx(c.evaluate(X[i]))

    def test_repr(self):
        a = np.array([1.0, 0.0])
        c = LinearConstraint(normal=a, shift=0.0)
        assert "LinearConstraint" in repr(c)
