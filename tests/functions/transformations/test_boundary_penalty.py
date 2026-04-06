"""Tests for BoundaryPenaltyTransform (f_pen from BBOB/Hansen et al. 2009)."""

import numpy as np

from pyMOFL.functions.transformations.base import PenaltyTransform
from pyMOFL.functions.transformations.boundary_penalty import BoundaryPenaltyTransform


class TestBoundaryPenaltyTransform:
    """Tests for BoundaryPenaltyTransform."""

    def test_is_penalty_transform(self):
        bp = BoundaryPenaltyTransform()
        assert isinstance(bp, PenaltyTransform)

    def test_call_returns_float(self):
        bp = BoundaryPenaltyTransform()
        result = bp(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, (float, np.floating))

    def test_compute_batch_returns_1d_ndarray(self):
        bp = BoundaryPenaltyTransform()
        X = np.ones((5, 3))
        result = bp.compute_batch(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_batch_consistent_with_single(self):
        bp = BoundaryPenaltyTransform()
        rng = np.random.default_rng(42)
        X = rng.uniform(-10, 10, size=(8, 4))
        batch_result = bp.compute_batch(X)
        for i in range(len(X)):
            single_result = bp(X[i])
            np.testing.assert_allclose(
                batch_result[i],
                single_result,
                rtol=1e-14,
                err_msg=f"Mismatch at row {i}",
            )

    def test_zero_penalty_within_bounds(self):
        bp = BoundaryPenaltyTransform()
        x = np.array([0.0, 1.0, -3.0, 4.9])
        assert bp(x) == 0.0

    def test_zero_penalty_at_exact_bound(self):
        bp = BoundaryPenaltyTransform()
        x = np.array([5.0, -5.0])
        assert bp(x) == 0.0

    def test_zero_penalty_at_origin(self):
        bp = BoundaryPenaltyTransform()
        assert bp(np.zeros(10)) == 0.0

    def test_single_violation(self):
        bp = BoundaryPenaltyTransform()
        x = np.array([6.0, 0.0, 0.0])
        np.testing.assert_allclose(bp(x), 1.0)  # (6-5)^2 = 1

    def test_multiple_violations(self):
        bp = BoundaryPenaltyTransform()
        x = np.array([7.0, -8.0, 3.0])
        # (7-5)^2 + (8-5)^2 + 0 = 4 + 9 = 13
        np.testing.assert_allclose(bp(x), 13.0)

    def test_symmetric(self):
        bp = BoundaryPenaltyTransform()
        assert bp(np.array([6.0])) == bp(np.array([-6.0]))

    def test_all_violating(self):
        bp = BoundaryPenaltyTransform()
        x = np.array([10.0, -10.0, 10.0])
        # 3 * (10-5)^2 = 3*25 = 75
        np.testing.assert_allclose(bp(x), 75.0)

    def test_custom_bound(self):
        bp = BoundaryPenaltyTransform(bound=3.0)
        x = np.array([4.0, 2.0])
        # (4-3)^2 + 0 = 1
        np.testing.assert_allclose(bp(x), 1.0)

    def test_default_bound_is_five(self):
        bp = BoundaryPenaltyTransform()
        assert bp.bound == 5.0

    def test_quadratic_growth(self):
        bp = BoundaryPenaltyTransform()
        # Verify penalty grows quadratically: penalty(6) = 1, penalty(7) = 4, penalty(8) = 9
        p1 = bp(np.array([6.0]))
        p2 = bp(np.array([7.0]))
        p3 = bp(np.array([8.0]))
        np.testing.assert_allclose(p1, 1.0)
        np.testing.assert_allclose(p2, 4.0)
        np.testing.assert_allclose(p3, 9.0)

    def test_high_dimension(self):
        bp = BoundaryPenaltyTransform()
        x = np.full(100, 10.0)
        # 100 * (10-5)^2 = 100*25 = 2500
        np.testing.assert_allclose(bp(x), 2500.0)

    def test_batch_mixed_penalties(self):
        bp = BoundaryPenaltyTransform()
        X = np.array(
            [
                [0.0, 0.0, 0.0],  # penalty = 0
                [6.0, 0.0, 0.0],  # penalty = 1
                [10.0, -10.0, 0.0],  # penalty = 50
            ]
        )
        result = bp.compute_batch(X)
        np.testing.assert_allclose(result, [0.0, 1.0, 50.0])
