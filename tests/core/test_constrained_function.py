"""Tests for ConstrainedFunction."""

import numpy as np
import pytest

from pyMOFL.core.constrained_function import ConstrainedFunction
from pyMOFL.core.linear_constraint import LinearConstraint
from pyMOFL.functions.benchmark.sphere import SphereFunction


@pytest.fixture
def sphere():
    return SphereFunction(dimension=2)


@pytest.fixture
def constraints():
    # Constraint 1: x1 <= 0 (feasible when x1 <= 0)
    c1 = LinearConstraint(normal=np.array([1.0, 0.0]), shift=0.0, is_active=True)
    # Constraint 2: x2 <= 1 (feasible when x2 <= 1)
    c2 = LinearConstraint(normal=np.array([0.0, 1.0]), shift=1.0, is_active=False)
    return [c1, c2]


class TestConstrainedFunction:
    """Tests for ConstrainedFunction."""

    def test_evaluate_is_objective_only(self, sphere, constraints):
        """evaluate() should return the objective function value only."""
        cf = ConstrainedFunction(base_function=sphere, constraints=constraints)
        x = np.array([1.0, 2.0])
        # Sphere value at [1, 2] = 1 + 4 = 5
        assert cf.evaluate(x) == pytest.approx(5.0)

    def test_evaluate_constraints_shape(self, sphere, constraints):
        """evaluate_constraints should return array of constraint values."""
        cf = ConstrainedFunction(base_function=sphere, constraints=constraints)
        x = np.array([1.0, 2.0])
        g = cf.evaluate_constraints(x)
        assert g.shape == (2,)
        # g1(x) = 1*1 + 0*2 = 1.0 (infeasible)
        assert g[0] == pytest.approx(1.0)
        # g2(x) = 0*1 + 1*(2 - 1) = 1.0 (infeasible)
        assert g[1] == pytest.approx(1.0)

    def test_violations_at_feasible_point(self, sphere, constraints):
        """violations() should return zeros at feasible point."""
        cf = ConstrainedFunction(base_function=sphere, constraints=constraints)
        x = np.array([-1.0, 0.0])
        v = cf.violations(x)
        assert v.shape == (2,)
        # g1 = -1.0 <= 0 → violation = 0
        # g2 = 0 - 1 = -1 <= 0 → violation = 0
        np.testing.assert_array_equal(v, [0.0, 0.0])

    def test_violations_at_infeasible_point(self, sphere, constraints):
        """violations() should return positive values at infeasible point."""
        cf = ConstrainedFunction(base_function=sphere, constraints=constraints)
        x = np.array([2.0, 3.0])
        v = cf.violations(x)
        # g1 = 2.0 > 0 → violation = 2.0
        assert v[0] == pytest.approx(2.0)
        # g2 = 3 - 1 = 2.0 > 0 → violation = 2.0
        assert v[1] == pytest.approx(2.0)

    def test_is_feasible(self, sphere, constraints):
        """is_feasible should check all constraints."""
        cf = ConstrainedFunction(base_function=sphere, constraints=constraints)
        assert cf.is_feasible(np.array([-1.0, 0.0]))
        assert not cf.is_feasible(np.array([1.0, 0.0]))

    def test_dimension_propagation(self, sphere, constraints):
        """Dimension should match base function."""
        cf = ConstrainedFunction(base_function=sphere, constraints=constraints)
        assert cf.dimension == 2

    def test_num_constraints(self, sphere, constraints):
        """Should expose number of constraints."""
        cf = ConstrainedFunction(base_function=sphere, constraints=constraints)
        assert cf.num_constraints == 2

    def test_callable(self, sphere, constraints):
        """Should be callable like any OptimizationFunction."""
        cf = ConstrainedFunction(base_function=sphere, constraints=constraints)
        result = cf(np.array([1.0, 1.0]))
        assert isinstance(result, float)

    def test_empty_constraints(self, sphere):
        """Should work with no constraints."""
        cf = ConstrainedFunction(base_function=sphere, constraints=[])
        assert cf.num_constraints == 0
        assert cf.is_feasible(np.array([1.0, 1.0]))
        assert cf(np.array([1.0, 1.0])) == pytest.approx(2.0)
