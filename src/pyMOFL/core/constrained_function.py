"""ConstrainedFunction — optimization function with linear constraints."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.linear_constraint import LinearConstraint


class ConstrainedFunction(OptimizationFunction):
    """Optimization function with associated linear constraints.

    evaluate() returns the objective value only.
    evaluate_constraints() returns the constraint values.
    violations() returns max(0, g_i) for each constraint.

    Parameters
    ----------
    base_function : OptimizationFunction
        The objective function.
    constraints : list[LinearConstraint]
        List of linear constraints.
    xopt : np.ndarray, optional
        The known optimum point (for feasibility checks).
    """

    def __init__(
        self,
        base_function: OptimizationFunction,
        constraints: list[LinearConstraint],
        xopt: np.ndarray | None = None,
    ):
        super().__init__(
            dimension=base_function.dimension,
            initialization_bounds=base_function.initialization_bounds,
            operational_bounds=base_function.operational_bounds,
        )
        if not isinstance(constraints, list) or not all(
            isinstance(c, LinearConstraint) for c in constraints
        ):
            raise TypeError("constraints must be a list of LinearConstraint objects")
        self.base_function = base_function
        self.constraints = constraints
        self.xopt = xopt
        if self.xopt is not None:
            xopt_arr = np.asarray(self.xopt)
            if xopt_arr.shape != (self.dimension,):
                raise ValueError(f"xopt must have shape ({self.dimension},), got {xopt_arr.shape}.")
            self.xopt = xopt_arr

    @property
    def num_constraints(self) -> int:
        return len(self.constraints)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate objective function only."""
        return float(self.base_function(x))

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all constraint values.

        Returns array of g_i(x). Feasible when all <= 0.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.dimension,):
            raise ValueError(f"Input must have shape ({self.dimension},), got {x.shape}.")
        return np.array([c.evaluate(x) for c in self.constraints])

    def violations(self, x: NDArray[Any]) -> NDArray[Any]:
        """Return max(0, g_i(x)) for each constraint."""
        g = self.evaluate_constraints(x)
        return np.maximum(0.0, g)

    def is_feasible(self, x: np.ndarray) -> bool:
        """Check if all constraints are satisfied."""
        if self.num_constraints == 0:
            return True
        return bool(np.all(self.evaluate_constraints(x) <= 1e-10))

    def __repr__(self) -> str:
        return f"ConstrainedFunction(dim={self.dimension}, constraints={self.num_constraints})"
