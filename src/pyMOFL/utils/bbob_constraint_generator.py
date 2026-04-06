"""BBOBConstraintGenerator — generates linear constraints for bbob-constrained.

Each constraint: g_i(x) = a_i^T (x - shift * a_i/||a_i||) <= 0

Active constraints have shift=0 (binding at optimum).
Inactive constraints have shift > 0 (feasible at optimum).
"""

from __future__ import annotations

import numpy as np

from pyMOFL.core.linear_constraint import LinearConstraint

# Config -> number of active constraints
_ACTIVE_COUNTS = {
    1: lambda d: 1,
    2: lambda d: 2,
    3: lambda d: 6,
    4: lambda d: 6 + d // 2,
    5: lambda d: 6 + d,
    6: lambda d: 6 + 3 * d,
}


class BBOBConstraintGenerator:
    """Generate linear constraints for BBOB constrained suite.

    Generates constraints where:
    - First constraint normal is related to -gradient at optimum (disguised)
    - Remaining normals are random
    - Active constraints are binding at xopt (shift=0)
    - Inactive constraints are feasible at xopt (shift > 0)
    """

    def generate(
        self,
        fid: int,
        iid: int,
        dim: int,
        config: int,
        xopt: np.ndarray,
    ) -> list[LinearConstraint]:
        """Generate constraints for a given function/config.

        Parameters
        ----------
        fid : int
            Objective function ID (determines first normal).
        iid : int
            Instance ID.
        dim : int
            Dimension.
        config : int
            Constraint configuration (1-6).
        xopt : np.ndarray
            Optimum point (constraints are binding/feasible here).

        Returns
        -------
        list[LinearConstraint]
            Active constraints first, then inactive.
        """
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}.")
        if config < 1 or config > 6:
            raise ValueError(f"Constraint config must be 1-6, got {config}.")
        if fid <= 0:
            raise ValueError(f"Function ID must be positive, got {fid}.")

        xopt = np.asarray(xopt, dtype=np.float64)
        if xopt.shape != (dim,):
            raise ValueError(f"xopt must have shape ({dim},), got {xopt.shape}.")

        n_active = _ACTIVE_COUNTS[config](dim)
        n_inactive = n_active // 2
        n_total = n_active + n_inactive

        # Deterministic RNG based on fid, iid, config
        seed = fid * 100000 + iid * 1000 + config * 10
        rng = np.random.default_rng(seed)

        # Generate constraint normals
        normals = self._generate_normals(fid, dim, n_total, rng)

        constraints: list[LinearConstraint] = []

        # Active constraints: shift=0 (binding at xopt)
        for i in range(n_active):
            # Compute shift so that a^T (xopt - shift * a_hat) = 0
            # => shift = a^T xopt / ||a||
            a = normals[i]
            norm = np.linalg.norm(a)
            shift = float(a @ xopt / norm) if norm > 0 else 0.0
            constraints.append(LinearConstraint(normal=a, shift=shift, is_active=True))

        # Inactive constraints: shift so that g(xopt) < 0
        for i in range(n_active, n_total):
            a = normals[i]
            norm = np.linalg.norm(a)
            # shift = a^T xopt / ||a|| + margin
            base_shift = float(a @ xopt / norm) if norm > 0 else 0.0
            margin = float(rng.uniform(0.1, 1.0) * norm)
            shift = float(base_shift + margin)
            constraints.append(LinearConstraint(normal=a, shift=shift, is_active=False))

        return constraints

    def _generate_normals(
        self,
        fid: int,
        dim: int,
        n_total: int,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Generate constraint normal vectors.

        First normal is based on function ID (gradient disguising).
        Remaining normals are random.
        """
        normals: list[np.ndarray] = []

        # First normal: pseudo-gradient direction based on fid
        # Different fids get different "gradient" directions
        first_rng = np.random.default_rng(fid * 7919)
        first_normal = first_rng.standard_normal(dim)
        # Apply gradient disguising: rotate by a fixed amount
        disguise_rng = np.random.default_rng(fid * 1009)
        disguise = disguise_rng.standard_normal(dim)
        first_normal = first_normal + 0.1 * disguise
        norm = np.linalg.norm(first_normal)
        if norm > 0:
            first_normal = first_normal / norm
        normals.append(first_normal)

        # Remaining normals: random
        for _ in range(n_total - 1):
            a = rng.standard_normal(dim)
            norm = np.linalg.norm(a)
            if norm > 0:
                a = a / norm
            normals.append(a)

        return normals
