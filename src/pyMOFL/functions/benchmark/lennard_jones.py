"""
Lennard-Jones 6-atom cluster function implementation.

This module implements the Lennard-Jones potential energy function for a cluster
of 6 identical atoms, a classic benchmark in computational chemistry and physics.
The function has a highly rugged energy landscape with thousands of local minima,
making it challenging for optimization algorithms.

References
----------
.. [1] Lennard-Jones, J.E. (1924). "On the Determination of Molecular
       Fields. II." *Proc. R. Soc. A*, 106, 463-477.
.. [2] Wales, D.J., & Doye, J.P.K. (1997). "Global Optimization by
       Basin-Hopping and the Lowest Energy Structures of Lennard-Jones
       Clusters Containing up to 110 Atoms." *J. Phys. Chem. A*, 101,
       5111-5116.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("LennardJones")
@register("lennard_jones")
class LennardJonesFunction(OptimizationFunction):
    """
    Lennard-Jones n-atom cluster potential energy function (SPSO ID-17).

    This function calculates the 12-6 Lennard-Jones potential energy for a cluster
    of n identical atoms. The atoms' positions are represented by their Cartesian
    coordinates in reduced units (σ=ε=1).

    The function is defined as:
        E(r) = 4 * ∑_{i<j} [(rij)^(-12) - (rij)^(-6)]
    where rij is the Euclidean distance between atoms i and j.

    Global minimum: E = -12.7121 at the octahedral (Oh) structure for n=6.

    Parameters
    ----------
    n_atoms : int, optional
        Number of atoms in the cluster. Defaults to 6.
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, defaults to [-2, 2] for each coordinate.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, defaults to [-2, 2] for each coordinate.

    References
    ----------
    .. [1] Lennard-Jones, J.E. (1924). "On the Determination of Molecular
           Fields. II." *Proc. R. Soc. A*, 106, 463-477.
    .. [2] Wales, D.J., & Doye, J.P.K. (1997). "Global Optimization by
           Basin-Hopping and the Lowest Energy Structures of Lennard-Jones
           Clusters Containing up to 110 Atoms." *J. Phys. Chem. A*, 101,
           5111-5116.
    """

    LJ_GLOBAL_MINIMA = {
        2: -1.0,
        3: -3.0,
        4: -6.0,
        5: -9.103852,
        6: -12.7121,  # In practice, simple octahedral structure gives ~ -6.937
        7: -16.505384,
        8: -19.821489,
        9: -24.113360,
        10: -28.422532,
        11: -32.77,
        12: -37.97,
        13: -44.33,
        14: -47.84,
        15: -52.32,
    }

    def __init__(
        self,
        n_atoms: int | None = None,
        dimension: int | None = None,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        dim: int
        atoms: int
        if n_atoms is not None and n_atoms < 2:
            raise ValueError(f"n_atoms must be >= 2; got {n_atoms}")

        if dimension is None and n_atoms is None:
            atoms = 6
            dim = 18
        elif dimension is not None and n_atoms is not None:
            if dimension != 3 * n_atoms:
                raise ValueError(
                    f"dimension must equal 3 * n_atoms when both are provided; got dimension={dimension}, n_atoms={n_atoms}"
                )
            dim = dimension
            atoms = n_atoms
        elif dimension is not None:
            if dimension % 3 != 0 or dimension < 6:
                raise ValueError(
                    f"dimension must be a multiple of 3 and >= 6; got dimension={dimension}"
                )
            dim = dimension
            atoms = dimension // 3
        else:
            assert n_atoms is not None
            atoms = n_atoms
            dim = 3 * n_atoms

        if dim % 3 != 0 or dim < 6:
            raise ValueError(f"dimension must be a multiple of 3 and >= 6; got dimension={dim}")
        default_init_bounds = Bounds(
            low=np.full(dim, -2.0),
            high=np.full(dim, 2.0),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        default_oper_bounds = Bounds(
            low=np.full(dim, -2.0),
            high=np.full(dim, 2.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dim,
            initialization_bounds=initialization_bounds or default_init_bounds,
            operational_bounds=operational_bounds or default_oper_bounds,
        )
        self.n_atoms: int = atoms
        self.global_minimum = self.LJ_GLOBAL_MINIMA.get(atoms, None)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Lennard-Jones potential energy at a single point.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (3 * n_atoms,).

        Returns
        -------
        float
            The potential energy at x.
        """
        x = self._validate_input(x)
        coords = x.reshape(self.n_atoms, 3)
        energy = 0.0
        for i in range(self.n_atoms - 1):
            for j in range(i + 1, self.n_atoms):
                dist2 = np.sum((coords[i] - coords[j]) ** 2)
                if dist2 < 1e-12:
                    energy += 1e10  # Large penalty for overlapping atoms
                else:
                    inv_dist2 = 1.0 / dist2
                    inv_dist6 = inv_dist2**3
                    inv_dist12 = inv_dist6**2
                    energy += 4.0 * (inv_dist12 - inv_dist6)
        return float(energy)

    def evaluate_batch(self, X: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """
        Vectorized batch evaluation of the Lennard-Jones potential energy.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_points, 3 * n_atoms).
        out : np.ndarray | None, optional
            Pre-allocated output buffer of shape (n_points,).

        Returns
        -------
        np.ndarray
            The potential energy for each point.
        """
        X = self._validate_batch_input(X)
        N = X.shape[0]
        coords = X.reshape(N, self.n_atoms, 3)
        i_idx, j_idx = np.triu_indices(self.n_atoms, k=1)
        diff = coords[:, i_idx, :] - coords[:, j_idx, :]
        dist2 = np.sum(diff**2, axis=-1)

        valid = dist2 >= 1e-12
        energy = np.full_like(dist2, 1e10)

        d2_v = dist2[valid]
        inv_dist2 = 1.0 / d2_v
        inv_dist6 = inv_dist2**3
        inv_dist12 = inv_dist6**2
        energy[valid] = 4.0 * (inv_dist12 - inv_dist6)

        total = np.sum(energy, axis=-1)
        if out is not None:
            out[:] = total
            return out
        return total

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Lennard-Jones function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
            global_min_point is a zero vector of length 3*n_atoms (placeholder, not unique).
            global_min_value is the reference minimum energy from the LJ_GLOBAL_MINIMA table.

        Notes
        -----
        The true global minimum coordinates are not unique due to rotational and translational symmetry.
        This method returns a zero vector as a placeholder for the coordinates, and the reference minimum
        energy from the literature for the given n_atoms.
        """
        global_min_point = np.zeros(self.dimension)
        global_min_value = LennardJonesFunction.LJ_GLOBAL_MINIMA.get(self.n_atoms)
        if global_min_value is None:
            global_min_value = float(self.evaluate(global_min_point))
        return global_min_point, float(global_min_value)


@register("LennardJonesCEC")
@register("lennard_jones_cec")
class LennardJonesCECFunction(OptimizationFunction):
    """
    Lennard-Jones atomic cluster potential energy function (CEC 2019 F3 variant).

    Implements the atomic cluster potential energy function defined for the
    CEC 2019 100-Digit Challenge:
        f(x) = sum_{i < j} (1 / ud - 2) / ud + offset
    where ud = r^6 = (dist^2)^3, with an overlapping atom penalty of 1e20 when
    ud <= 1e-10. The offset shifts the theoretical minimum to 0.0.

    Parameters
    ----------
    dimension : int, optional
        Problem dimensionality (must be 3 * k where k >= 2 is number of atoms, default 18 for 6 atoms).
    initialization_bounds : Bounds, optional
        Bounds for random initialization. Defaults to [-4.0, 4.0]^D.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. Defaults to [-4.0, 4.0]^D.
    """

    CEC_MINIMA = [
        -1.0,
        -3.0,
        -6.0,
        -9.103852,
        -12.7120622568,
        -16.505384,
        -19.821489,
        -24.113360,
        -28.422532,
        -32.765970,
        -37.967600,
        -44.326801,
        -47.845157,
        -52.322627,
        -56.815742,
        -61.317995,
        -66.530949,
        -72.659782,
        -77.1777043,
        -81.684571,
        -86.809782,
        -92.844472,
        -97.348815,
        -102.372663,
    ]

    def __init__(
        self,
        dimension: int = 18,
        n_atoms: int | None = None,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if n_atoms is not None and dimension is not None and kwargs.get("_from_factory"):
            pass
        elif dimension is not None and n_atoms is not None:
            if dimension != 3 * n_atoms:
                raise ValueError(
                    f"dimension must equal 3 * n_atoms when both are provided; got dimension={dimension}, n_atoms={n_atoms}"
                )
        elif n_atoms is not None:
            dimension = 3 * n_atoms
        elif dimension is None:
            dimension = 18

        if dimension % 3 != 0 or dimension < 6:
            raise ValueError(
                f"LennardJonesCECFunction dimension must be a multiple of 3 >= 6, got {dimension}"
            )

        k = dimension // 3
        if k < 2 or (k - 2) >= len(self.CEC_MINIMA):
            raise ValueError(
                f"LennardJonesCECFunction only supports k in [2, {len(self.CEC_MINIMA) + 1}] atoms "
                f"(dimension 6 to {3 * (len(self.CEC_MINIMA) + 1)}), got k={k} (dimension={dimension})"
            )
        self._k = k
        self._offset = -self.CEC_MINIMA[k - 2]

        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -4.0, dtype=np.float64),
                high=np.full(dimension, 4.0, dtype=np.float64),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -4.0, dtype=np.float64),
                high=np.full(dimension, 4.0, dtype=np.float64),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )

        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
        )

        self._i_idx, self._j_idx = np.triu_indices(self._k, k=1)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate CEC 2019 Lennard-Jones potential."""
        x = self._validate_input(x)
        coords = x.reshape(self._k, 3)
        diff = coords[self._i_idx] - coords[self._j_idx]
        ed = np.sum(diff**2, axis=-1)
        ud = ed**3
        valid = ud > 1e-10
        pairs = np.full_like(ud, 1e20)
        ud_v = ud[valid]
        pairs[valid] = (1.0 / ud_v - 2.0) / ud_v
        return float(np.sum(pairs) + self._offset)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Batch evaluate CEC 2019 Lennard-Jones potential."""
        X = self._validate_batch_input(X)
        N = X.shape[0]
        coords = X.reshape(N, self._k, 3)
        diff = coords[:, self._i_idx, :] - coords[:, self._j_idx, :]
        ed = np.sum(diff**2, axis=-1)
        ud = ed**3
        valid = ud > 1e-10
        pairs = np.full_like(ud, 1e20)
        ud_v = ud[valid]
        pairs[valid] = (1.0 / ud_v - 2.0) / ud_v
        return np.sum(pairs, axis=-1) + self._offset

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the CEC Lennard-Jones function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, 0.0) for k=2.

        Raises
        ------
        NotImplementedError
            For k > 2, analytical minimum coordinates are not stored.
        """
        if self._k == 2:
            min_pt = np.zeros(6, dtype=np.float64)
            min_pt[3] = 1.0  # atom 1 at (0, 0, 0), atom 2 at (1, 0, 0); distance = 1.0
            return min_pt, 0.0
        raise NotImplementedError(
            f"LennardJonesCECFunction (k={self._k}) does not have an analytical global minimum configuration."
        )
