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
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction

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
        15: -52.32
    }

    def __init__(self, n_atoms: int = 6,
                 initialization_bounds: Bounds = None,
                 operational_bounds: Bounds = None):
        dimension = 3 * n_atoms
        default_init_bounds = Bounds(
            low=np.full(dimension, -2.0),
            high=np.full(dimension, 2.0),
            mode=BoundModeEnum.INITIALIZATION,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        default_oper_bounds = Bounds(
            low=np.full(dimension, -2.0),
            high=np.full(dimension, 2.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_init_bounds,
            operational_bounds=operational_bounds or default_oper_bounds
        )
        self.n_atoms = n_atoms
        self.global_minimum = self.LJ_GLOBAL_MINIMA.get(n_atoms, None)

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
                dist2 = np.sum((coords[i] - coords[j])**2)
                if dist2 < 1e-12:
                    energy += 1e10  # Large penalty for overlapping atoms
                else:
                    inv_dist2 = 1.0 / dist2
                    inv_dist6 = inv_dist2**3
                    inv_dist12 = inv_dist6**2
                    energy += 4.0 * (inv_dist12 - inv_dist6)
        return float(energy)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation of the Lennard-Jones potential energy.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_points, 3 * n_atoms).

        Returns
        -------
        np.ndarray
            The potential energy for each point.
        """
        X = self._validate_batch_input(X)
        return np.array([self.evaluate(x) for x in X])

    @staticmethod
    def get_global_minimum(n_atoms: int = 6) -> tuple:
        """
        Get the global minimum of the Lennard-Jones function for a given number of atoms.

        Parameters
        ----------
        n_atoms : int, optional
            Number of atoms in the cluster. Defaults to 6.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
            global_min_point is a zero vector of length 3*n_atoms (placeholder, not unique).
            global_min_value is the reference minimum energy from the LJ_GLOBAL_MINIMA table.

        Notes
        -----
        The true global minimum coordinates are not unique due to rotational and translational symmetry.
        This method returns a zero vector as a placeholder for the coordinates, and the reference minimum
        energy from the literature for the given n_atoms.
        """
        dimension = 3 * n_atoms
        global_min_point = np.zeros(dimension)
        global_min_value = LennardJonesFunction.LJ_GLOBAL_MINIMA.get(n_atoms, None)
        return global_min_point, global_min_value