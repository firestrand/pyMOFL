"""
Rotation utilities for generating orthogonal matrices.
Includes implementation of the Givens-based interaction rotation from GNBG.
"""

import numpy as np
from numpy.typing import NDArray


def build_rotation_from_theta(dimension: int, theta: NDArray) -> NDArray:
    """
    Build a rotation matrix from an interaction matrix Theta (Algorithm 1 from GNBG).

    Theta is a d x d matrix where elements Theta[p, q] (p < q) define the
    rotation angle in the (p, q) plane.

    Parameters
    ----------
    dimension : int
        The dimensionality of the space.
    theta : NDArray
        An upper triangular matrix of angles (in radians).

    Returns
    -------
    NDArray
        The combined orthogonal rotation matrix.
    """
    R = np.eye(dimension)

    # Iterate through all pairs p, q where p < q
    for p in range(dimension - 1):
        for q in range(p + 1, dimension):
            angle = theta[p, q]
            if angle == 0:
                continue

            # Construct Givens rotation matrix G for the p-q plane
            G = np.eye(dimension)
            c = np.cos(angle)
            s = np.sin(angle)

            G[p, p] = c
            G[q, q] = c
            G[p, q] = -s
            G[q, p] = s

            # Combine
            R = R @ G

    return R


def random_rotation_matrix(dimension: int, rng: np.random.Generator) -> NDArray:
    """Generate a random orthogonal rotation matrix using QR decomposition."""
    if dimension == 1:
        return np.array([[1.0]])
    H = rng.standard_normal((dimension, dimension))
    Q, R = np.linalg.qr(H)
    # Correct signs of Q to ensure it's a proper rotation (det=1) and Haar-distributed
    d = np.diag(R)
    Q *= np.sign(d)
    return Q
