"""
BBOB instance parameter generator — COCO-compatible.

Generates seeded, deterministic parameters for BBOB benchmark function instances:
x_opt, f_opt, rotation matrices, and conditioning vectors.

Implements the exact algorithms from the COCO framework (github.com/numbbo/coco)
to produce bit-for-bit identical instance parameters:

- PRNG: Bays-Durham shuffled Park-Miller LCG (``bbob2009_unif``)
- xopt: ``bbob2009_compute_xopt``
- fopt: ``bbob2009_compute_fopt`` (Cauchy via Gaussian ratio)
- Rotation: ``bbob2009_compute_rotation`` (classical Gram-Schmidt)

References
----------
.. [1] Hansen, N., et al. (2009). "Real-parameter black-box optimization
       benchmarking 2009: Noiseless functions definitions."
       INRIA Technical Report RR-6829.
.. [2] COCO source: ``code-experiments/src/suite_bbob_legacy_code.c``
"""

from __future__ import annotations

import math

import numpy as np

_COCO_PI = 3.14159265358979323846


# ---------------------------------------------------------------------------
# COCO PRNG — Bays-Durham shuffled Park-Miller LCG
# ---------------------------------------------------------------------------


def _bbob2009_unif(n: int, inseed: int) -> list[float]:
    """Generate *n* uniform random numbers using the COCO legacy PRNG.

    This is a faithful Python translation of ``bbob2009_unif`` from
    ``suite_bbob_legacy_code.c``.
    """
    if inseed < 0:
        inseed = -inseed
    if inseed < 1:
        inseed = 1
    aktseed = inseed

    rgrand = [0] * 32

    # Warm-up: 40 LCG iterations, last 32 fill the shuffle table
    for i in range(39, -1, -1):
        tmp = math.floor(aktseed / 127773)
        aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp
        if aktseed < 0:
            aktseed += 2147483647
        if i < 32:
            rgrand[i] = aktseed

    aktrand = rgrand[0]

    r = [0.0] * n
    for i in range(n):
        tmp = math.floor(aktseed / 127773)
        aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp
        if aktseed < 0:
            aktseed += 2147483647
        idx = math.floor(aktrand / 67108865)
        aktrand = rgrand[idx]
        rgrand[idx] = aktseed
        r[i] = aktrand / 2.147483647e9
        if r[i] == 0.0:
            r[i] = 1e-99

    return r


def _bbob2009_gauss(n: int, seed: int) -> list[float]:
    """Generate *n* Gaussian random numbers via Box-Muller.

    Faithful translation of ``bbob2009_gauss``.
    """
    unif = _bbob2009_unif(2 * n, seed)
    g = [0.0] * n
    for i in range(n):
        g[i] = math.sqrt(-2.0 * math.log(unif[i])) * math.cos(2.0 * _COCO_PI * unif[n + i])
        if g[i] == 0.0:
            g[i] = 1e-99
    return g


# ---------------------------------------------------------------------------
# xopt, fopt, rotation — COCO-compatible
# ---------------------------------------------------------------------------


def _compute_rseed(fid: int, iid: int) -> int:
    """Compute the base random seed for a BBOB function instance.

    Special cases match COCO exactly: f4 uses base 3, f18 uses base 17.
    """
    if fid == 4:
        base = 3
    elif fid == 18:
        base = 17
    else:
        base = fid
    return base + 10000 * iid


def _compute_xopt(seed: int, dim: int) -> np.ndarray:
    """Compute x_opt using COCO's ``bbob2009_compute_xopt``.

    Algorithm: generate uniform(0,1) values, transform to [-4, 4]
    with 4 decimal place truncation, replace exact zeros with -1e-5.
    """
    u = _bbob2009_unif(dim, seed)
    xopt = np.empty(dim)
    for i in range(dim):
        xopt[i] = 8.0 * math.floor(1e4 * u[i]) / 1e4 - 4.0
        if xopt[i] == 0.0:
            xopt[i] = -1e-5
    return xopt


def _compute_fopt(fid: int, iid: int) -> float:
    """Compute f_opt using COCO's ``bbob2009_compute_fopt``.

    Cauchy distribution via ratio of two independent Gaussians.
    """
    rseed = _compute_rseed(fid, iid)
    gval = _bbob2009_gauss(1, rseed)[0]
    gval2 = _bbob2009_gauss(1, rseed + 1)[0]
    fopt = round(100.0 * 100.0 * gval / gval2) / 100.0
    return min(1000.0, max(-1000.0, fopt))


def _compute_rotation(dim: int, seed: int) -> np.ndarray:
    """Compute a rotation matrix using COCO's Gram-Schmidt algorithm.

    Steps:
    1. Generate DIM*DIM Gaussian values
    2. Reshape column-major into DIM x DIM matrix
    3. Apply classical Gram-Schmidt orthogonalization on columns
    """
    if dim == 1:
        return np.array([[1.0]])

    gauss = _bbob2009_gauss(dim * dim, seed)

    # Column-major reshape: B[row][col] = gauss[col * dim + row]
    B = np.empty((dim, dim))
    for col in range(dim):
        for row in range(dim):
            B[row, col] = gauss[col * dim + row]

    # Classical Gram-Schmidt on columns
    for i in range(dim):
        for j in range(i):
            prod = np.dot(B[:, i], B[:, j])
            B[:, i] -= prod * B[:, j]
        norm = math.sqrt(np.dot(B[:, i], B[:, i]))
        B[:, i] /= norm

    return B


class BBOBInstanceGenerator:
    """Generates instance-specific parameters for BBOB functions.

    All generation is deterministic given (fid, iid, dim) and produces
    results identical to the COCO reference implementation.
    """

    def generate_xopt(self, fid: int, iid: int, dim: int) -> np.ndarray:
        """Generate the optimal point x_opt.

        Parameters
        ----------
        fid : int
            Function ID (1-24).
        iid : int
            Instance ID.
        dim : int
            Dimension.

        Returns
        -------
        np.ndarray of shape (dim,)
            The optimal point, in [-4, 4]^D.
            Special cases: f5 (boundary ±5), f8 (scaled by 0.75),
            f12 (uses rseed + 1000000).
        """
        rseed = _compute_rseed(fid, iid)

        if fid == 5:
            # Linear slope: x_opt at boundary ±5
            # COCO generates xopt normally, then signs define boundary
            xopt = _compute_xopt(rseed, dim)
            return np.where(xopt >= 0, 5.0, -5.0)

        # COCO uses rseed + 1000000 for xopt of f12 (Bent Cigar)
        xopt_seed = rseed + 1000000 if fid == 12 else rseed
        xopt = _compute_xopt(xopt_seed, dim)

        if fid == 8:
            # Rosenbrock: xopt scaled by 0.75
            xopt *= 0.75

        return xopt

    def generate_fopt(self, fid: int, iid: int) -> float:
        """Generate the optimal function value f_opt.

        Uses the COCO Cauchy distribution (ratio of two Gaussians),
        rounded to 2 decimal places and clamped to [-1000, 1000].

        Parameters
        ----------
        fid : int
            Function ID (1-24).
        iid : int
            Instance ID.

        Returns
        -------
        float
            The optimal function value.
        """
        return _compute_fopt(fid, iid)

    def generate_rotation(self, dim: int, seed: int) -> np.ndarray:
        """Generate a random orthogonal rotation matrix.

        Uses COCO's classical Gram-Schmidt algorithm for exact
        compatibility with the reference implementation.

        Parameters
        ----------
        dim : int
            Matrix dimension.
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        np.ndarray of shape (dim, dim)
            Orthogonal rotation matrix.
        """
        return _compute_rotation(dim, seed)

    def generate_lambda(self, alpha: float, dim: int) -> np.ndarray:
        """Generate diagonal conditioning vector Lambda^alpha.

        Lambda_i = sqrt(condition)^(i/(D-1)) for i = 0..D-1,
        where condition = alpha (matching COCO's convention).

        For COCO compatibility: ``sqrt(condition)^(k/(dim-1))``.

        Parameters
        ----------
        alpha : float
            Conditioning number. alpha=0 gives no conditioning.
        dim : int
            Dimension.

        Returns
        -------
        np.ndarray of shape (dim,)
            The conditioning vector.
        """
        if dim == 1:
            return np.array([1.0])
        exponents = (alpha / 2.0) * np.arange(dim) / (dim - 1)
        # Clamp to avoid float64 overflow (10^308 is max)
        exponents = np.clip(exponents, -300.0, 300.0)
        return np.power(10.0, exponents)

    def generate_instance(self, fid: int, iid: int, dim: int) -> dict:
        """Generate all instance parameters for a given BBOB function.

        Parameters
        ----------
        fid : int
            Function ID (1-24).
        iid : int
            Instance ID.
        dim : int
            Dimension.

        Returns
        -------
        dict
            Keys: xopt, fopt, R, Q (rotation matrices), plus
            function-specific parameters.
        """
        rseed = _compute_rseed(fid, iid)
        xopt = self.generate_xopt(fid, iid, dim)
        fopt = self.generate_fopt(fid, iid)

        # Rotation seeds follow COCO convention:
        # rot1 (R) = rseed + 1000000, rot2 (Q) = rseed
        R = self.generate_rotation(dim, rseed + 1000000)
        Q = self.generate_rotation(dim, rseed)

        return {
            "xopt": xopt,
            "fopt": fopt,
            "R": R,
            "Q": Q,
            "fid": fid,
            "iid": iid,
            "dim": dim,
        }

    def generate_permutation(self, dim: int, seed: int) -> np.ndarray:
        """Generate a truncated swap permutation for large-scale BBOB.

        For D <= 40, returns identity permutation.
        Otherwise, performs nb_swaps = D/3 local swaps, each swapping
        a random position with a nearby position within swap_range = D/3.
        This matches the COCO truncated uniform swap algorithm.

        Parameters
        ----------
        dim : int
            Dimension.
        seed : int
            RNG seed.

        Returns
        -------
        np.ndarray of shape (dim,)
            Permutation array.
        """
        if dim <= 0:
            raise ValueError("Dimension must be a positive integer.")

        if dim <= 40:
            return np.arange(dim)

        rng = np.random.default_rng(seed)
        perm = np.arange(dim)
        swap_range = dim // 3
        nb_swaps = dim // 3

        for _ in range(nb_swaps):
            i = rng.integers(0, dim - 1)
            max_j = min(swap_range, dim - 1 - i)
            j_offset = rng.integers(1, max_j + 1)
            j = i + j_offset
            perm[i], perm[j] = perm[j], perm[i]

        return perm

    def generate_block_diagonal_rotation(
        self, dim: int, seed: int, block_size: int = 40
    ) -> list[np.ndarray]:
        """Generate block-diagonal rotation matrices for large-scale BBOB.

        Parameters
        ----------
        dim : int
            Total dimension.
        seed : int
            Base seed. Each block uses seed + block_index.
        block_size : int
            Maximum block size (default 40).

        Returns
        -------
        list[np.ndarray]
            List of orthogonal rotation matrices.
        """
        if dim <= 0:
            raise ValueError("Dimension must be a positive integer.")
        if block_size <= 0:
            raise ValueError("block_size must be a positive integer.")

        blocks = []
        remaining = dim
        idx = 0
        while remaining > 0:
            s = min(block_size, remaining)
            block = _compute_rotation(s, seed + idx)
            blocks.append(block)
            remaining -= s
            idx += 1
        return blocks
