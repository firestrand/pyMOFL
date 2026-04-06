"""
BBOB Noiseless Suite Factory.

Programmatic factory for the 24 BBOB noiseless functions (Hansen et al. 2009).
Unlike CEC 2005 (static data files + JSON config), BBOB uses seeded random
parameter generation. This factory generates config dicts on-the-fly and passes
them to the existing FunctionFactory.

References
----------
.. [1] Hansen, N., et al. (2009). "Real-parameter black-box optimization
       benchmarking 2009: Noiseless functions definitions."
       INRIA Technical Report RR-6829.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyMOFL.factories.function_factory import FunctionFactory
from pyMOFL.functions.transformations.composed import ComposedFunction
from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

# BBOB function metadata: name, category, base_function type
_BBOB_INFO: dict[int, dict[str, str]] = {
    1: {"name": "Sphere", "category": "separable", "base_function": "sphere"},
    2: {"name": "Ellipsoidal (separable)", "category": "separable", "base_function": "elliptic"},
    3: {"name": "Rastrigin (separable)", "category": "separable", "base_function": "rastrigin"},
    4: {"name": "Büche-Rastrigin", "category": "separable", "base_function": "buche_rastrigin"},
    5: {"name": "Linear Slope", "category": "separable", "base_function": "linear_slope"},
    6: {"name": "Attractive Sector", "category": "moderate", "base_function": "attractive_sector"},
    7: {"name": "Step Ellipsoidal", "category": "moderate", "base_function": "step_ellipsoid"},
    8: {"name": "Rosenbrock (original)", "category": "moderate", "base_function": "rosenbrock"},
    9: {"name": "Rosenbrock (rotated)", "category": "moderate", "base_function": "rosenbrock"},
    10: {
        "name": "Ellipsoidal (rotated)",
        "category": "ill-conditioned",
        "base_function": "elliptic",
    },
    11: {"name": "Discus", "category": "ill-conditioned", "base_function": "discus"},
    12: {"name": "Bent Cigar", "category": "ill-conditioned", "base_function": "bent_cigar"},
    13: {"name": "Sharp Ridge", "category": "ill-conditioned", "base_function": "sharp_ridge"},
    14: {
        "name": "Different Powers",
        "category": "ill-conditioned",
        "base_function": "different_powers",
    },
    15: {"name": "Rastrigin (rotated)", "category": "multimodal", "base_function": "rastrigin"},
    16: {"name": "Weierstrass", "category": "multimodal", "base_function": "weierstrass"},
    17: {"name": "Schaffer F7", "category": "multimodal", "base_function": "schaffer_f7"},
    18: {
        "name": "Schaffer F7 (ill-cond)",
        "category": "multimodal",
        "base_function": "schaffer_f7",
    },
    19: {
        "name": "Griewank-Rosenbrock",
        "category": "multimodal",
        "base_function": "griewank_of_rosenbrock",
    },
    20: {"name": "Schwefel", "category": "weakly-structured", "base_function": "schwefel_sin"},
    21: {
        "name": "Gallagher 101 Peaks",
        "category": "weakly-structured",
        "base_function": "gallagher_peaks",
    },
    22: {
        "name": "Gallagher 21 Peaks",
        "category": "weakly-structured",
        "base_function": "gallagher_peaks",
    },
    23: {"name": "Katsuura", "category": "weakly-structured", "base_function": "katsuura"},
    24: {
        "name": "Lunacek bi-Rastrigin",
        "category": "weakly-structured",
        "base_function": "lunacek",
    },
}


class BBOBSuiteFactory:
    """Factory for creating BBOB noiseless benchmark functions.

    Generates instance parameters on-the-fly via seeded RNG, builds
    config dicts, and delegates to FunctionFactory for construction.
    """

    def __init__(
        self,
        instance_generator: BBOBInstanceGenerator | None = None,
        function_factory: FunctionFactory | None = None,
    ):
        self._gen = instance_generator or BBOBInstanceGenerator()
        self._factory = function_factory or FunctionFactory()

    def create_function(self, fid: int, iid: int, dim: int) -> ComposedFunction:
        """Create a single BBOB function instance.

        Parameters
        ----------
        fid : int
            Function ID (1-24).
        iid : int
            Instance ID (determines xopt, fopt, rotations).
        dim : int
            Dimension.

        Returns
        -------
        ComposedFunction
            The fully configured BBOB function.
        """
        if fid < 1 or fid > 24:
            raise ValueError(f"Unknown BBOB function ID: {fid}. Must be 1-24.")

        config = self.build_config(fid, iid, dim)
        return self._factory.create_function(config)

    def create_suite(self, iid: int, dim: int) -> list[ComposedFunction]:
        """Create all 24 BBOB functions for a given instance and dimension."""
        return [self.create_function(fid, iid, dim) for fid in range(1, 25)]

    def get_function_info(self, fid: int) -> dict[str, str]:
        """Get metadata for a BBOB function.

        Parameters
        ----------
        fid : int
            Function ID (1-24).

        Returns
        -------
        dict with keys: name, category, base_function
        """
        if fid not in _BBOB_INFO:
            raise ValueError(f"Unknown BBOB function ID: {fid}")
        return dict(_BBOB_INFO[fid])

    def build_config(self, fid: int, iid: int = 1, dim: int = 10) -> dict[str, Any]:
        """Build nested config dict for a given BBOB function.

        Parameters
        ----------
        fid : int
            Function ID (1-24).
        iid : int
            Instance ID (determines xopt, fopt, rotations).
        dim : int
            Dimension.

        Returns
        -------
        dict
            Nested config compatible with ConfigParser / FunctionFactory.
        """
        if fid < 1 or fid > 24:
            raise ValueError(f"Unknown BBOB function ID: {fid}. Must be 1-24.")
        params = self._gen.generate_instance(fid, iid, dim)
        return self._build_config_from_params(fid, params)

    def _build_config_from_params(self, fid: int, params: dict[str, Any]) -> dict[str, Any]:
        """Build nested config dict from pre-generated instance params."""
        if fid <= 5:
            return self._build_separable(fid, params)
        if fid <= 9:
            return self._build_moderate(fid, params)
        if fid <= 14:
            return self._build_ill_conditioned(fid, params)
        if fid <= 19:
            return self._build_multimodal(fid, params)
        return self._build_weakly_structured(fid, params)

    def _lambda_scale_factor(self, alpha: float, dim: int) -> list[float]:
        """Compute reciprocal Lambda for ScaleTransform.

        ScaleTransform divides by factor, so to apply Lambda multiplication
        we pass 1/Lambda. Computed directly via negated exponents to avoid overflow.

        Exponents are clamped to [-150, 150] to keep values in float64 safe range
        even after squaring (10^150 squared = 10^300, within float64 max ~10^308).
        """
        if dim == 1:
            return [1.0]
        exponents = -(alpha / 2.0) * np.arange(dim) / (dim - 1)
        # Clamp to keep x*Lambda values safe for squaring in downstream functions
        exponents = np.clip(exponents, -150.0, 150.0)
        return np.power(10.0, exponents).tolist()

    @staticmethod
    def _build_affine_matrix(
        R1: np.ndarray, R2: np.ndarray, conditioning: float, dim: int
    ) -> np.ndarray:
        """Compute affine matrix M = R1 @ diag(sqrt(cond)^(k/(D-1))) @ R2.

        Matches COCO's affine matrix construction for rotated functions.
        """
        if dim == 1:
            return R1.copy()
        lam = np.power(np.sqrt(conditioning), np.arange(dim) / (dim - 1))
        return R1 @ np.diag(lam) @ R2

    # --- Config builders by category ---

    @staticmethod
    def _wrap(base_config: dict, transforms: list[tuple[str, dict]]) -> dict:
        """Wrap base config with transform layers.

        Parameters
        ----------
        base_config : dict
            The base function config (innermost).
        transforms : list of (type, params) tuples
            In application order: first = innermost (applied first to x).
        """
        config = base_config
        for ttype, tparams in transforms:
            config = {"type": ttype, "parameters": tparams, "function": config}
        return config

    def _build_separable(self, fid: int, params: dict) -> dict:
        dim = params["dim"]
        xopt = params["xopt"]
        fopt = params["fopt"]
        base_type = _BBOB_INFO[fid]["base_function"]

        base = {"type": base_type, "parameters": {"dimension": dim}}
        transforms: list[tuple[str, dict]] = []

        if fid == 1:
            # f1 Sphere: shift(xopt) → sphere + bias
            # COCO: no oscillation, no penalty
            transforms.append(("shift", {"vector": xopt.tolist()}))
        elif fid == 2:
            # f2 Ellipsoidal: shift(xopt) → T_osz → elliptic + bias
            # COCO: no penalty
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("t_osz", {}))
        elif fid == 3:
            # f3 Rastrigin: shift(xopt) → T_osz → T_asy^0.2 → Lambda^10 → rastrigin + bias
            # COCO: no penalty
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("t_osz", {}))
            transforms.append(("t_asy", {"beta": 0.2}))
            transforms.append(("scale", {"factor": self._lambda_scale_factor(10.0, dim)}))
        elif fid == 4:
            # f4 Büche-Rastrigin: shift(xopt) → T_osz → brs → buche_rastrigin + f_pen(100) + bias
            # COCO: has penalty with factor=100; oscillation and brs in base function
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("f_pen", {}))
        elif fid == 5:
            # f5 Linear Slope: sign_vector from xopt, no shift transform
            # COCO: xopt = sign(xopt_raw) * 5, s_i = sign_i * sqrt(10)^(i/(D-1))
            sign_vector = np.sign(xopt).tolist()
            base = {
                "type": base_type,
                "parameters": {
                    "dimension": dim,
                    "sign_vector": sign_vector,
                },
            }

        transforms.append(("bias", {"value": fopt}))

        return self._wrap(base, transforms)

    def _build_moderate(self, fid: int, params: dict) -> dict:
        dim = params["dim"]
        xopt = params["xopt"]
        fopt = params["fopt"]
        R = params["R"]
        Q = params["Q"]
        base_type = _BBOB_INFO[fid]["base_function"]

        base = {"type": base_type, "parameters": {"dimension": dim}}
        transforms: list[tuple[str, dict]] = []

        # NOTE: RotateTransform applies M.T @ x (CEC convention).
        # COCO wants M @ x, so we pass M.T to get (M.T).T @ x = M @ x.

        if fid == 6:
            # f6 Attractive Sector: shift(xopt) → affine(R1*L*R2) → attractive_sector
            #   + output: T_osz(y) → y^0.9 → +fopt
            # COCO: no penalty. Output transforms not yet supported (TODO).
            M = self._build_affine_matrix(R, Q, 10.0, dim)
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("rotate", {"matrix": M.T.tolist()}))
            transforms.append(("bias", {"value": fopt}))
        elif fid == 7:
            # f7 Step Ellipsoidal: MONOLITHIC in COCO (all transforms internal)
            # Approximation: shift → Q → Lambda^10 → R → step_ellipsoid + f_pen + bias
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("rotate", {"matrix": Q.T.tolist()}))
            transforms.append(("scale", {"factor": self._lambda_scale_factor(10.0, dim)}))
            transforms.append(("rotate", {"matrix": R.T.tolist()}))
            transforms.append(("f_pen", {}))
            transforms.append(("bias", {"value": fopt}))
        elif fid == 8:
            # f8 Rosenbrock (original): shift(xopt) → scale(factor) → shift(-1) → rosenbrock + bias
            # COCO: factor = max(1, sqrt(D)/8), shift(-1) adds 1 to each component
            # so optimum at (1,...,1) maps back to xopt. No penalty. No rotation.
            scale_factor = max(1.0, np.sqrt(dim) / 8.0)
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("scale", {"factor": 1.0 / scale_factor}))
            transforms.append(("shift", {"vector": (-np.ones(dim)).tolist()}))
            transforms.append(("bias", {"value": fopt}))
        elif fid == 9:
            # f9 Rosenbrock (rotated): affine(factor*R, b=0.5) → rosenbrock + bias
            # COCO: M = factor*R, b = (0.5,...), no shift by xopt. No penalty.
            scale_factor = max(1.0, np.sqrt(dim) / 8.0)
            M = scale_factor * R
            transforms.append(("rotate", {"matrix": M.T.tolist()}))
            transforms.append(("shift", {"vector": [-0.5] * dim}))
            transforms.append(("bias", {"value": fopt}))

        return self._wrap(base, transforms)

    def _build_ill_conditioned(self, fid: int, params: dict) -> dict:
        dim = params["dim"]
        xopt = params["xopt"]
        fopt = params["fopt"]
        R = params["R"]  # rotation(rseed + 1e6)
        Q = params["Q"]  # rotation(rseed)
        base_type = _BBOB_INFO[fid]["base_function"]

        base = {"type": base_type, "parameters": {"dimension": dim}}
        transforms: list[tuple[str, dict]] = []

        # NOTE: RotateTransform applies M.T @ x (CEC convention).
        # COCO wants M @ x, so we pass M.T to get (M.T).T @ x = M @ x.

        if fid == 10:
            # f10 Ellipsoidal (rotated): shift(xopt) → R1 → T_osz → elliptic + bias
            # COCO: just R1 rotation, no affine conditioning (base has 1e6 conditioning).
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("rotate", {"matrix": R.T.tolist()}))
            transforms.append(("t_osz", {}))
        elif fid == 11:
            # f11 Discus: shift(xopt) → R1 → T_osz → discus + bias
            # COCO: just R1 rotation.
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("rotate", {"matrix": R.T.tolist()}))
            transforms.append(("t_osz", {}))
        elif fid == 12:
            # f12 Bent Cigar: shift(xopt) → R1 → T_asy^0.5 → R1 → bent_cigar + bias
            # COCO: R1 used twice (not R1 then Q).
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("rotate", {"matrix": R.T.tolist()}))
            transforms.append(("t_asy", {"beta": 0.5}))
            transforms.append(("rotate", {"matrix": R.T.tolist()}))
        elif fid == 13:
            # f13 Sharp Ridge: shift(xopt) → affine(R1*Lambda_sqrt10*R2) → sharp_ridge + bias
            # COCO: M = R1 * diag(sqrt(10)^(k/(D-1))) * R2.
            M = self._build_affine_matrix(R, Q, 10.0, dim)
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("rotate", {"matrix": M.T.tolist()}))
        elif fid == 14:
            # f14 Different Powers: shift(xopt) → R1 → different_powers + bias
            # COCO: just R1 rotation.
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("rotate", {"matrix": R.T.tolist()}))

        transforms.append(("bias", {"value": fopt}))
        return self._wrap(base, transforms)

    def _build_multimodal(self, fid: int, params: dict) -> dict:
        dim = params["dim"]
        xopt = params["xopt"]
        fopt = params["fopt"]
        R = params["R"]  # rotation(rseed + 1e6) = R1
        Q = params["Q"]  # rotation(rseed) = R2
        base_type = _BBOB_INFO[fid]["base_function"]

        base = {"type": base_type, "parameters": {"dimension": dim}}
        transforms: list[tuple[str, dict]] = []

        # NOTE: RotateTransform applies M.T @ x (CEC convention).
        # COCO wants M @ x, so we pass M.T to get (M.T).T @ x = M @ x.

        if fid == 15:
            # f15 Rastrigin (rotated):
            # COCO: shift(xopt) → R1 → T_osz → T_asy^0.2 → affine(R1*L10*R2) → rastrigin + bias
            # No penalty.
            M = self._build_affine_matrix(R, Q, 10.0, dim)
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("rotate", {"matrix": R.T.tolist()}))
            transforms.append(("t_osz", {}))
            transforms.append(("t_asy", {"beta": 0.2}))
            transforms.append(("rotate", {"matrix": M.T.tolist()}))
        elif fid == 16:
            # f16 Weierstrass:
            # COCO: shift(xopt) → R1 → T_osz → affine(R1*L100*R2) → weierstrass + f_pen + bias
            M = self._build_affine_matrix(R, Q, 100.0, dim)
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("rotate", {"matrix": R.T.tolist()}))
            transforms.append(("t_osz", {}))
            transforms.append(("rotate", {"matrix": M.T.tolist()}))
            transforms.append(("f_pen", {}))
        elif fid == 17:
            # f17 Schaffer F7 (condition 10):
            # COCO: shift(xopt) → R1 → T_asy^0.5 → Lambda^10 → Q → schaffer_f7 + f_pen + bias
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("rotate", {"matrix": R.T.tolist()}))
            transforms.append(("t_asy", {"beta": 0.5}))
            transforms.append(("scale", {"factor": self._lambda_scale_factor(10.0, dim)}))
            transforms.append(("rotate", {"matrix": Q.T.tolist()}))
            transforms.append(("f_pen", {}))
        elif fid == 18:
            # f18 Schaffer F7 (condition 1000):
            # Same as f17 but with conditioning=1000. COCO: f_pen.
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("rotate", {"matrix": R.T.tolist()}))
            transforms.append(("t_asy", {"beta": 0.5}))
            transforms.append(("scale", {"factor": self._lambda_scale_factor(1000.0, dim)}))
            transforms.append(("rotate", {"matrix": Q.T.tolist()}))
            transforms.append(("f_pen", {}))
        elif fid == 19:
            # f19 Griewank-Rosenbrock:
            # COCO: affine(factor*R, b=0) → shift(-0.5) → griewank_rosenbrock + bias
            # factor = max(1, sqrt(D)/8), R = rotation(rseed). No penalty.
            scale_factor = max(1.0, np.sqrt(dim) / 8.0)
            M = scale_factor * Q  # Q = rotation(rseed) for f19
            transforms.append(("rotate", {"matrix": M.T.tolist()}))
            transforms.append(("shift", {"vector": [-0.5] * dim}))

        transforms.append(("bias", {"value": fopt}))
        return self._wrap(base, transforms)

    def _build_weakly_structured(self, fid: int, params: dict) -> dict:
        dim = params["dim"]
        xopt = params["xopt"]
        fopt = params["fopt"]
        R = params["R"]
        Q = params["Q"]
        base_type = _BBOB_INFO[fid]["base_function"]

        base = {"type": base_type, "parameters": {"dimension": dim}}
        transforms: list[tuple[str, dict]] = []

        # NOTE: RotateTransform applies M.T @ x (CEC convention).
        # COCO wants M @ x, so we pass M.T to get (M.T).T @ x = M @ x.

        if fid == 20:
            # f20 Schwefel sin: complex COCO chain (x_hat, z_hat, conditioning)
            # All var transforms and penalty are internal/monolithic in COCO.
            # Approximation: shift(xopt) → schwefel_sin + bias
            transforms.append(("shift", {"vector": xopt.tolist()}))
        elif fid == 21:
            # f21 Gallagher 101 peaks: all internal in COCO (rotation, peaks,
            # oscillation, penalty). Only external wrapper is fopt shift.
            pass
        elif fid == 22:
            # f22 Gallagher 21 peaks: same as f21 structure
            pass
        elif fid == 23:
            # f23 Katsuura: shift(xopt) → affine(R1*Lambda100*R2) → katsuura
            #   + f_pen(1.0) + bias. Penalty is EXTERNAL in COCO.
            M = self._build_affine_matrix(R, Q, 100.0, dim)
            transforms.append(("shift", {"vector": xopt.tolist()}))
            transforms.append(("rotate", {"matrix": M.T.tolist()}))
            transforms.append(("f_pen", {}))
        elif fid == 24:
            # f24 Lunacek bi-Rastrigin: all internal in COCO (x_hat, affine,
            # two basins, penalty*1e4). Only external wrapper is fopt shift.
            pass

        transforms.append(("bias", {"value": fopt}))
        return self._wrap(base, transforms)
