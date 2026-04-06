"""BBOBNoisySuiteFactory — 30 noisy BBOB functions (f101-f130).

10 base BBOB functions x 3 COCO noise models.
f101-f106: moderate noise (Sphere + Rosenbrock x 3 noise types)
f107-f130: severe noise (8 bases x 3 noise types)
"""

from __future__ import annotations

from typing import Any

from pyMOFL.factories.bbob_suite_factory import BBOBSuiteFactory
from pyMOFL.functions.transformations.cauchy_noise import CauchyNoiseTransform
from pyMOFL.functions.transformations.composed import ComposedFunction
from pyMOFL.functions.transformations.gaussian_noise import GaussianNoiseTransform
from pyMOFL.functions.transformations.uniform_noise import UniformNoiseTransform

# Noise type enum
_GAUSSIAN = "gaussian"
_UNIFORM = "uniform"
_CAUCHY = "cauchy"

# Mapping: noisy_fid -> (base_bbob_fid, noise_type, noise_params)
# f101-f106: moderate noise
# f107-f130: severe noise (8 bases x 3 noise types)
_NOISY_FUNCTIONS: dict[int, dict[str, Any]] = {}

# --- Moderate noise (f101-f106) ---
# Sphere-based
_NOISY_FUNCTIONS[101] = {
    "name": "Sphere + Moderate Gaussian",
    "base_fid": 1,
    "noise_type": _GAUSSIAN,
    "noise_params": {"beta": 0.01},
}
_NOISY_FUNCTIONS[102] = {
    "name": "Sphere + Moderate Uniform",
    "base_fid": 1,
    "noise_type": _UNIFORM,
    "noise_params": {"alpha_base": 0.01, "beta": 0.01},
}
_NOISY_FUNCTIONS[103] = {
    "name": "Sphere + Moderate Cauchy",
    "base_fid": 1,
    "noise_type": _CAUCHY,
    "noise_params": {"alpha": 0.01, "p": 0.05},
}
# Rosenbrock-based
_NOISY_FUNCTIONS[104] = {
    "name": "Rosenbrock + Moderate Gaussian",
    "base_fid": 8,
    "noise_type": _GAUSSIAN,
    "noise_params": {"beta": 0.01},
}
_NOISY_FUNCTIONS[105] = {
    "name": "Rosenbrock + Moderate Uniform",
    "base_fid": 8,
    "noise_type": _UNIFORM,
    "noise_params": {"alpha_base": 0.01, "beta": 0.01},
}
_NOISY_FUNCTIONS[106] = {
    "name": "Rosenbrock + Moderate Cauchy",
    "base_fid": 8,
    "noise_type": _CAUCHY,
    "noise_params": {"alpha": 0.01, "p": 0.05},
}

# --- Severe noise (f107-f130): 8 bases x 3 noise types ---
_SEVERE_BASES = [
    (1, "Sphere"),
    (2, "Ellipsoidal"),
    (8, "Rosenbrock"),
    (3, "Rastrigin"),
    (15, "Rastrigin (rotated)"),
    (10, "Ellipsoidal (rotated)"),
    (20, "Schwefel"),
    (21, "Gallagher 101-Peaks"),
]

_SEVERE_NOISE_TYPES = [
    (_GAUSSIAN, "Gaussian", {"beta": 1.0}),
    (_UNIFORM, "Uniform", {"alpha_base": 1.0, "beta": 1.0}),
    (_CAUCHY, "Cauchy", {"alpha": 1.0, "p": 0.2}),
]

for i, (base_fid, base_name) in enumerate(_SEVERE_BASES):
    for j, (noise_type, noise_name, noise_params) in enumerate(_SEVERE_NOISE_TYPES):
        fid = 107 + i * 3 + j
        _NOISY_FUNCTIONS[fid] = {
            "name": f"{base_name} + Severe {noise_name}",
            "base_fid": base_fid,
            "noise_type": noise_type,
            "noise_params": dict(noise_params),
        }


def _make_noise_transform(
    noise_type: str, noise_params: dict[str, Any], dim: int
) -> GaussianNoiseTransform | UniformNoiseTransform | CauchyNoiseTransform:
    """Create the appropriate noise transform."""
    if noise_type == _GAUSSIAN:
        return GaussianNoiseTransform(beta=noise_params["beta"])
    if noise_type == _UNIFORM:
        alpha = noise_params["alpha_base"] * (0.49 + 1.0 / dim)
        return UniformNoiseTransform(alpha=alpha, beta=noise_params["beta"])
    if noise_type == _CAUCHY:
        return CauchyNoiseTransform(alpha=noise_params["alpha"], p=noise_params["p"])
    raise ValueError(f"Unknown noise type: {noise_type}")


class BBOBNoisySuiteFactory:
    """Factory for 30 noisy BBOB functions (f101-f130).

    Wraps standard BBOBSuiteFactory, adding a noise output transform.

    Parameters
    ----------
    bbob_factory : BBOBSuiteFactory, optional
        Base BBOB factory. Created if not provided.
    """

    def __init__(self, bbob_factory: BBOBSuiteFactory | None = None):
        self._bbob = bbob_factory or BBOBSuiteFactory()

    def create_function(self, fid: int, iid: int, dim: int) -> ComposedFunction:
        """Create a single noisy BBOB function.

        Parameters
        ----------
        fid : int
            Noisy function ID (101-130).
        iid : int
            Instance ID.
        dim : int
            Dimension.

        Returns
        -------
        ComposedFunction
            BBOB function with noise output transform appended.
        """
        if fid not in _NOISY_FUNCTIONS:
            raise ValueError(f"Unknown noisy BBOB function ID: {fid}. Must be 101-130.")

        info = _NOISY_FUNCTIONS[fid]
        base_fid = info["base_fid"]
        noise_type = info["noise_type"]
        noise_params = info["noise_params"]

        # Build the base BBOB function
        base_func = self._bbob.create_function(base_fid, iid, dim)

        # Create noise transform and append to output transforms
        noise_transform = _make_noise_transform(noise_type, noise_params, dim)
        output_transforms = [*base_func.output_transforms, noise_transform]

        return ComposedFunction(
            base_function=base_func.base_function,
            input_transforms=list(base_func.input_transforms),
            output_transforms=output_transforms,
            penalty_transforms=list(base_func.penalty_transforms),
        )

    def create_suite(self, iid: int, dim: int) -> list[ComposedFunction]:
        """Create all 30 noisy BBOB functions."""
        return [self.create_function(fid, iid, dim) for fid in range(101, 131)]

    def get_function_info(self, fid: int) -> dict[str, Any]:
        """Get metadata for a noisy function."""
        if fid not in _NOISY_FUNCTIONS:
            raise ValueError(f"Unknown noisy BBOB function ID: {fid}. Must be 101-130.")
        return dict(_NOISY_FUNCTIONS[fid])
