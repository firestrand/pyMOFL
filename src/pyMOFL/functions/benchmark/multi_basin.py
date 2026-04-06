"""
Multi-basin benchmark function generator.

Implements a generalized parametric baseline function for creating
controllable optimization landscapes by composing multiple basins.
This follows the baseline mathematical formulation described in the GNBG framework.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pyMOFL.compositions.min_composition import MinComposition
from pyMOFL.functions.transformations import (
    BiasTransform,
    ComposedFunction,
    LogSinTransform,
    PowerTransform,
    RotateTransform,
    ScaleTransform,
    ShiftTransform,
)
from pyMOFL.registry import register
from pyMOFL.utils.rotation import build_rotation_from_theta, random_rotation_matrix

from .sphere import SphereFunction


def create_basin_component(
    dimension: int,
    center: NDArray,
    rotation: NDArray | None = None,
    conditioning: NDArray | float = 1.0,
    linearity: float = 1.0,
    bias: float = 0.0,
    mu: tuple[float, float] = (0.0, 0.0),
    omega: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> ComposedFunction:
    """
    Create a single basin component (basin of attraction).

    Mathematical form:
    Psi(x) = bias + ( sum_i ( H_i * T( z_i ) )^2 )^linearity
    where z = R(x - center).

    Parameters
    ----------
    dimension : int
        Dimensionality of the function.
    center : NDArray
        Position of the basin minimum.
    rotation : NDArray, optional
        Orthogonal rotation matrix. If None, identity is used.
    conditioning : NDArray or float, optional
        Diagonal scaling elements (conditioning matrix H).
    linearity : float, optional
        The degree of linearity (lambda). 1.0 for quadratic, 0.5 for linear.
    bias : float, optional
        The minimum value of this component.
    mu : tuple[float, float], optional
        Multipliers for the log-sin nonlinear transform.
    omega : tuple[float, float, float, float], optional
        Frequencies for the log-sin nonlinear transform.
    """
    base = SphereFunction(dimension=dimension)

    input_transforms = [ShiftTransform(center)]
    if rotation is not None:
        input_transforms.append(RotateTransform(rotation))

    # Apply log-sin ruggedness/asymmetry if parameters are non-zero
    if np.any(mu) or np.any(omega):
        input_transforms.append(LogSinTransform(mu=mu, omega=omega))

    # Diagonal conditioning
    input_transforms.append(ScaleTransform(conditioning))

    output_transforms = [
        PowerTransform(linearity),
        BiasTransform(bias),
    ]

    return ComposedFunction(
        base_function=base,
        input_transforms=input_transforms,
        output_transforms=output_transforms,
    )


@register("GNBG")
@register("multi_basin")
class MultiBasinFunction(MinComposition):
    """
    Generalized Multi-Basin generator (GNBG Baseline).

    f(x) = min_k ( Psi_k(x) )

    Allows for the construction of diverse optimization landscapes with
    controllable modality, ruggedness, symmetry, and conditioning.
    """

    def __init__(
        self,
        dimension: int,
        n_components: int = 1,
        seed: int | None = None,
        **kwargs,
    ):
        """
        Initialize the multi-basin function.

        Parameters
        ----------
        dimension : int
            Dimensionality of the function.
        n_components : int
            Number of components (basins).
        seed : int, optional
            Seed for random parameter generation.
        centers : NDArray, optional
            (n_components, dimension) basin centers.
        biases : NDArray, optional
            (n_components,) minimum values.
        linearities : NDArray, optional
            (n_components,) linearity values.
        rotations : list[NDArray], optional
            List of n_components rotation matrices.
        thetas : list[NDArray], optional
            List of n_components interaction matrices (upper triangular).
        conditionings : list[NDArray | float], optional
            List of n_components conditioning factors.
        mus : list[tuple[float, float]], optional
            List of n_components mu ruggedness parameters.
        omegas : list[tuple[float, float, float, float]], optional
            List of n_components omega ruggedness parameters.
        """
        rng = np.random.default_rng(seed)

        centers = kwargs.get("centers")
        if centers is None:
            centers = rng.uniform(-4.0, 4.0, size=(n_components, dimension))

        biases = kwargs.get("biases")
        if biases is None:
            biases = rng.uniform(0.0, 100.0, size=n_components)
            biases[0] = 0.0  # Default global minimum

        linearities = kwargs.get("linearities")
        if linearities is None:
            linearities = np.full(n_components, 1.0)  # Default quadratic

        components = []
        for i in range(n_components):
            # Rotation resolution: rotations > thetas > random
            rotation = None
            if "rotations" in kwargs:
                rotation = kwargs["rotations"][i]
            elif "thetas" in kwargs:
                rotation = build_rotation_from_theta(dimension, kwargs["thetas"][i])
            elif kwargs.get("rotate", True):
                rotation = random_rotation_matrix(dimension, rng)

            # Conditioning
            cond = 1.0
            if "conditionings" in kwargs:
                cond = kwargs["conditionings"][i]
            elif kwargs.get("random_conditioning", False):
                alpha = rng.uniform(0, 3, size=dimension)
                cond = 10**alpha

            # Ruggedness
            mu = (0.0, 0.0)
            if "mus" in kwargs:
                mu = kwargs["mus"][i]
            elif kwargs.get("rugged", False):
                mu = (rng.uniform(0.1, 1.0), rng.uniform(0.1, 1.0))

            omega = (0.0, 0.0, 0.0, 0.0)
            if "omegas" in kwargs:
                omega = kwargs["omegas"][i]
            elif kwargs.get("rugged", False):
                omega = tuple(rng.uniform(5.0, 100.0, size=4))

            comp = create_basin_component(
                dimension=dimension,
                center=centers[i],
                rotation=rotation,
                conditioning=cond,
                linearity=linearities[i],
                bias=biases[i],
                mu=mu,
                omega=omega,
            )
            components.append(comp)

        super().__init__(dimension=dimension, components=components)
