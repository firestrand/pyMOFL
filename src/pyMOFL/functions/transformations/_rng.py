"""Shared constructor contract for explicitly supplied NumPy noise streams."""

import numpy as np


def resolve_generator(seed: int | None, rng: np.random.Generator | None) -> np.random.Generator:
    """Resolve a local stream without changing the process-global RNG."""
    if rng is not None:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator")
        if seed is not None:
            raise ValueError("Specify either seed or rng, not both")
        return rng
    return np.random.default_rng(seed)
