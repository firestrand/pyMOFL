"""
Noise: Output-transforming decorator for adding noise to an OptimizationFunction.

Inherits from OutputTransformingFunction. Subclasses should only implement _apply and _apply_batch, never override evaluate or evaluate_batch.

Usage:
    # As a decorator
    base = SphereFunction(...)
    f = Noise(base_function=base, noise_type='gaussian', noise_level=0.1)
    value = f(x)
"""

import numpy as np
from pyMOFL.core.composable_function import OutputTransformingFunction

class Noise(OutputTransformingFunction):
    def __init__(self, base_function=None, dimension=None, noise_type='gaussian', noise_level=0.1, noise_seed=None, initialization_bounds=None, operational_bounds=None):
        self.noise_type = noise_type.lower()
        self.noise_level = noise_level
        if noise_seed is not None:
            np.random.seed(noise_seed)
        if self.noise_type not in ['gaussian', 'uniform']:
            raise ValueError(f"Unsupported noise type: {noise_type}. Use 'gaussian' or 'uniform'.")
        super().__init__(base_function=base_function, dimension=dimension, initialization_bounds=initialization_bounds, operational_bounds=operational_bounds)

    def _generate_noise(self, size=None):
        if self.noise_type == 'gaussian':
            if size is not None:
                return 1.0 + self.noise_level * np.abs(np.random.normal(size=size))
            else:
                return 1.0 + self.noise_level * np.abs(np.random.normal())
        elif self.noise_type == 'uniform':
            if size is not None:
                return 1.0 + self.noise_level * np.random.uniform(size=size)
            else:
                return 1.0 + self.noise_level * np.random.uniform()

    def _apply(self, y):
        y = np.asarray(y)
        noise = self._generate_noise(size=y.shape) if y.shape != () else self._generate_noise()
        return y * noise

    def _apply_batch(self, Y):
        Y = np.asarray(Y)
        noise = self._generate_noise(size=Y.shape[0])
        return Y * noise