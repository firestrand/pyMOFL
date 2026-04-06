"""
Rastrigin function implementation.

The Rastrigin function is a non-convex multimodal benchmark function with many local minima.
It is often used to test the ability of optimization algorithms to escape local optima.

References:
    .. [1] Rastrigin, L.A. (1974). "Systems of extremal control". Mir, Moscow.
    .. [2] Kumar, K.E.S., et al. (2024). "Benchmarking of GPU-optimized Quantum-Inspired
           Evolutionary Optimization Algorithm using Functional Analysis". arXiv:2412.08992
           Local documentation: docs/literature_rastrigin/kumar_2024_gpu_benchmarking.md
    .. [3] Mühlenbein, H., Schomisch, D., & Born, J. (1991). "The parallel genetic algorithm as function
           optimizer". Parallel Computing, 17(6-7), 619-632.
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Rastrigin")
@register("rastrigin")
class RastriginFunction(OptimizationFunction):
    """
    Rastrigin function: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))

    Global minimum: f(0, 0, ..., 0) = 0

    Attributes:
        dimension (int): The dimensionality of the function.
        bounds (np.ndarray): Default bounds are [-5.12, 5.12] for each dimension.

    References:
        .. [1] Rastrigin, L.A. (1974). "Systems of extremal control". Mir, Moscow.
        .. [2] Kumar, K.E.S., et al. (2024). "Benchmarking of GPU-optimized Quantum-Inspired
               Evolutionary Optimization Algorithm using Functional Analysis". arXiv:2412.08992
               Local documentation: docs/literature_rastrigin/kumar_2024_gpu_benchmarking.md
        .. [3] Mühlenbein, H., Schomisch, D., & Born, J. (1991). "The parallel genetic algorithm as function
               optimizer". Parallel Computing, 17(6-7), 619-632.

    Note:
        To add a bias to the function, use the BiasWrapper from the transformations module.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        """
        Initialize the Rastrigin function.

        Args:
            dimension (int): The dimensionality of the function.
            input_transforms (List[Transform], optional): Input transforms to apply before computing.
            initialization_bounds (Bounds, optional): Bounds for initialization.
                                                    Defaults to [-5.12, 5.12] for each dimension.
            operational_bounds (Bounds, optional): Bounds for operational use.
                                                  Defaults to [-5.12, 5.12] for each dimension.
        """
        default_bounds = Bounds(
            low=np.full(dimension, -5.12),
            high=np.full(dimension, 5.12),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the Rastrigin function value."""
        x = self._validate_input(x)
        return float(np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Rastrigin function for batch."""
        X = self._validate_batch_input(X)
        return np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """
        Get the global minimum of the function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        global_min_point = np.zeros(self.dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value
