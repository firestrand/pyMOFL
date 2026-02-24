"""
Composed function that chains transformations with optimization functions.

Allows building compositions like: bias(sphere(shift(x)))
"""

import numpy as np

from pyMOFL.core.function import OptimizationFunction

from .base import ScalarTransform, VectorTransform


class ComposedFunction(OptimizationFunction):
    """
    A composed function that chains transformations with an optimization function.

    Applies vector transformations to input, evaluates base function,
    then applies scalar transformations to output.

    Example: bias(sphere(shift(rotate(x))))
    - rotate and shift are vector transforms (applied to input)
    - sphere is the base optimization function
    - bias is a scalar transform (applied to output)
    """

    def __init__(
        self,
        base_function: OptimizationFunction,
        input_transforms: list[VectorTransform] | None = None,
        output_transforms: list[ScalarTransform] | None = None,
    ):
        """
        Initialize composed function.

        Args:
            base_function: The optimization function to evaluate
            input_transforms: List of vector transforms to apply to input (in order)
            output_transforms: List of scalar transforms to apply to output (in order)
        """
        super().__init__(
            dimension=base_function.dimension,
            initialization_bounds=base_function.initialization_bounds,
            operational_bounds=base_function.operational_bounds,
        )

        self.base_function = base_function
        self.input_transforms = input_transforms or []
        self.output_transforms = output_transforms or []

        # Set component function for any normalize transforms that need it
        for transform in self.output_transforms:
            if hasattr(transform, "set_component_function"):
                # Build the function that the normalize transform should evaluate
                # This is the base function with all input transforms applied
                from .normalize import NormalizeTransform

                if isinstance(transform, NormalizeTransform) and not transform._f_max_computed:
                    # Only set component function if f_max needs lazy computation.
                    # When f_max is pre-computed (passed to constructor), respect it.
                    def partial_func(x):
                        for input_transform in self.input_transforms:
                            x = input_transform(x)
                        return self.base_function.evaluate(x)

                    class PartialFunctionWrapper:
                        def __init__(self, func, dimension):
                            self.evaluate = func
                            self.dimension = dimension

                    wrapper = PartialFunctionWrapper(partial_func, base_function.dimension)
                    transform.set_component_function(wrapper)

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the composed function.

        Applies input transforms, evaluates base function, applies output transforms.

        Args:
            x: Input vector

        Returns:
            Final scalar result
        """
        x = self._validate_input(x)

        # Apply input transformations in order
        for transform in self.input_transforms:
            x = transform(x)

        # Evaluate base function
        result = self.base_function.evaluate(x)

        # Apply output transformations in order
        for transform in self.output_transforms:
            result = transform(result)

        return float(result)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the composed function on a batch.

        Args:
            X: Batch of input vectors

        Returns:
            Batch of results
        """
        X = self._validate_batch_input(X)

        # Apply input transformations in order
        for transform in self.input_transforms:
            X = transform.transform_batch(X)

        # Evaluate base function
        results = self.base_function.evaluate_batch(X)

        # Apply output transformations in order
        for transform in self.output_transforms:
            results = transform.transform_batch(results)

        return results
