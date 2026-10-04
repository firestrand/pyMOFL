"""
Composed function that chains transformations with optimization functions.

Allows building compositions like: bias(sphere(shift(x)))
"""

from collections.abc import Callable
from inspect import signature
from types import FunctionType, MethodType
from weakref import WeakKeyDictionary

import numpy as np

from pyMOFL.core.function import OptimizationFunction

from .base import PenaltyTransform, ScalarTransform, VectorTransform

_BUFFER_METHODS: WeakKeyDictionary[FunctionType, bool] = WeakKeyDictionary()


def _method_accepts_out(function: FunctionType) -> bool:
    """Reuse live class-method capabilities without retaining their functions."""
    if function in _BUFFER_METHODS:
        return _BUFFER_METHODS[function]
    try:
        signature(function).bind(None, None, out=None)
    except (TypeError, ValueError):
        accepts_out = False
    else:
        accepts_out = True
    _BUFFER_METHODS[function] = accepts_out
    return accepts_out


def _call_batch(
    method: Callable[..., np.ndarray], values: np.ndarray, out: np.ndarray | None
) -> np.ndarray:
    """Select optional-buffer invocation before calling the current component.

    Opaque callables use the established bufferless contract. The caller copies
    their result into out when needed. Only signature/binding errors are caught;
    failures from component execution propagate unchanged, without a retry.
    """
    if (
        isinstance(method, MethodType)
        and isinstance(method.__func__, FunctionType)
        and getattr(type(method.__self__), method.__func__.__name__, None) is method.__func__
    ):
        accepts_out = _method_accepts_out(method.__func__)
    else:
        # Instance replacements and opaque callables use their current signature.
        try:
            signature(method).bind(values, out=out)
        except (TypeError, ValueError):
            accepts_out = False
        else:
            accepts_out = True
    if not accepts_out:
        return method(values)
    return method(values, out=out)


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
        penalty_transforms: list[PenaltyTransform] | None = None,
    ):
        """
        Initialize composed function.

        Args:
            base_function: The optimization function to evaluate
            input_transforms: List of vector transforms to apply to input (in order)
            output_transforms: List of scalar transforms to apply to output (in order)
            penalty_transforms: List of penalty transforms (vector→scalar) whose
                results are added to the output. Penalties are computed on the
                raw input vector before input_transforms are applied.
        """
        super().__init__(
            dimension=base_function.dimension,
            initialization_bounds=base_function.initialization_bounds,
            operational_bounds=base_function.operational_bounds,
        )

        self.base_function = base_function
        self.input_transforms = input_transforms or []
        self.output_transforms = output_transforms or []
        self.penalty_transforms = penalty_transforms or []

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

        Applies input transforms, evaluates base function, applies output transforms,
        then adds penalty transforms computed on the raw (pre-transform) input.

        Args:
            x: Input vector

        Returns:
            Final scalar result
        """
        x = self._validate_input(x)
        raw_x = x  # Preserve for penalty computation

        # Apply input transformations in order
        for transform in self.input_transforms:
            x = transform(x)

        # Evaluate base function
        result = self.base_function.evaluate(x)

        # Apply output transformations in order
        for transform in self.output_transforms:
            result = transform(result)

        # Add penalty transforms (computed on raw input)
        for penalty in self.penalty_transforms:
            result += penalty(raw_x)

        return float(result)

    def evaluate_batch(self, X: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """
        Evaluate the composed function on a batch.

        Args:
            X: Batch of input vectors
            out: Optional pre-allocated buffer of shape (n_points,)

        Returns:
            Batch of results
        """
        X = self._validate_batch_input(X)
        raw_X = X  # Preserve for penalty computation

        # Apply input transformations in order
        for transform in self.input_transforms:
            X = transform.transform_batch(X)

        # Evaluate base function
        results = _call_batch(self.base_function.evaluate_batch, X, out)

        if out is not None and results is not out:
            out[:] = results
            results = out

        # Apply output transformations in order
        for transform in self.output_transforms:
            results = _call_batch(transform.transform_batch, results, out)
            if out is not None and results is not out:
                out[:] = results
                results = out

        # Add penalty transforms (computed on raw input)
        for penalty in self.penalty_transforms:
            pen_val = penalty.compute_batch(raw_X)
            if out is not None:
                out += pen_val
                results = out
            else:
                results = results + pen_val

        return results
