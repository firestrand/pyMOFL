"""
ComposableFunction base class for unified base/decorator pattern in pyMOFL.

This class allows a function to be used as either a base function (operating on input)
or as a decorator (operating on the output of another function). Subclasses implement
_apply(y) to define the transformation. If a base_function is provided, it is called first;
otherwise, the input is passed directly to _apply.

Usage:
    # As a base function
    f = SomeComposableFunction(dimension=10, ...)
    f(x)

    # As a decorator
    base = SphereFunction(...)
    f = SomeComposableFunction(base_function=base)
    f(x)

Properties like dimension, bounds, etc. are delegated to the base if present, or must be set directly.
"""

from .function import OptimizationFunction
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from typing import Any

class ComposableFunction(OptimizationFunction):
    """
    Abstract base for all composable optimization functions and decorators.
    Handles property delegation and input validation.
    Subclasses must implement _apply and _apply_batch.
    """
    def __init__(self, base_function: Optional[OptimizationFunction] = None, dimension: Optional[int] = None, initialization_bounds=None, operational_bounds=None):
        self.base_function = base_function
        if base_function is None:
            assert dimension is not None, "Must specify dimension if no base_function"
            self.dimension = dimension
            self.initialization_bounds = initialization_bounds
            self.operational_bounds = operational_bounds
        # else: properties are delegated

    @property
    def dimension(self):
        if self.base_function is not None:
            return self.base_function.dimension
        return self._dimension

    @dimension.setter
    def dimension(self, value):
        self._dimension = value

    @property
    def initialization_bounds(self):
        if self.base_function is not None:
            return self.base_function.initialization_bounds
        return self._initialization_bounds

    @initialization_bounds.setter
    def initialization_bounds(self, value):
        self._initialization_bounds = value

    @property
    def operational_bounds(self):
        if self.base_function is not None:
            return self.base_function.operational_bounds
        return self._operational_bounds

    @operational_bounds.setter
    def operational_bounds(self, value):
        self._operational_bounds = value

    @property
    def bounds(self):
        if self.base_function is not None:
            return self.base_function.bounds
        # Legacy: return as (dim, 2) array if possible
        if self.operational_bounds is not None:
            return np.stack([self.operational_bounds.low, self.operational_bounds.high], axis=1)
        return None

    def evaluate(self, x):
        if self.base_function is not None:
            y = self.base_function(x)
            return self._apply(y)
        else:
            x = self._validate_input(x)
            return self._apply(x)

    def evaluate_batch(self, X):
        if self.base_function is not None:
            Y = self.base_function.evaluate_batch(X)
            return self._apply_batch(Y)
        else:
            X = self._validate_batch_input(X)
            return self._apply_batch(X)

    def _apply(self, y):
        """
        Subclasses must implement this: applies the transformation to a single input or output.
        """
        raise NotImplementedError

    def _apply_batch(self, Y):
        """
        Subclasses must implement this: applies the transformation to a batch of inputs or outputs.
        """
        raise NotImplementedError

    def violations(self, x):
        if self.base_function is not None:
            return self.base_function.violations(self._apply(x))
        return 0.0

    def __call__(self, x: NDArray[Any]) -> float:
        """
        Evaluate the function at x, enforcing operational bounds and constraints.
        Returns np.nan if constraints are violated.
        Applies input transformation if this is an InputTransformingFunction.
        """
        # If this is an InputTransformingFunction, apply input transformation before enforcement
        if isinstance(self, InputTransformingFunction):
            x = self._apply(x)
        x_proj = self._enforce(x)
        if np.any(np.isnan(x_proj)):
            return np.nan
        return self.evaluate(x_proj)

class InputTransformingFunction(ComposableFunction):
    """
    Abstract base for decorators that transform the input before calling the base function.

    Evaluation pattern:
        - evaluate(x): calls base_function(_apply(x))
        - evaluate_batch(X): calls base_function.evaluate_batch(_apply_batch(X))

    Subclassing contract:
        - Subclasses must implement _apply and _apply_batch.
        - Subclasses MUST NOT override evaluate or evaluate_batch.
        - Use for decorators like Shifted, Rotated, Scaled, BoundaryAdjustedShift, etc.
    """
    def evaluate(self, x):
        if self.base_function is not None:
            return self.base_function(self._apply(x))
        else:
            x = self._validate_input(x)
            return self._apply(x)
    def evaluate_batch(self, X):
        if self.base_function is not None:
            return self.base_function.evaluate_batch(self._apply_batch(X))
        else:
            X = self._validate_batch_input(X)
            return self._apply_batch(X)

    def __call__(self, x: NDArray[Any]) -> float:
        """
        Evaluate the function at x, enforcing operational bounds and constraints.
        Always applies input transformation (_apply) before enforcement and evaluation.
        """
        x = self._apply(x)
        x_proj = self._enforce(x)
        if np.any(np.isnan(x_proj)):
            return np.nan
        return self.evaluate(x_proj)

class OutputTransformingFunction(ComposableFunction):
    """
    Abstract base for decorators that transform the output of the base function.

    Evaluation pattern:
        - evaluate(x): calls _apply(base_function(x))
        - evaluate_batch(X): calls _apply_batch(base_function.evaluate_batch(X))

    Subclassing contract:
        - Subclasses must implement _apply and _apply_batch.
        - Subclasses MUST NOT override evaluate or evaluate_batch.
        - Use for decorators like Biased, Noise, MaxAbsolute, etc.
    """
    def evaluate(self, x):
        if self.base_function is not None:
            return self._apply(self.base_function(x))
        else:
            x = self._validate_input(x)
            return self._apply(x)
    def evaluate_batch(self, X):
        if self.base_function is not None:
            return self._apply_batch(self.base_function.evaluate_batch(X))
        else:
            X = self._validate_batch_input(X)
            return self._apply_batch(X) 