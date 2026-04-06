"""
Shared test utilities for validating OptimizationFunction subclasses.

Provides reusable assertion helpers that verify any benchmark function
satisfies the standard contract: evaluate returns float, evaluate_batch
returns correct shape, get_global_minimum is consistent, bounds are set,
and input validation rejects wrong dimensions.

Usage in test files::

    from tests.utils.benchmark_validation import BenchmarkValidator

    class TestMyFunction:
        def test_contract(self):
            func = MyFunction(dimension=5)
            BenchmarkValidator.assert_contract(func, dimensions=[2, 5, 10])
"""

import numpy as np
import pytest

from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction


class BenchmarkValidator:
    """Reusable assertions for OptimizationFunction subclass contracts."""

    @staticmethod
    def assert_evaluate_returns_float(func: OptimizationFunction) -> None:
        """Assert that evaluate() returns a Python float for a random input."""
        x = np.random.default_rng(42).uniform(-1.0, 1.0, size=func.dimension)
        result = func.evaluate(x)
        assert isinstance(result, (float, np.floating)), (
            f"evaluate() returned {type(result).__name__}, expected float"
        )

    @staticmethod
    def assert_evaluate_batch_shape(func: OptimizationFunction, batch_size: int = 5) -> None:
        """Assert that evaluate_batch() returns an array with shape (batch_size,)."""
        rng = np.random.default_rng(42)
        X = rng.uniform(-1.0, 1.0, size=(batch_size, func.dimension))
        result = func.evaluate_batch(X)
        assert isinstance(result, np.ndarray), (
            f"evaluate_batch() returned {type(result).__name__}, expected ndarray"
        )
        assert result.shape == (batch_size,), (
            f"evaluate_batch() returned shape {result.shape}, expected ({batch_size},)"
        )

    @staticmethod
    def assert_evaluate_batch_consistent(func: OptimizationFunction, batch_size: int = 5) -> None:
        """Assert that evaluate_batch() gives the same results as individual evaluate() calls."""
        rng = np.random.default_rng(42)
        X = rng.uniform(-1.0, 1.0, size=(batch_size, func.dimension))
        batch_result = func.evaluate_batch(X)
        for i in range(batch_size):
            single_result = func.evaluate(X[i])
            np.testing.assert_allclose(
                batch_result[i],
                single_result,
                rtol=1e-12,
                err_msg=f"evaluate_batch()[{i}] != evaluate(X[{i}])",
            )

    @staticmethod
    def assert_callable_consistent(func: OptimizationFunction) -> None:
        """Assert that __call__ delegates to evaluate correctly."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-1.0, 1.0, size=func.dimension)
        call_result = func(x)
        eval_result = func.evaluate(x)
        np.testing.assert_allclose(call_result, eval_result, rtol=1e-12)

    @staticmethod
    def assert_global_minimum(
        func: OptimizationFunction, atol: float = 1e-8, rtol: float = 1e-6
    ) -> None:
        """Assert that get_global_minimum() returns a point where evaluate matches the declared value."""
        try:
            point, value = func.get_global_minimum()
        except NotImplementedError:
            pytest.skip(f"{type(func).__name__} does not implement get_global_minimum")
        assert point.shape == (func.dimension,), (
            f"get_global_minimum point shape {point.shape}, expected ({func.dimension},)"
        )
        actual = func.evaluate(point)
        np.testing.assert_allclose(
            actual,
            value,
            atol=atol,
            rtol=rtol,
            err_msg=(f"evaluate(global_min_point)={actual} != declared global_min_value={value}"),
        )

    @staticmethod
    def assert_bounds_set(func: OptimizationFunction) -> None:
        """Assert that initialization and operational bounds are Bounds instances with correct shapes."""
        assert isinstance(func.initialization_bounds, Bounds), (
            f"initialization_bounds is {type(func.initialization_bounds).__name__}"
        )
        assert isinstance(func.operational_bounds, Bounds), (
            f"operational_bounds is {type(func.operational_bounds).__name__}"
        )
        assert func.initialization_bounds.low.shape == (func.dimension,)
        assert func.initialization_bounds.high.shape == (func.dimension,)
        assert func.operational_bounds.low.shape == (func.dimension,)
        assert func.operational_bounds.high.shape == (func.dimension,)

    @staticmethod
    def assert_dimension_validation(func: OptimizationFunction) -> None:
        """Assert that evaluate and evaluate_batch reject wrong dimensions."""
        wrong_dim = func.dimension + 1
        with pytest.raises(ValueError):
            func.evaluate(np.zeros(wrong_dim))
        with pytest.raises(ValueError):
            func.evaluate_batch(np.zeros((3, wrong_dim)))

    @staticmethod
    def assert_contract(
        func: OptimizationFunction,
        *,
        check_global_minimum: bool = True,
        global_min_atol: float = 1e-8,
        global_min_rtol: float = 1e-6,
        batch_size: int = 5,
    ) -> None:
        """Run the full contract validation suite on a function instance.

        Parameters
        ----------
        func : OptimizationFunction
            The function instance to validate.
        check_global_minimum : bool
            Whether to test get_global_minimum consistency.
        global_min_atol : float
            Absolute tolerance for global minimum comparison.
        global_min_rtol : float
            Relative tolerance for global minimum comparison.
        batch_size : int
            Number of vectors for batch tests.
        """
        BenchmarkValidator.assert_evaluate_returns_float(func)
        BenchmarkValidator.assert_evaluate_batch_shape(func, batch_size)
        BenchmarkValidator.assert_evaluate_batch_consistent(func, batch_size)
        BenchmarkValidator.assert_callable_consistent(func)
        BenchmarkValidator.assert_bounds_set(func)
        BenchmarkValidator.assert_dimension_validation(func)
        if check_global_minimum:
            BenchmarkValidator.assert_global_minimum(
                func, atol=global_min_atol, rtol=global_min_rtol
            )

    @staticmethod
    def assert_contract_multiple_dimensions(
        func_class: type,
        dimensions: list[int],
        *,
        check_global_minimum: bool = True,
        global_min_atol: float = 1e-8,
        global_min_rtol: float = 1e-6,
        **func_kwargs,
    ) -> None:
        """Run the full contract across multiple dimensions for a scalable function.

        Parameters
        ----------
        func_class : type
            The OptimizationFunction subclass to instantiate.
        dimensions : list[int]
            Dimensions to test.
        check_global_minimum : bool
            Whether to test get_global_minimum consistency.
        global_min_atol : float
            Absolute tolerance for global minimum comparison.
        global_min_rtol : float
            Relative tolerance for global minimum comparison.
        **func_kwargs
            Additional keyword arguments passed to the constructor.
        """
        for dim in dimensions:
            func = func_class(dimension=dim, **func_kwargs)
            BenchmarkValidator.assert_contract(
                func,
                check_global_minimum=check_global_minimum,
                global_min_atol=global_min_atol,
                global_min_rtol=global_min_rtol,
            )
