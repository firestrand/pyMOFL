"""Opt-in evaluation helpers; existing objective methods retain their contracts."""

from operator import index

import numpy as np
from numpy.typing import NDArray

from .core.function import OptimizationFunction


def _real_numeric(array: NDArray[np.generic]) -> bool:
    return np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)


def evaluate_chunks(
    function: OptimizationFunction,
    X: NDArray[np.generic],
    chunk_size: int,
    *,
    deterministic: bool,
    batch_independent: bool,
    out: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Evaluate independent deterministic rows with bounded input chunks.

    Parameters
    ----------
    function
        Objective whose batch evaluator is deterministic and independent of
        batch boundaries. The caller must verify both properties.
    X
        Integer or floating NumPy matrix of shape (N, function.dimension).
        Each chunk is copied to float64 so the child cannot modify X.
    chunk_size
        Positive integer maximum rows per call. Booleans are rejected.
    deterministic, batch_independent
        Both must explicitly be True. These declarations do not inspect or
        enforce evaluator state; noisy and stateful objectives need care.
    out
        Optional writable float64 array of shape (N,) whose elements do not
        overlap each other or X.

    Returns
    -------
    numpy.ndarray
        Float64 values in input row order; out itself when supplied. Child
        results must be integer or floating arrays of shape (chunk_rows,).

    Notes
    -----
    An empty batch does not invoke the evaluator. Child exceptions propagate
    without retry; completed output chunks remain written on failure. The
    helper bounds its input copies, not arbitrary child internal allocations.
    It allocates the returned (N,) output when out is omitted. Conversion to
    float64 follows NumPy casting, including normal precision limitations.
    """
    if deterministic is not True or batch_independent is not True:
        raise ValueError("deterministic and batch_independent must both be True")
    if isinstance(chunk_size, (bool, np.bool_)):
        raise TypeError("chunk_size must be an integer, not bool")
    size = index(chunk_size)
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if not isinstance(X, np.ndarray) or not _real_numeric(X):
        raise TypeError("X must be an integer or floating NumPy array")
    if X.ndim != 2 or X.shape[1] != function.dimension:
        raise ValueError(f"X must have shape (N, {function.dimension})")
    if out is not None:
        if not isinstance(out, np.ndarray) or out.dtype != np.float64:
            raise TypeError("out must be a float64 NumPy array")
        if out.shape != (len(X),) or not out.flags.writeable:
            raise ValueError("out must be writable with shape (N,)")
        if len(out) > 1 and abs(out.strides[0]) < out.itemsize:
            raise ValueError("out elements must not overlap each other")
        if np.shares_memory(X, out):
            raise ValueError("out must not overlap X")
    result = np.empty(len(X), dtype=np.float64) if out is None else out
    for start in range(0, len(X), size):
        stop = min(start + size, len(X))
        chunk = np.array(X[start:stop], dtype=np.float64, copy=True)
        values = function.evaluate_batch(chunk)
        if not isinstance(values, np.ndarray) or not _real_numeric(values):
            raise TypeError("child results must be an integer or floating NumPy array")
        if values.shape != (stop - start,):
            raise ValueError("child results must have shape (chunk_rows,)")
        result[start:stop] = values
    return result
