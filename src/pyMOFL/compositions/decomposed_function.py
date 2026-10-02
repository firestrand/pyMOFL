"""
DecomposedFunction — cooperative decomposition function composition.

Combines sub-functions evaluated on variable subsets split by GroupingTransform,
handling separable, m-dimensional non-separable, and overlapping sub-components.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.functions.transformations.decomposition import (
    DecomposedTransform,
    GroupingTransform,
)
from pyMOFL.registry import register


@register("DecomposedFunction")
@register("decomposed_function")
class DecomposedFunction(OptimizationFunction):
    """
    Decomposed optimization function.

    Evaluates individual component functions over partitioned or overlapping
    variable groups defined by a GroupingTransform / DecomposedTransform.

    Parameters
    ----------
    grouping_transform : GroupingTransform
        The decomposition transform defining variable groupings and block rotations.
    component_functions : list[OptimizationFunction] | None, optional
        List of sub-functions matching each group in grouping_transform.groups.
    non_separable_function : OptimizationFunction | list[OptimizationFunction] | None, optional
        Function(s) applied to non-separable groups.
    separable_function : OptimizationFunction | None, optional
        Function applied to the separable component.
    overlapping_function : OptimizationFunction | list[OptimizationFunction] | None, optional
        Function(s) applied to overlapping groups.
    weights : Sequence[float] | None, optional
        Weights applied to each component in the summation. Defaults to 1.0.
    bias : float, default=0.0
        Additive scalar offset (f_bias).
    initialization_bounds : Bounds | None, optional
        Bounds for initialization.
    operational_bounds : Bounds | None, optional
        Bounds for domain enforcement.
    """

    def __init__(
        self,
        grouping_transform: GroupingTransform | DecomposedTransform,
        component_functions: list[OptimizationFunction] | None = None,
        *,
        non_separable_function: OptimizationFunction | list[OptimizationFunction] | None = None,
        separable_function: OptimizationFunction | None = None,
        overlapping_function: OptimizationFunction | list[OptimizationFunction] | None = None,
        weights: Sequence[float] | None = None,
        bias: float = 0.0,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs: Any,
    ) -> None:
        self.grouping_transform = grouping_transform
        dim = grouping_transform.dimension

        super().__init__(
            dimension=dim,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
            **kwargs,
        )

        self.bias = float(bias)

        # Assemble the ordered list of sub-functions matching grouping_transform.groups
        if component_functions is not None:
            if len(component_functions) != len(grouping_transform.groups):
                raise ValueError(
                    f"Number of component_functions ({len(component_functions)}) must match "
                    f"number of groups ({len(grouping_transform.groups)})"
                )
            self._funcs = list(component_functions)
        else:
            self._funcs = []
            non_sep_idx = 0
            overlap_idx = 0

            # Normalize non-separable functions
            if isinstance(non_separable_function, list):
                non_sep_list = non_separable_function
            elif non_separable_function is not None:
                non_sep_list = [non_separable_function]
            else:
                non_sep_list = []

            # Normalize overlapping functions
            if isinstance(overlapping_function, list):
                overlap_list = overlapping_function
            elif overlapping_function is not None:
                overlap_list = [overlapping_function]
            else:
                overlap_list = []

            for g in grouping_transform.groups:
                if g.group_type == "non_separable":
                    if not non_sep_list:
                        raise ValueError(
                            "grouping_transform contains non_separable group but "
                            "no non_separable_function provided."
                        )
                    func = (
                        non_sep_list[non_sep_idx % len(non_sep_list)]
                        if len(non_sep_list) > 1
                        else non_sep_list[0]
                    )
                    self._funcs.append(func)
                    non_sep_idx += 1
                elif g.group_type == "separable":
                    if separable_function is None:
                        raise ValueError(
                            "grouping_transform contains separable group but "
                            "no separable_function provided."
                        )
                    self._funcs.append(separable_function)
                elif g.group_type == "overlapping":
                    if not overlap_list:
                        raise ValueError(
                            "grouping_transform contains overlapping group but "
                            "no overlapping_function provided."
                        )
                    func = (
                        overlap_list[overlap_idx % len(overlap_list)]
                        if len(overlap_list) > 1
                        else overlap_list[0]
                    )
                    self._funcs.append(func)
                    overlap_idx += 1

        # Check dimension compatibility for each component
        for i, (g, func) in enumerate(zip(grouping_transform.groups, self._funcs, strict=False)):
            if func.dimension != g.dimension:
                raise ValueError(
                    f"Component function {i} dimension ({func.dimension}) does not match "
                    f"group dimension ({g.dimension})"
                )

        n_comps = len(self._funcs)
        if weights is not None:
            w_arr = np.asarray(weights, dtype=np.float64)
            if w_arr.shape != (n_comps,):
                raise ValueError(
                    f"weights shape {w_arr.shape} must match number of components ({n_comps})"
                )
            self.weights: np.ndarray = w_arr
        else:
            self.weights = np.ones(n_comps, dtype=np.float64)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the decomposed function at point x."""
        x = self._validate_input(x)
        split_result = self.grouping_transform.split(x)

        total = self.bias
        for w, sub_x, func in zip(self.weights, split_result.components, self._funcs, strict=False):
            total += w * func.evaluate(sub_x)

        return float(total)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorized batch evaluation across all decomposed sub-functions."""
        X = self._validate_batch_input(X)
        batch_split = self.grouping_transform.split_batch(X)
        n_points = X.shape[0]

        total = np.full(n_points, self.bias, dtype=np.float64)
        for w, sub_X, func in zip(self.weights, batch_split.components, self._funcs, strict=False):
            total += w * func.evaluate_batch(sub_X)

        return total

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Compute the global minimum when groups partition the space and sub-optima are known."""
        if not self.grouping_transform.is_partition:
            raise NotImplementedError(
                "Global minimum cannot be trivially computed for overlapping or non-partitioning groups."
            )

        perm_opt = np.zeros(self.dimension, dtype=np.float64)
        total_opt_val = self.bias

        for w, g, func in zip(
            self.weights, self.grouping_transform.groups, self._funcs, strict=False
        ):
            x_sub_opt, f_sub_opt = func.get_global_minimum()
            total_opt_val += w * f_sub_opt
            if g.rotation_matrix is not None:
                # u = R @ x => x = R^T @ u
                x_sub_orig = g.rotation_matrix.T @ x_sub_opt
            else:
                x_sub_orig = x_sub_opt
            perm_opt[g.indices] = x_sub_orig

        if self.grouping_transform.permutation is not None:
            global_opt = np.empty_like(perm_opt)
            global_opt[self.grouping_transform.permutation] = perm_opt
        else:
            global_opt = perm_opt

        return global_opt, float(total_opt_val)
