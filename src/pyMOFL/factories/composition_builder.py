"""Generic composition builder.

Composition nodes are assembled by:

1. Parsing each component config with ConfigParser.
2. Building each component function from the registry (or recursively from nested
   composition configs).
3. Adding shared composition transforms.
4. Appending normalization for weighted-composition behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pyMOFL.core.bounds_optimum_transform import (
    BoundsOptimumTransform,
    normalize_optimum_pattern,
)
from pyMOFL.factories.data_loader import DataLoader
from pyMOFL.factories.transform_builder import TransformBuilder

if TYPE_CHECKING:
    from pyMOFL.core.function import OptimizationFunction
    from pyMOFL.factories.config_parser import ConfigParser
    from pyMOFL.factories.function_factory import FunctionRegistry


def _coerce_float_list(values: Any, expected_len: int, name: str) -> list[float]:
    if values is None:
        raise ValueError(f"Composition requires '{name}' array")
    if not isinstance(values, list):
        raise ValueError(f"Composition parameter '{name}' must be a list")
    if len(values) != expected_len:
        raise ValueError(
            f"Composition parameter '{name}' must contain {expected_len} values, got {len(values)}"
        )
    return [float(v) for v in values]


def _load_optima(path: str, *, dimension: int, n_components: int) -> list[np.ndarray]:
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim == 0:
        raise ValueError(f"Shift data at '{path}' is empty")

    if raw.ndim == 1:
        if raw.size == dimension:
            return [raw.copy() for _ in range(n_components)]
        if raw.size >= n_components * dimension:
            values = raw[: n_components * dimension].reshape((n_components, dimension))
            return [values[i].copy() for i in range(n_components)]
        raise ValueError(
            f"Shift data at '{path}' must be length {dimension} or at least "
            f"{n_components * dimension} for {n_components} components, got {raw.size}"
        )

    if raw.ndim == 2:
        if raw.shape[0] >= n_components and raw.shape[1] >= dimension:
            return [raw[i, :dimension].copy() for i in range(n_components)]
        if raw.shape[1] >= n_components and raw.shape[0] >= dimension:
            return [raw[:dimension, i].copy() for i in range(n_components)]

    raise ValueError(f"Unsupported shift data shape at '{path}': {raw.shape}")


def _coerce_optima_source(
    shift_source: Any, *, dimension: int, n_components: int
) -> list[np.ndarray]:
    if isinstance(shift_source, str):
        values = np.loadtxt(shift_source, dtype=np.float64)
    else:
        values = np.asarray(shift_source, dtype=np.float64)

    if values.ndim == 0:
        return [np.full(dimension, float(values), dtype=np.float64) for _ in range(n_components)]

    if values.ndim == 1:
        if values.size == dimension:
            return [values.copy() for _ in range(n_components)]
        if values.size >= n_components * dimension:
            reshape = values[: n_components * dimension].reshape((n_components, dimension))
            return [reshape[i].copy() for i in range(n_components)]
        raise ValueError(
            f"Shift data must be length {dimension} or at least "
            f"{n_components * dimension} for {n_components} components, got {values.size}"
        )

    if values.ndim == 2:
        if values.shape[0] >= n_components and values.shape[1] >= dimension:
            return [values[i, :dimension].copy() for i in range(n_components)]
        if values.shape[1] >= n_components and values.shape[0] >= dimension:
            return [values[:dimension, i].copy() for i in range(n_components)]

    raise ValueError(f"Unsupported shift data shape: {values.shape}")


def _load_rotation_blocks(
    raw: np.ndarray, *, dimension: int, n_components: int
) -> list[np.ndarray]:
    if raw.ndim == 1:
        if raw.size != dimension * dimension:
            raise ValueError(
                f"Flattened rotation data must have {dimension * dimension} entries, got {raw.size}"
            )
        matrix = raw.reshape((dimension, dimension))
        return [matrix.copy() for _ in range(n_components)]

    if raw.ndim == 2:
        if raw.shape == (dimension, dimension):
            return [raw.copy() for _ in range(n_components)]
        if raw.shape[0] % dimension == 0 and raw.shape[1] == dimension:
            blocks = raw.reshape((raw.shape[0] // dimension, dimension, dimension))
            return [blocks[i].copy() for i in range(min(n_components, blocks.shape[0]))]

    if raw.ndim == 3 and raw.shape[1] == dimension and raw.shape[2] == dimension:
        return [raw[i].copy() for i in range(min(n_components, raw.shape[0]))]

    raise ValueError(f"Unsupported rotation data shape: {raw.shape}")


def _is_valid_rotation_block(block: np.ndarray, *, dimension: int) -> bool:
    if block.ndim != 2:
        return False
    if block.shape[0] != block.shape[1] or block.shape[0] != dimension:
        return False
    return bool(np.isfinite(block).all())


def _pad_component_blocks(
    blocks: list[np.ndarray], *, dimension: int, n_components: int
) -> list[np.ndarray]:
    if len(blocks) >= n_components:
        return blocks[:n_components]
    if not blocks:
        return [np.eye(dimension, dtype=np.float64) for _ in range(n_components)]
    identity = np.eye(dimension, dtype=np.float64)
    while len(blocks) < n_components:
        blocks.append(identity.copy())
    return blocks


def extract_base_type(comp: dict[str, Any], known_types: frozenset[str]) -> str:
    """Walk a component config and return the first recognized base function type."""
    node = comp
    while node:
        t = node.get("type")
        if t in known_types or t in {"composition", "weight"}:
            return t
        node = node.get("function") if isinstance(node.get("function"), dict) else None
    return ""


def extract_base_params(comp: dict[str, Any], known_types: frozenset[str]) -> dict[str, Any]:
    """Walk a component config and return base params, excluding dimensions."""
    node = comp
    while node:
        t = node.get("type")
        if t in known_types:
            params = dict(node.get("parameters", {}))
            params.pop("dimension", None)
            params.pop("dim", None)
            return params
        node = node.get("function") if isinstance(node.get("function"), dict) else None
    return {}


def has_non_continuous_wrapper(comp: dict[str, Any], known_types: frozenset[str]) -> bool:
    """Detect a `non_continuous` wrapper before a base node."""
    node = comp
    while node:
        t = node.get("type")
        if t == "non_continuous":
            return True
        if t in known_types:
            return False
        node = node.get("function") if isinstance(node.get("function"), dict) else None
    return False


class CompositionBuilder:
    """Build composition functions from generic component configs."""

    def __init__(
        self,
        data_loader: DataLoader,
        registry: FunctionRegistry,
        parser: ConfigParser,
    ) -> None:
        self._data_loader = data_loader
        self._registry = registry
        self._parser = parser
        self._transform_builder = TransformBuilder(self._data_loader)

    def _build_min_composition(
        self, config: dict[str, Any], dimension: int
    ) -> OptimizationFunction:
        """Build a MinComposition from config."""
        from pyMOFL.compositions.min_composition import MinComposition
        from pyMOFL.functions.transformations import ComposedFunction

        functions_configs = config.get("functions", [])
        if not functions_configs:
            raise ValueError("MinComposition requires 'functions' array")

        components: list[OptimizationFunction] = []
        for comp_cfg in functions_configs:
            parsed = self._parser.parse(comp_cfg)
            if parsed.base_type is None:
                raise ValueError("Composition: base function not found in JSON component")

            if parsed.is_composition:
                assert parsed.raw_composition_config is not None
                component_base = self.build(parsed.raw_composition_config, dimension=dimension)
            else:
                base_params = dict(parsed.base_params)
                base_params["dimension"] = dimension
                component_base = self._registry.create_base_function(
                    parsed.base_type, **base_params
                )

            input_transforms, output_transforms, penalty_transforms = (
                self._transform_builder.build_many(parsed.transforms, dimension)
            )

            components.append(
                ComposedFunction(
                    base_function=component_base,
                    input_transforms=input_transforms,
                    output_transforms=output_transforms,
                    penalty_transforms=penalty_transforms,
                )
            )

        return MinComposition(
            dimension=dimension,
            components=components,
        )

    def _build_hybrid_component(
        self, config: dict[str, Any], dimension: int, shuffle: np.ndarray | None = None
    ) -> OptimizationFunction:
        """Build a hybrid function from a component config.

        Used for hybrid components inside compositions (CEC 2014 F29/F30).
        """
        from math import ceil

        from pyMOFL.compositions.hybrid_function import HybridFunction
        from pyMOFL.functions.transformations import ComposedFunction, PermutationTransform

        params = config.get("parameters", {})
        functions_configs = config.get("functions", [])
        if not functions_configs:
            raise ValueError("Hybrid component requires 'functions' array")

        percentages = params.get("percentages", [])
        if not percentages or len(percentages) != len(functions_configs):
            raise ValueError("Hybrid 'percentages' must match 'functions' array length")

        # Use provided shuffle or load from config
        if shuffle is None:
            shuffle_source = params.get("shuffle") or params.get("shuffle_file")
            if shuffle_source is not None:
                if isinstance(shuffle_source, str):
                    shuffle_path = self._data_loader.resolve_path(shuffle_source, dimension)
                    raw = np.loadtxt(shuffle_path, dtype=int).flatten()
                else:
                    raw = np.asarray(shuffle_source, dtype=int).flatten()
                shuffle = raw[:dimension] - 1  # 1-indexed → 0-indexed

        # Compute partition sizes from percentages
        # CEC 2020 hf01/hf05/hf06 assign remainder to FIRST component
        remainder_first = params.get("remainder_first", False)
        partition_sizes: list[int] = []
        remaining = dimension
        if remainder_first:
            for i in range(1, len(percentages)):
                size = ceil(percentages[i] * dimension)
                size = min(size, remaining)
                partition_sizes.append(size)
                remaining -= size
            partition_sizes.insert(0, remaining)
        else:
            for i, pct in enumerate(percentages):
                if i == len(percentages) - 1:
                    partition_sizes.append(remaining)
                else:
                    size = ceil(pct * dimension)
                    size = min(size, remaining)
                    partition_sizes.append(size)
                    remaining -= size

        # Build partitions as (start, end) tuples
        partitions: list[tuple[int, int]] = []
        offset = 0
        for size in partition_sizes:
            partitions.append((offset, offset + size))
            offset += size

        # Build component functions (with internal transforms if present)
        components: list[OptimizationFunction] = []
        for i, comp_cfg in enumerate(functions_configs):
            dim_i = partition_sizes[i]
            if "function" in comp_cfg:
                # Component has nested transforms — parse and build
                parsed = self._parser.parse(comp_cfg)
                if parsed.base_type is None:
                    raise ValueError("Hybrid component: no base function found")
                base_params = dict(parsed.base_params)
                base_params["dimension"] = dim_i
                base_func = self._registry.create_base_function(parsed.base_type, **base_params)
                input_xforms, output_xforms, penalty_xforms = self._transform_builder.build_many(
                    parsed.transforms, dim_i
                )
                component = ComposedFunction(
                    base_function=base_func,
                    input_transforms=input_xforms,
                    output_transforms=output_xforms,
                    penalty_transforms=penalty_xforms,
                )
            else:
                comp_type = comp_cfg.get("type", "").lower()
                comp_params = dict(comp_cfg.get("parameters", {}))
                comp_params["dimension"] = dim_i
                component = self._registry.create_base_function(comp_type, **comp_params)
            components.append(component)

        hybrid = HybridFunction(
            components=components,
            partitions=partitions,
            weights=None,
            normalize_weights=False,
        )

        # If shuffle is provided, apply it as a PermutationTransform
        if shuffle is not None:
            return ComposedFunction(
                base_function=hybrid,
                input_transforms=[PermutationTransform(shuffle)],
                output_transforms=[],
                penalty_transforms=[],
            )

        return hybrid

    def build(self, config: dict[str, Any], dimension: int) -> OptimizationFunction:
        """Build a composition from its full config dict."""
        from pyMOFL.compositions.weighted_composition import WeightedComposition
        from pyMOFL.functions.transformations import (
            ComposedFunction,
            NormalizeTransform,
            RotateTransform,
            ScaleTransform,
            ShiftTransform,
        )

        comp_type = config.get("type", "weight").lower()
        if comp_type in {"min", "lower_envelope"}:
            return self._build_min_composition(config, dimension)

        params = config.get("parameters", {})
        functions_configs = config.get("functions", [])
        if not functions_configs:
            raise ValueError("Composition function requires 'functions' array")

        ncomp = len(functions_configs)
        if "num_functions" in params and int(params["num_functions"]) != ncomp:
            raise ValueError("Composition 'num_functions' must match 'functions' array length")

        lambdas = _coerce_float_list(params.get("lambdas", [1.0] * ncomp), ncomp, "lambdas")
        sigmas = _coerce_float_list(params.get("sigmas", [1.0] * ncomp), ncomp, "sigmas")
        biases = _coerce_float_list(params.get("biases", [0.0] * ncomp), ncomp, "biases")

        # CEC 2014 multiplies by lambda (sh_rate), CEC 2005 divides.
        # When multiply_lambda=True, invert so ScaleTransform (which divides) effectively multiplies.
        multiply_lambda = bool(params.get("multiply_lambda", False))
        if multiply_lambda:
            lambdas = [1.0 / l if l != 0.0 else 1.0 for l in lambdas]

        # When True, skip builder's auto-normalization (CEC 2014 has it in per-component config).
        skip_normalize = bool(params.get("skip_normalize", False))

        C = float(params.get("C", 2000.0))
        reference_point = float(params.get("reference_point", 5.0))

        shift_source = params.get("shift_file") or params.get("optima_file") or params.get("shift")
        if shift_source is None:
            raise ValueError("Composition requires 'shift_file' parameter")

        if isinstance(shift_source, str):
            shift_path = self._data_loader.resolve_path(str(shift_source), dimension)
            optima = _load_optima(str(shift_path), dimension=dimension, n_components=ncomp)
        else:
            optima = _coerce_optima_source(shift_source, dimension=dimension, n_components=ncomp)

        optimum_pattern = params.get("optimum_pattern")
        if optimum_pattern:
            normalized_pattern = normalize_optimum_pattern(optimum_pattern)

            if normalized_pattern in {"alternate_odds"}:
                lower_bound = float(
                    params.get("alternate_odds_value", params.get("bounds_value", 5.0))
                )
                upper_bound = float(params.get("upper_bound", params.get("bounds_max", 100.0)))
            else:
                lower_bound = float(params.get("lower_bound", params.get("bounds_min", -100.0)))
                upper_bound = float(params.get("upper_bound", params.get("bounds_max", 100.0)))

            transform = BoundsOptimumTransform.create_from_config({"pattern": normalized_pattern})
            if normalized_pattern == "alternate_odds":
                optima[0] = transform.get_optimum(
                    dimension,
                    optima[0],
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            else:
                optima = [
                    transform.get_optimum(
                        dimension,
                        vector,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )
                    for vector in optima
                ]

        if params.get("zero_last_optimum", False):
            optima[-1] = np.zeros(dimension, dtype=np.float64)

        rotation_source = (
            params.get("rotation_file")
            or params.get("rotation")
            or params.get("rotation_matrix")
            or params.get("matrix")
        )
        if rotation_source is None:
            rotations = [np.eye(dimension, dtype=np.float64) for _ in range(ncomp)]
        else:
            if isinstance(rotation_source, str):
                rotation_path = self._data_loader.resolve_path(str(rotation_source), dimension)
                raw_rotation = np.loadtxt(rotation_path, dtype=np.float64)
            else:
                raw_rotation = np.asarray(rotation_source, dtype=np.float64)
            rotations = _pad_component_blocks(
                _load_rotation_blocks(raw_rotation, dimension=dimension, n_components=ncomp),
                dimension=dimension,
                n_components=ncomp,
            )
            rotations = [
                (
                    matrix
                    if _is_valid_rotation_block(matrix, dimension=dimension)
                    else np.eye(dimension, dtype=np.float64)
                )
                for matrix in rotations
            ]

        # Transpose rotation matrices if requested (CEC 2014 uses M @ x convention)
        transpose_rotation = bool(params.get("transpose_rotation", False))
        if transpose_rotation:
            rotations = [r.T for r in rotations]

        # r_flags: per-component rotation enable/disable (1=rotate, 0=identity)
        r_flags = params.get("r_flags")
        if r_flags is not None:
            for i, flag in enumerate(r_flags):
                if i < len(rotations) and not flag:
                    rotations[i] = np.eye(dimension, dtype=np.float64)

        # Load composition-level shuffle data for hybrid-composition (F29/F30)
        comp_shuffle_source = params.get("shuffle_file") or params.get("shuffle")
        comp_shuffle_components = int(params.get("shuffle_components", 0))
        comp_shuffles: list[np.ndarray] | None = None
        if comp_shuffle_source is not None and comp_shuffle_components > 0:
            if isinstance(comp_shuffle_source, str):
                shuffle_path = self._data_loader.resolve_path(str(comp_shuffle_source), dimension)
                raw_shuffle = np.loadtxt(shuffle_path, dtype=int).flatten()
            else:
                raw_shuffle = np.asarray(comp_shuffle_source, dtype=int).flatten()
            # Stacked shuffle: cf_num * dimension values, split per component
            comp_shuffles = []
            for ci in range(comp_shuffle_components):
                start = ci * dimension
                perm = raw_shuffle[start : start + dimension] - 1  # 1-indexed → 0-indexed
                comp_shuffles.append(perm)

        components: list[OptimizationFunction] = []
        for i, comp_cfg in enumerate(functions_configs):
            parsed = self._parser.parse(comp_cfg)
            if parsed.base_type is None:
                raise ValueError("Composition: base function not found in JSON component")

            if parsed.is_composition:
                if parsed.raw_composition_config is None:
                    raise ValueError("Composition component has malformed nested composition")
                # Handle hybrid components in compositions (F29/F30)
                if parsed.base_type == "hybrid":
                    shuffle = comp_shuffles[i] if comp_shuffles and i < len(comp_shuffles) else None
                    component_base = self._build_hybrid_component(
                        parsed.raw_composition_config, dimension, shuffle=shuffle
                    )
                else:
                    component_base = self.build(parsed.raw_composition_config, dimension=dimension)
            else:
                base_params = dict(parsed.base_params)
                base_params["dimension"] = dimension
                component_base = self._registry.create_base_function(
                    parsed.base_type, **base_params
                )

            component_input_transforms, component_output_transforms, component_penalties = (
                self._transform_builder.build_many(parsed.transforms, dimension)
            )
            lambda_i = lambdas[i] if lambdas[i] != 0.0 else 1.0
            component_input_transforms = [
                ShiftTransform(optima[i]),
                ScaleTransform(lambda_i),
                RotateTransform(rotations[i]),
                *component_input_transforms,
            ]

            if skip_normalize:
                # CEC 2014: normalization (if any) is in the per-component config
                components.append(
                    ComposedFunction(
                        base_function=component_base,
                        input_transforms=component_input_transforms,
                        output_transforms=list(component_output_transforms),
                        penalty_transforms=list(component_penalties),
                    )
                )
            else:
                # CEC 2005: compute f_max at reference point and add NormalizeTransform
                output_transforms_for_norm = list(component_output_transforms)
                component_for_norm = ComposedFunction(
                    base_function=component_base,
                    input_transforms=list(component_input_transforms),
                    output_transforms=output_transforms_for_norm,
                    penalty_transforms=list(component_penalties),
                )
                reference_value = optima[i] + reference_point
                f_max = abs(component_for_norm.evaluate(reference_value))
                if f_max < 1e-20:
                    f_max = 1.0

                final_output_transforms = [
                    *output_transforms_for_norm,
                    NormalizeTransform(C=C, f_max=f_max),
                ]

                components.append(
                    ComposedFunction(
                        base_function=component_base,
                        input_transforms=component_input_transforms,
                        output_transforms=final_output_transforms,
                        penalty_transforms=component_penalties,
                    )
                )

        return WeightedComposition(
            dimension=dimension,
            components=components,
            optima=optima,
            sigmas=sigmas,
            biases=biases,
            global_bias=float(params.get("global_bias", 0.0)),
            inverse_distance_weight=bool(params.get("inverse_distance_weight", False)),
            dominance_suppression=bool(params.get("dominance_suppression", False)),
            non_continuous=bool(params.get("non_continuous", False)),
        )
