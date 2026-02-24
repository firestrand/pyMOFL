"""TransformBuilder — construct transform objects from parsed config tuples."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyMOFL.core.bounds_optimum_transform import (
    BoundsOptimumTransform,
    normalize_optimum_pattern,
)
from pyMOFL.factories.data_loader import DataLoader
from pyMOFL.functions.transformations import (
    BiasTransform,
    IndexedRotateTransform,
    IndexedScaleTransform,
    IndexedShiftTransform,
    NoiseTransform,
    NonContinuousTransform,
    NormalizeTransform,
    OffsetTransform,
    RotateTransform,
    ScalarTransform,
    ScaleTransform,
    ShiftTransform,
    VectorTransform,
)


class TransformBuilder:
    """Build transform objects from (type, params) tuples.

    Supports transform aliases so composition configs from multiple benchmarks can
    share the same functional pipeline.
    """

    def __init__(self, data_loader: DataLoader) -> None:
        self._data_loader = data_loader

    @staticmethod
    def _normalize_type(transform_type: str) -> str:
        return transform_type.lower()

    @staticmethod
    def _first(params: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in params:
                return params[key]
        return None

    def build(
        self, transform_type: str, params: dict[str, Any], dimension: int
    ) -> VectorTransform | ScalarTransform:
        """Build a single transform from its type string and parameters."""
        transform_type = self._normalize_type(transform_type)
        params = dict(params or {})

        # Vector transforms
        if transform_type in {"shift", "translate"}:
            return self._build_shift(params, dimension)
        if transform_type in {"rotate", "rotation"}:
            return self._build_rotate(params, dimension)
        if transform_type in {"scale", "lambda"}:
            return self._build_scale(params, dimension)
        if transform_type == "offset":
            return self._build_offset(params, dimension)
        if transform_type == "indexed_shift":
            return self._build_indexed_shift(params, dimension)
        if transform_type == "indexed_rotate":
            return self._build_indexed_rotate(params, dimension)
        if transform_type == "indexed_scale":
            return self._build_indexed_scale(params)
        if transform_type == "non_continuous":
            return NonContinuousTransform()

        # Scalar transforms
        if transform_type == "bias":
            return BiasTransform(float(params.get("value", params.get("bias", 0.0))))
        if transform_type == "noise":
            noise_level = float(
                params.get("noise_level", params.get("level", params.get("factor", 0.4)))
            )
            return NoiseTransform(noise_level=noise_level, seed=params.get("seed"))
        if transform_type == "normalize":
            return NormalizeTransform(
                C=float(params.get("C", 2000.0)),
                f_max=params.get("f_max", "computed"),
                reference_point=float(params.get("reference_point", 5.0)),
            )

        raise ValueError(f"Unknown transform type: {transform_type}")

    def build_many(
        self, transforms: list[tuple[str, dict[str, Any]]], dimension: int
    ) -> tuple[list[VectorTransform], list[ScalarTransform]]:
        """Build multiple transforms, separating into input/output groups."""
        input_transforms: list[VectorTransform] = []
        output_transforms: list[ScalarTransform] = []
        for ttype, tparams in transforms:
            t = self.build(ttype, tparams, dimension)
            if isinstance(t, VectorTransform):
                input_transforms.append(t)
            else:
                output_transforms.append(t)
        return input_transforms, output_transforms

    # --- Private builders -------------------------------------------------

    def _build_shift(self, params: dict[str, Any], dimension: int) -> ShiftTransform:
        pattern = params.get("pattern")
        bounds_mapping = params.get("bounds_mapping")

        if not pattern and isinstance(bounds_mapping, str):
            pattern = normalize_optimum_pattern(bounds_mapping)

        src = self._first(params, "vector", "shift", "shift_file")

        if isinstance(src, str):
            vec = self._data_loader.load_vector(src, dimension)
        elif src is None:
            vec = np.zeros(dimension, dtype=np.float64)
        elif isinstance(src, (int, float)):
            vec = np.full(dimension, src, dtype=np.float64)
        else:
            vec = np.asarray(src, dtype=np.float64)

        if pattern:
            pattern_key = str(pattern).lower()
            pattern_key = normalize_optimum_pattern(pattern_key)
            cfg = {"pattern": pattern_key}
            lower_bound = float(params.get("lower_bound", params.get("bounds_min", -100.0)))
            upper_bound = float(params.get("upper_bound", params.get("bounds_max", 100.0)))

            if pattern_key == "alternate_bounds":
                from pyMOFL.core.bounds_optimum_transform import AlternateShiftOptimumPattern

                cfg["lower_bound"] = float(
                    params.get("fixed_value", params.get("bounds_value", -32.0))
                )
                vec = AlternateShiftOptimumPattern().construct_optimum(
                    dimension, vec, lower_bound=cfg["lower_bound"]
                )
            elif pattern_key == "bounds_shift":
                lower_bound = float(params.get("lower_bound", params.get("bounds_min", -100.0)))
                upper_bound = float(params.get("upper_bound", params.get("bounds_max", 100.0)))
                vec = BoundsOptimumTransform.create_from_config(
                    {"pattern": "bounds_shift"}
                ).get_optimum(dimension, vec, lower_bound=lower_bound, upper_bound=upper_bound)
            elif pattern_key == "composition_bounds":
                lower_bound = float(params.get("lower_bound", params.get("bounds_min", -5.0)))
                upper_bound = float(params.get("upper_bound", params.get("bounds_max", 5.0)))
                vec = BoundsOptimumTransform.create_from_config(
                    {"pattern": "composition_bounds"}
                ).get_optimum(dimension, vec, lower_bound=lower_bound, upper_bound=upper_bound)
            elif pattern_key == "alternate_odds":
                lower_bound = float(
                    params.get("alternate_odds_value", params.get("bounds_value", 5.0))
                )
                vec = BoundsOptimumTransform.create_from_config(
                    {"pattern": "alternate_odds"}
                ).get_optimum(
                    dimension,
                    vec,
                    lower_bound=lower_bound,
                    upper_bound=float(params.get("upper_bound", 5.0)),
                )
            else:
                pattern_transform = BoundsOptimumTransform.create_from_config(cfg)
                vec = pattern_transform.get_optimum(
                    dimension,
                    vec,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )

        return ShiftTransform(vec)

    def _build_offset(self, params: dict[str, Any], dimension: int) -> OffsetTransform:
        src = self._first(params, "value", "offset", "vector", "shift", "shift_file")
        if isinstance(src, str):
            offset_vec = self._data_loader.load_vector(src, dimension)
        elif src is None:
            offset_vec = np.zeros(dimension, dtype=np.float64)
        elif isinstance(src, (int, float)):
            offset_vec = np.full(dimension, src, dtype=np.float64)
        else:
            offset_vec = np.asarray(src, dtype=np.float64)
        return OffsetTransform(offset_vec)

    def _build_rotate(
        self, params: dict[str, Any], dimension: int
    ) -> RotateTransform | IndexedRotateTransform:
        matrix_source = self._first(
            params, "matrix", "rotation", "rotation_file", "rotation_matrix"
        )

        # Indexed rotation support via parameters.
        if "component_index" in params or "matrix_dimension" in params:
            matrices_source = params.get("matrixes")
            if matrices_source is None:
                matrices_source = params.get("matrices", matrix_source)

            if isinstance(matrices_source, str):
                path = self._data_loader.resolve_path(
                    matrices_source, params.get("matrix_dimension", dimension)
                )
                matrices = np.loadtxt(path, dtype=np.float64)
            else:
                matrices = np.asarray(matrices_source, dtype=np.float64)
            return IndexedRotateTransform(
                matrices=matrices,
                component_index=params.get("component_index"),
                matrix_dimension=params.get("matrix_dimension"),
            )

        if isinstance(matrix_source, str):
            if matrix_source == "identity":
                mat = np.eye(dimension)
            else:
                mat = self._data_loader.load_matrix(matrix_source, dimension)
        else:
            mat = np.asarray(matrix_source, dtype=np.float64)
        return RotateTransform(mat)

    def _build_scale(
        self, params: dict[str, Any], dimension: int
    ) -> ScaleTransform | IndexedScaleTransform:
        if "factors" in params:
            return IndexedScaleTransform(
                factors=params["factors"],
                component_index=params.get("component_index"),
                default_factor=params.get("default_factor", 1.0),
            )
        return ScaleTransform(float(params.get("factor", params.get("lambda", 1.0))))

    def _build_indexed_shift(self, params: dict[str, Any], dimension: int) -> IndexedShiftTransform:
        shifts = params.get("shifts")
        if shifts is None:
            shifts = params.get("shift")
        if isinstance(shifts, str):
            shifts = self._data_loader.load_vector(shifts, dimension)
        if shifts is None:
            raise ValueError("Indexed shift requires non-empty 'shifts' parameter")
        return IndexedShiftTransform(shifts=shifts, component_index=params.get("component_index"))

    def _build_indexed_rotate(
        self, params: dict[str, Any], dimension: int
    ) -> IndexedRotateTransform:
        matrices = params.get("matrices")
        if matrices is None:
            matrices = params.get("matrix")
        if matrices is None:
            matrices = params.get("rotation")
        if isinstance(matrices, str):
            matrices = self._data_loader.load_matrix(
                matrices, params.get("matrix_dimension", dimension)
            )
        if matrices is None:
            raise ValueError("Indexed rotate requires non-empty 'matrices' parameter")
        return IndexedRotateTransform(
            matrices=matrices,
            component_index=params.get("component_index"),
            matrix_dimension=params.get("matrix_dimension"),
        )

    def _build_indexed_scale(self, params: dict[str, Any]) -> IndexedScaleTransform:
        return IndexedScaleTransform(
            factors=params["factors"],
            component_index=params.get("component_index"),
            default_factor=params.get("default_factor", 1.0),
        )
