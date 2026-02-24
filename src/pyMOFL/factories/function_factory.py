"""
Simplified FunctionFactory with FunctionRegistry.

This module provides test-oriented factory utilities that construct
wrapper-free composed functions with explicit transformation metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyMOFL.core.function import OptimizationFunction
from pyMOFL.factories.composition_builder import CompositionBuilder
from pyMOFL.factories.config_parser import ConfigParser
from pyMOFL.factories.data_loader import DataLoader
from pyMOFL.factories.transform_builder import TransformBuilder
from pyMOFL.functions.transformations import (
    ComposedFunction,
    ScalarTransform,
    VectorTransform,
)

# Re-export for backward compatibility
__all__ = ["DataLoader", "FunctionFactory", "FunctionRegistry"]


class FunctionRegistry:
    """Registry for base optimization functions."""

    def __init__(self) -> None:
        from pyMOFL.functions.benchmark.ackley import AckleyFunction
        from pyMOFL.functions.benchmark.elliptic import HighConditionedElliptic
        from pyMOFL.functions.benchmark.griewank import GriewankFunction
        from pyMOFL.functions.benchmark.rastrigin import RastriginFunction
        from pyMOFL.functions.benchmark.rosenbrock import GriewankOfRosenbrock, RosenbrockFunction
        from pyMOFL.functions.benchmark.schaffer import Schaffer_F6_Expanded
        from pyMOFL.functions.benchmark.schwefel import Schwefel_1_2, Schwefel_2_6, Schwefel_2_13
        from pyMOFL.functions.benchmark.sphere import SphereFunction
        from pyMOFL.functions.benchmark.weierstrass import WeierstrassFunction

        self.base_functions: dict[str, type[OptimizationFunction]] = {
            "sphere": SphereFunction,
            "ackley": AckleyFunction,
            "rastrigin": RastriginFunction,
            "non_continuous_rastrigin": RastriginFunction,
            "griewank": GriewankFunction,
            "weierstrass": WeierstrassFunction,
            "rosenbrock": RosenbrockFunction,
            "elliptic": HighConditionedElliptic,
            "high_conditioned_elliptic": HighConditionedElliptic,
            "schwefel_1_2": Schwefel_1_2,
            "schwefel_2_6": Schwefel_2_6,
            "schwefel_2_13": Schwefel_2_13,
            "schaffer_f6_expanded": Schaffer_F6_Expanded,
            "griewank_of_rosenbrock": GriewankOfRosenbrock,
        }

    def register_base(self, name: str, func_cls: type[OptimizationFunction]) -> None:
        self.base_functions[name] = func_cls

    def create_base_function(self, func_type: str, **params) -> OptimizationFunction:
        if func_type not in self.base_functions:
            raise ValueError(f"Unknown base function: {func_type}")
        cls = self.base_functions[func_type]
        # Normalize parameter name 'dimension'
        if "dimension" not in params and "dim" in params:
            params["dimension"] = params.pop("dim")
        return cls(**params)


class FunctionFactory:
    """Creates composed functions from nested config using registry + loader."""

    def __init__(
        self, data_loader: DataLoader | None = None, registry: FunctionRegistry | None = None
    ):
        self.data_loader = data_loader or DataLoader()
        self.registry = registry or FunctionRegistry()
        self._parser = ConfigParser(known_base_types=frozenset(self.registry.base_functions))
        self._transform_builder = TransformBuilder(self.data_loader)
        self._composition_builder = CompositionBuilder(
            data_loader=self.data_loader,
            registry=self.registry,
            parser=self._parser,
        )

    def create_function(self, config: dict[str, Any]) -> ComposedFunction:
        parsed = self._parser.parse(config)
        if parsed.base_type is None:
            raise ValueError("No base function found in configuration")

        # Composition delegation
        if parsed.is_composition:
            comp_config = parsed.raw_composition_config
            assert comp_config is not None
            dim = comp_config.get("parameters", {}).get(
                "dimension",
                ConfigParser.extract_dimension(comp_config) or 2,
            )
            # If an outer non_continuous wrapper exists, pass it as a parameter to the composition
            outer_has_noncont = any(ttype == "non_continuous" for ttype, _ in parsed.transforms)
            if outer_has_noncont:
                comp_config = dict(comp_config)
                params = dict(comp_config.get("parameters", {}))
                params["non_continuous"] = True
                comp_config["parameters"] = params
            composed = self._composition_builder.build(comp_config, dim)
            # Compositions use raw x for distance-based weighting, so outer vector
            # transforms are skipped. Scalar output transforms (e.g., bias) still apply.
            input_transforms: list[VectorTransform] = []
            output_transforms: list[ScalarTransform] = []
            for ttype, tparams in parsed.transforms:
                t = self._transform_builder.build(ttype, tparams, composed.dimension)
                if isinstance(t, ScalarTransform):
                    output_transforms.append(t)
                # Vector transforms are intentionally skipped for compositions
            # Transforms are already in application order (innermost first) from ConfigParser
            return ComposedFunction(
                base_function=composed,
                input_transforms=input_transforms,
                output_transforms=output_transforms,
            )

        # Instantiate base function (load any file-backed base params first)
        processed_base_params = dict(parsed.base_params)
        dim = processed_base_params.get("dimension")

        if parsed.base_type == "schwefel_2_6":
            if "matrix_A" in processed_base_params and "A" not in processed_base_params:
                processed_base_params["A"] = processed_base_params.pop("matrix_A")
            if "matrix_B" in processed_base_params and "B" not in processed_base_params:
                processed_base_params["B"] = processed_base_params.pop("matrix_B")

            optimum_pattern = processed_base_params.pop("optimum_pattern", None)
            if optimum_pattern and dim is not None:
                shift_cfg: dict[str, Any] | None = None
                for idx in range(len(parsed.transforms) - 1, -1, -1):
                    ttype, tparams = parsed.transforms[idx]
                    if ttype == "shift":
                        shift_cfg = tparams
                        break
                if shift_cfg is not None:
                    src = (
                        shift_cfg.get("vector")
                        or shift_cfg.get("shift")
                        or shift_cfg.get("shift_file")
                    )
                    if isinstance(src, str):
                        raw_shift = self.data_loader.load_vector(src, dim)
                    elif src is not None:
                        raw_shift = np.asarray(src, dtype=np.float64)
                    else:
                        raw_shift = np.zeros(dim, dtype=np.float64)
                    lower_bound = float(
                        shift_cfg.get("lower_bound", shift_cfg.get("bounds_min", -100.0))
                    )
                    upper_bound = float(
                        shift_cfg.get("upper_bound", shift_cfg.get("bounds_max", 100.0))
                    )
                    from pyMOFL.core.bounds_optimum_transform import BoundsOptimumTransform

                    optimum_vec = BoundsOptimumTransform.create_from_config(
                        {"pattern": optimum_pattern}
                    ).get_optimum(dim, raw_shift, lower_bound=lower_bound, upper_bound=upper_bound)
                    A_src = processed_base_params.get("A")
                    if isinstance(A_src, str):
                        A_matrix = self.data_loader.load_matrix(A_src, dim)
                    else:
                        A_matrix = np.asarray(A_src, dtype=np.float64)
                    xopt = optimum_vec - raw_shift
                    processed_base_params["A"] = A_matrix
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        processed_base_params["B"] = A_matrix @ xopt
                    processed_base_params.setdefault("optimum_on_bounds", True)
                    processed_base_params.setdefault("optimum_point", optimum_vec)
                else:
                    processed_base_params.setdefault("compute_B", True)
            elif "B" not in processed_base_params and "shift" in processed_base_params:
                processed_base_params.setdefault("compute_B", True)

        elif parsed.base_type == "schwefel_2_13":
            if "matrix_A" in processed_base_params and "a" not in processed_base_params:
                processed_base_params["a"] = processed_base_params.pop("matrix_A")
            if "matrix_B" in processed_base_params and "b" not in processed_base_params:
                processed_base_params["b"] = processed_base_params.pop("matrix_B")

        for key in ("shift", "vector", "B", "alpha", "optimum_point"):
            val = processed_base_params.get(key)
            if isinstance(val, str):
                processed_base_params[key] = self.data_loader.load_vector(val, dim)
        for key in ("A", "a", "B_mat", "b", "matrix", "rotation"):
            val = processed_base_params.get(key)
            if isinstance(val, str):
                processed_base_params[key] = self.data_loader.load_matrix(val, dim)
        base_func = self.registry.create_base_function(parsed.base_type, **processed_base_params)

        # Build transform objects — already in application order from ConfigParser
        input_transforms: list[VectorTransform] = []
        output_transforms: list[ScalarTransform] = []
        for ttype, tparams in parsed.transforms:
            t = self._transform_builder.build(ttype, tparams, base_func.dimension)
            if isinstance(t, VectorTransform):
                input_transforms.append(t)
            else:
                output_transforms.append(t)

        return ComposedFunction(
            base_function=base_func,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
        )
