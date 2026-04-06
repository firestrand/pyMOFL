"""
Simplified FunctionFactory with FunctionRegistry.

This module provides test-oriented factory utilities that construct
wrapper-free composed functions with explicit transformation metadata.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

import numpy as np

from pyMOFL.core.function import OptimizationFunction
from pyMOFL.factories.composition_builder import CompositionBuilder
from pyMOFL.factories.config_parser import ConfigParser
from pyMOFL.factories.data_loader import DataLoader
from pyMOFL.factories.transform_builder import TransformBuilder
from pyMOFL.functions.transformations import (
    ComposedFunction,
    PenaltyTransform,
    ScalarTransform,
    VectorTransform,
)

# Re-export for backward compatibility
__all__ = ["DataLoader", "FunctionFactory", "FunctionRegistry"]


class FunctionRegistry:
    """Registry for base optimization functions."""

    def __init__(self) -> None:
        from pyMOFL.functions.benchmark.ackley import AckleyFunction
        from pyMOFL.functions.benchmark.attractive_sector import AttractiveSectorFunction
        from pyMOFL.functions.benchmark.beale import BealeFunction
        from pyMOFL.functions.benchmark.bent_cigar import BentCigarFunction, DiscusFunction
        from pyMOFL.functions.benchmark.bohachevsky import (
            Bohachevsky1Function,
            Bohachevsky2Function,
            Bohachevsky3Function,
        )
        from pyMOFL.functions.benchmark.booth import BoothFunction
        from pyMOFL.functions.benchmark.brown import BrownFunction
        from pyMOFL.functions.benchmark.buche_rastrigin import BucheRastriginFunction
        from pyMOFL.functions.benchmark.bukin import Bukin6Function
        from pyMOFL.functions.benchmark.camel import SixHumpCamelFunction, ThreeHumpCamelFunction
        from pyMOFL.functions.benchmark.chung_reynolds import ChungReynoldsFunction
        from pyMOFL.functions.benchmark.colville import ColvilleFunction
        from pyMOFL.functions.benchmark.cross_in_tray import CrossInTrayFunction
        from pyMOFL.functions.benchmark.different_powers import DifferentPowersFunction
        from pyMOFL.functions.benchmark.drop_wave import DropWaveFunction
        from pyMOFL.functions.benchmark.eggholder import EggholderFunction
        from pyMOFL.functions.benchmark.elliptic import HighConditionedElliptic
        from pyMOFL.functions.benchmark.gallagher_peaks import GallagherPeaksFunction
        from pyMOFL.functions.benchmark.griewank import GriewankFunction
        from pyMOFL.functions.benchmark.hartmann import Hartmann3Function, Hartmann6Function
        from pyMOFL.functions.benchmark.holder_table import HolderTableFunction
        from pyMOFL.functions.benchmark.katsuura import KatsuuraFunction
        from pyMOFL.functions.benchmark.langermann import LangermannFunction
        from pyMOFL.functions.benchmark.levy import LevyCECFunction, LevyFunction
        from pyMOFL.functions.benchmark.linear_slope import LinearSlopeFunction
        from pyMOFL.functions.benchmark.lunacek import (
            LunacekBiRastriginCECFunction,
            LunacekBiRastriginFunction,
        )
        from pyMOFL.functions.benchmark.michalewicz import MichalewiczFunction
        from pyMOFL.functions.benchmark.multi_basin import MultiBasinFunction
        from pyMOFL.functions.benchmark.qing import QingFunction
        from pyMOFL.functions.benchmark.quartic import QuarticFunction
        from pyMOFL.functions.benchmark.rastrigin import RastriginFunction
        from pyMOFL.functions.benchmark.rosenbrock import GriewankOfRosenbrock, RosenbrockFunction
        from pyMOFL.functions.benchmark.salomon import SalomonFunction
        from pyMOFL.functions.benchmark.schaffer import (
            Schaffer_F6_Expanded,
            SchaffersF7CECFunction,
            SchaffersF7Function,
        )
        from pyMOFL.functions.benchmark.schwefel import Schwefel_1_2, Schwefel_2_6, Schwefel_2_13
        from pyMOFL.functions.benchmark.schwefel_sin import SchwefelSinFunction
        from pyMOFL.functions.benchmark.sharp_ridge import SharpRidgeFunction
        from pyMOFL.functions.benchmark.sphere import SphereFunction
        from pyMOFL.functions.benchmark.step_ellipsoid import StepEllipsoidFunction
        from pyMOFL.functions.benchmark.styblinski_tang import StyblinskiTangFunction
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
            "linear_slope": LinearSlopeFunction,
            "attractive_sector": AttractiveSectorFunction,
            "sharp_ridge": SharpRidgeFunction,
            "schwefel_sin": SchwefelSinFunction,
            "gallagher_peaks": GallagherPeaksFunction,
            "multi_basin": MultiBasinFunction,
            "gnbg": MultiBasinFunction,
            # BBOB Phase 3 additions
            "discus": DiscusFunction,
            "bent_cigar": BentCigarFunction,
            "different_powers": DifferentPowersFunction,
            "schaffer_f7": SchaffersF7Function,
            "schaffers_f7": SchaffersF7Function,
            "schaffer_f7_cec": SchaffersF7CECFunction,
            "levy": LevyFunction,
            "levy_cec": LevyCECFunction,
            "katsuura": KatsuuraFunction,
            "lunacek": LunacekBiRastriginFunction,
            "lunacek_bi_rastrigin": LunacekBiRastriginFunction,
            "lunacek_bi_rastrigin_cec": LunacekBiRastriginCECFunction,
            "buche_rastrigin": BucheRastriginFunction,
            "step_ellipsoid": StepEllipsoidFunction,
            # Phase 4: Classical scalable functions
            "styblinski_tang": StyblinskiTangFunction,
            "salomon": SalomonFunction,
            "michalewicz": MichalewiczFunction,
            "langermann": LangermannFunction,
            "brown": BrownFunction,
            "chung_reynolds": ChungReynoldsFunction,
            "qing": QingFunction,
            "quartic": QuarticFunction,
            # Phase 4: Classical fixed-dimension functions
            "beale": BealeFunction,
            "booth": BoothFunction,
            "bohachevsky1": Bohachevsky1Function,
            "bohachevsky2": Bohachevsky2Function,
            "bohachevsky3": Bohachevsky3Function,
            "bukin6": Bukin6Function,
            "six_hump_camel": SixHumpCamelFunction,
            "three_hump_camel": ThreeHumpCamelFunction,
            "cross_in_tray": CrossInTrayFunction,
            "drop_wave": DropWaveFunction,
            "eggholder": EggholderFunction,
            "holder_table": HolderTableFunction,
            "hartmann3": Hartmann3Function,
            "hartmann6": Hartmann6Function,
            "colville": ColvilleFunction,
        }
        self._register_discovered_benchmark_functions()

    @staticmethod
    def _snake_alias(name: str) -> str:
        """Convert class-like names to snake_case aliases."""
        base = name[:-8] if name.endswith("Function") else name
        if "_" in base:
            return base.lower()
        return re.sub(r"(?<!^)(?=[A-Z])", "_", base).lower()

    def _register_discovered_benchmark_functions(self) -> None:
        """Auto-register benchmark classes exported by the benchmark package.

        Discovers functions from two sources:
        1. The benchmark __all__ exports (snake_case + lowercase aliases)
        2. The global @register decorator registry (explicit aliases like "hgbat")
        """
        from pyMOFL.functions import benchmark as benchmark_pkg
        from pyMOFL.registry import _COMPONENTS

        for symbol in getattr(benchmark_pkg, "__all__", []):
            obj = getattr(benchmark_pkg, symbol, None)
            if not inspect.isclass(obj):
                continue
            if not issubclass(obj, OptimizationFunction):
                continue

            aliases = {
                self._snake_alias(obj.__name__),
                obj.__name__.lower(),
            }
            for alias in aliases:
                self.base_functions.setdefault(alias, obj)

        # Also register aliases from the global @register decorator registry
        for alias, cls in _COMPONENTS.items():
            if issubclass(cls, OptimizationFunction):
                self.base_functions.setdefault(alias.lower(), cls)

    def register_base(self, name: str, func_cls: type[OptimizationFunction]) -> None:
        self.base_functions[name] = func_cls

    def create_base_function(self, func_type: str, **params) -> OptimizationFunction:
        func_type = func_type.lower()
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
        self._parser = ConfigParser(frozenset(self.registry.base_functions))
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

        # Composition/hybrid delegation
        if parsed.is_composition:
            comp_config = parsed.raw_composition_config
            assert comp_config is not None

            # Hybrid function delegation
            if parsed.base_type == "hybrid":
                return self._build_hybrid(comp_config, parsed.transforms)

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
            # transforms cannot be applied. Scalar/penalty transforms still apply.
            input_transforms: list[VectorTransform] = []
            output_transforms: list[ScalarTransform] = []
            penalty_transforms: list[PenaltyTransform] = []
            for ttype, tparams in parsed.transforms:
                # non_continuous is handled above as a composition parameter
                if ttype == "non_continuous":
                    continue
                t = self._transform_builder.build(ttype, tparams, composed.dimension)
                if isinstance(t, PenaltyTransform):
                    penalty_transforms.append(t)
                elif isinstance(t, ScalarTransform):
                    output_transforms.append(t)
                elif isinstance(t, VectorTransform):
                    raise ValueError(
                        f"Vector transform '{ttype}' cannot wrap a composition function. "
                        f"Composition functions require raw input for distance-based "
                        f"weighting. Move vector transforms inside individual components."
                    )
            # Transforms are already in application order (innermost first) from ConfigParser
            return ComposedFunction(
                base_function=composed,
                input_transforms=input_transforms,
                output_transforms=output_transforms,
                penalty_transforms=penalty_transforms,
            )

        # Instantiate base function (load any file-backed base params first)
        processed_base_params = dict(parsed.base_params)
        dim = processed_base_params.get("dimension")

        for key in ("shift", "vector", "B", "alpha", "optimum_point"):
            val = processed_base_params.get(key)
            if isinstance(val, str):
                processed_base_params[key] = self.data_loader.load_vector(val, dim)
        for key in ("A", "a", "B_mat", "b", "matrix", "rotation", "cosine_rotation"):
            val = processed_base_params.get(key)
            if isinstance(val, str):
                processed_base_params[key] = self.data_loader.load_matrix(val, dim)
        # shift_signs: load vector from file, then extract signs
        if "shift_signs" in processed_base_params:
            val = processed_base_params["shift_signs"]
            if isinstance(val, str):
                vec = self.data_loader.load_vector(val, dim)
                processed_base_params["shift_signs"] = np.sign(vec)
        # cosine_conditioning: apply 100^(i/(2(D-1))) scaling to cosine term
        # Used by CEC 2013 Lunacek (applied even without rotation)
        if processed_base_params.pop("cosine_conditioning", False) and dim is not None:
            cond = np.power(100.0, np.arange(dim) / (2.0 * (dim - 1)))
            processed_base_params["cosine_rotation"] = np.diag(cond)
        # cosine_rotation_double: load stacked rotation matrices → R1, conditioning, R2 → combined
        # Used by CEC 2013 F18 where cosine uses R2 @ diag(100^(i/(2(D-1)))) @ R1
        if "cosine_rotation_double" in processed_base_params:
            val = processed_base_params.pop("cosine_rotation_double")
            if isinstance(val, str) and dim is not None:
                path = self.data_loader.resolve_path(val, dim)
                full_mat = np.loadtxt(path, dtype=np.float64)
                r1 = full_mat[:dim, :dim]
                r2 = full_mat[dim : 2 * dim, :dim]
                cond = np.power(100.0, np.arange(dim) / (2.0 * (dim - 1)))
                processed_base_params["cosine_rotation"] = r2 @ np.diag(cond) @ r1
        base_func = self.registry.create_base_function(parsed.base_type, **processed_base_params)

        # Build transform objects — already in application order from ConfigParser
        input_transforms: list[VectorTransform] = []
        output_transforms: list[ScalarTransform] = []
        penalty_transforms: list[PenaltyTransform] = []
        for ttype, tparams in parsed.transforms:
            t = self._transform_builder.build(ttype, tparams, base_func.dimension)
            if isinstance(t, PenaltyTransform):
                penalty_transforms.append(t)
            elif isinstance(t, VectorTransform):
                input_transforms.append(t)
            else:
                output_transforms.append(t)

        return ComposedFunction(
            base_function=base_func,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            penalty_transforms=penalty_transforms,
        )

    def _build_hybrid(
        self,
        hybrid_config: dict[str, Any],
        outer_transforms: list[tuple[str, dict[str, Any]]],
    ) -> ComposedFunction:
        """Build a hybrid function from config.

        CEC 2014 hybrid pipeline:
        1. Outer transforms on full vector (shift → scale → rotate)
        2. Shuffle (permute) the transformed vector
        3. Partition into chunks based on percentages
        4. Evaluate each component on its chunk (with internal transforms) and sum
        """
        from math import ceil

        from pyMOFL.compositions.hybrid_function import HybridFunction
        from pyMOFL.functions.transformations import PermutationTransform
        from pyMOFL.utils.suite_config import force_inject_dimension

        params = hybrid_config.get("parameters", {})
        functions_configs = hybrid_config.get("functions", [])
        if not functions_configs:
            raise ValueError("Hybrid function requires 'functions' array")

        percentages = params.get("percentages", [])
        if not percentages or len(percentages) != len(functions_configs):
            raise ValueError("Hybrid 'percentages' must match 'functions' array length")

        # Parse inner transforms from the nested "function" node
        inner_config = hybrid_config.get("function", {})
        inner_parsed = self._parser.parse(inner_config) if inner_config else None

        # We need to know the dimension to build transforms and components.
        dim = params.get("dimension") or params.get("dim")
        if dim is None and inner_parsed and inner_parsed.base_params:
            dim = inner_parsed.base_params.get("dimension")
        if dim is None:
            dim = ConfigParser.extract_dimension(hybrid_config)
        if dim is None:
            raise ValueError("Hybrid function: cannot determine dimension")
        dim = int(dim)

        # Build input transforms from inner "function" chain
        input_transforms: list[VectorTransform] = []
        if inner_parsed:
            for ttype, tparams in inner_parsed.transforms:
                t = self._transform_builder.build(ttype, tparams, dim)
                if isinstance(t, VectorTransform):
                    input_transforms.append(t)

        # Load shuffle permutation (1-indexed → 0-indexed)
        shuffle_source = params.get("shuffle") or params.get("shuffle_file")
        if shuffle_source is not None:
            if isinstance(shuffle_source, str):
                shuffle_path = self.data_loader.resolve_path(shuffle_source, dim)
                raw_shuffle = np.loadtxt(shuffle_path, dtype=int).flatten()
            else:
                raw_shuffle = np.asarray(shuffle_source, dtype=int).flatten()
            # CEC data is 1-indexed
            perm = raw_shuffle[:dim] - 1
            input_transforms.append(PermutationTransform(perm))

        # Compute partitions from percentages using ceil
        # CEC 2020 hf01/hf05/hf06 assign remainder to FIRST component
        remainder_first = params.get("remainder_first", False)
        partition_sizes = []
        remaining = dim
        if remainder_first:
            # Compute all but first, then first gets remainder
            for i in range(1, len(percentages)):
                size = ceil(percentages[i] * dim)
                size = min(size, remaining)
                partition_sizes.append(size)
                remaining -= size
            partition_sizes.insert(0, remaining)
        else:
            for i, pct in enumerate(percentages):
                if i == len(percentages) - 1:
                    partition_sizes.append(remaining)
                else:
                    size = ceil(pct * dim)
                    size = min(size, remaining)
                    partition_sizes.append(size)
                    remaining -= size

        # Build partitions as (start, end) tuples
        partitions: list[tuple[int, int]] = []
        offset = 0
        for size in partition_sizes:
            partitions.append((offset, offset + size))
            offset += size

        # Build component functions.
        # Components may have internal transforms (scale, offset) matching
        # the CEC C code's per-function sr_func + offset logic.
        # IMPORTANT: force_inject_dimension overrides any dimension that was
        # already set by the outer inject_dimension call on the full config.
        components: list[OptimizationFunction] = []
        for i, comp_cfg in enumerate(functions_configs):
            dim_i = partition_sizes[i]
            if "function" in comp_cfg:
                # Component has nested transforms — build as full function
                comp_with_dim = force_inject_dimension(comp_cfg, dim_i)
                component = self.create_function(comp_with_dim)
            else:
                # Bare component — just create base function
                comp_type = comp_cfg.get("type", "").lower()
                comp_params = dict(comp_cfg.get("parameters", {}))
                comp_params["dimension"] = dim_i
                component = self.registry.create_base_function(comp_type, **comp_params)
            components.append(component)

        hybrid = HybridFunction(
            components=components,
            partitions=partitions,
            weights=None,
            normalize_weights=False,
        )

        # Build outer transforms (e.g., bias wrapping the hybrid)
        output_transforms: list[ScalarTransform] = []
        penalty_transforms: list[PenaltyTransform] = []
        for ttype, tparams in outer_transforms:
            t = self._transform_builder.build(ttype, tparams, dim)
            if isinstance(t, PenaltyTransform):
                penalty_transforms.append(t)
            elif isinstance(t, ScalarTransform):
                output_transforms.append(t)
            elif isinstance(t, VectorTransform):
                input_transforms.append(t)

        return ComposedFunction(
            base_function=hybrid,
            input_transforms=input_transforms,
            output_transforms=output_transforms,
            penalty_transforms=penalty_transforms,
        )
