"""BBOBConstrainedSuiteFactory — 54 constrained BBOB functions.

9 objective types × 6 constraint configurations = 54 functions.
"""

from __future__ import annotations

from typing import Any

from pyMOFL.core.constrained_function import ConstrainedFunction
from pyMOFL.factories.bbob_suite_factory import BBOBSuiteFactory
from pyMOFL.utils.bbob_constraint_generator import BBOBConstraintGenerator
from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

# 9 objective function types (BBOB function IDs)
_OBJECTIVES: list[tuple[int, str]] = [
    (1, "Sphere"),
    (2, "Ellipsoidal"),
    (5, "Linear Slope"),
    (10, "Ellipsoidal (rotated)"),
    (11, "Discus"),
    (12, "Bent Cigar"),
    (14, "Different Powers"),
    (3, "Rastrigin"),
    (15, "Rastrigin (rotated)"),
]


class BBOBConstrainedSuiteFactory:
    """Factory for 54 constrained BBOB functions.

    9 objectives × 6 constraint configurations.

    Parameters
    ----------
    bbob_factory : BBOBSuiteFactory, optional
        Base BBOB factory.
    constraint_generator : BBOBConstraintGenerator, optional
        Constraint generator.
    instance_generator : BBOBInstanceGenerator, optional
        For generating xopt.
    """

    def __init__(
        self,
        bbob_factory: BBOBSuiteFactory | None = None,
        constraint_generator: BBOBConstraintGenerator | None = None,
        instance_generator: BBOBInstanceGenerator | None = None,
    ):
        self._bbob = bbob_factory or BBOBSuiteFactory()
        self._cgen = constraint_generator or BBOBConstraintGenerator()
        self._igen = instance_generator or BBOBInstanceGenerator()

    def create_function(self, obj_idx: int, config: int, iid: int, dim: int) -> ConstrainedFunction:
        """Create a single constrained function.

        Parameters
        ----------
        obj_idx : int
            Objective function index (0-8).
        config : int
            Constraint configuration (1-6).
        iid : int
            Instance ID.
        dim : int
            Dimension.
        """
        if obj_idx < 0 or obj_idx >= len(_OBJECTIVES):
            raise ValueError(f"Objective index must be 0-{len(_OBJECTIVES) - 1}, got {obj_idx}.")
        if config < 1 or config > 6:
            raise ValueError(f"Constraint config must be 1-6, got {config}.")

        fid, _name = _OBJECTIVES[obj_idx]

        # Build the base BBOB function
        base_func = self._bbob.create_function(fid, iid, dim)

        # Get xopt for constraint generation
        xopt = self._igen.generate_xopt(fid, iid, dim)

        # Generate constraints
        constraints = self._cgen.generate(fid=fid, iid=iid, dim=dim, config=config, xopt=xopt)

        return ConstrainedFunction(
            base_function=base_func,
            constraints=constraints,
            xopt=xopt,
        )

    def create_suite(self, iid: int, dim: int) -> list[ConstrainedFunction]:
        """Create all 54 constrained functions (9 objectives × 6 configs)."""
        funcs = []
        for obj_idx in range(len(_OBJECTIVES)):
            for config in range(1, 7):
                funcs.append(self.create_function(obj_idx, config, iid, dim))
        return funcs

    def get_function_info(self) -> list[dict[str, Any]]:
        """Get metadata for all 54 functions."""
        info = []
        for obj_idx, (obj_fid, obj_name) in enumerate(_OBJECTIVES):
            for config in range(1, 7):
                info.append(
                    {
                        "obj_idx": obj_idx,
                        "obj_fid": obj_fid,
                        "obj_name": obj_name,
                        "config": config,
                    }
                )
        return info
