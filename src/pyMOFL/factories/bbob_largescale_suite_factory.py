"""BBOBLargeScaleSuiteFactory — 24 large-scale BBOB functions (D=20-640).

Replaces full D×D rotations with P1·B·P2 structure:
- P1, P2: truncated swap permutations
- B: block-diagonal rotation (block_size = min(D, 40))

For D <= 40, identical to standard BBOB.
"""

from __future__ import annotations

from pyMOFL.factories.bbob_suite_factory import BBOBSuiteFactory
from pyMOFL.functions.transformations.block_diagonal_rotate import (
    BlockDiagonalRotateTransform,
)
from pyMOFL.functions.transformations.composed import ComposedFunction
from pyMOFL.functions.transformations.permutation import PermutationTransform
from pyMOFL.functions.transformations.rotate import RotateTransform
from pyMOFL.utils.bbob_instance import BBOBInstanceGenerator

_BLOCK_SIZE = 40


class BBOBLargeScaleSuiteFactory:
    """Factory for 24 large-scale BBOB functions.

    For D > 40, replaces each RotateTransform with P1·B·P2 chain.
    For D <= 40, produces identical results to standard BBOB.

    Parameters
    ----------
    bbob_factory : BBOBSuiteFactory, optional
        Base BBOB factory.
    instance_generator : BBOBInstanceGenerator, optional
        For generating permutations and block-diagonal rotations.
    """

    def __init__(
        self,
        bbob_factory: BBOBSuiteFactory | None = None,
        instance_generator: BBOBInstanceGenerator | None = None,
    ):
        self._bbob = bbob_factory or BBOBSuiteFactory()
        self._gen = instance_generator or BBOBInstanceGenerator()

    def create_function(self, fid: int, iid: int, dim: int) -> ComposedFunction:
        """Create a single large-scale BBOB function.

        Parameters
        ----------
        fid : int
            Function ID (1-24).
        iid : int
            Instance ID.
        dim : int
            Dimension.
        """
        if fid < 1 or fid > 24:
            raise ValueError(f"Unknown BBOB function ID: {fid}. Must be 1-24.")

        # Build the standard BBOB function
        base_func = self._bbob.create_function(fid, iid, dim)

        if dim <= _BLOCK_SIZE:
            return base_func

        # Replace RotateTransforms with P1·B·P2 chains
        new_input_transforms = []
        rot_index = 0
        for t in base_func.input_transforms:
            if isinstance(t, RotateTransform):
                # Generate unique seeds for this rotation's P1, B, P2
                base_seed = fid * 10000 + iid * 100 + rot_index
                p1 = self._gen.generate_permutation(dim, seed=base_seed)
                p2 = self._gen.generate_permutation(dim, seed=base_seed + 1)
                blocks = self._gen.generate_block_diagonal_rotation(
                    dim, seed=base_seed + 2, block_size=_BLOCK_SIZE
                )
                new_input_transforms.append(PermutationTransform(p2))
                new_input_transforms.append(BlockDiagonalRotateTransform(blocks))
                new_input_transforms.append(PermutationTransform(p1))
                rot_index += 1
            else:
                new_input_transforms.append(t)

        return ComposedFunction(
            base_function=base_func.base_function,
            input_transforms=new_input_transforms,
            output_transforms=list(base_func.output_transforms),
            penalty_transforms=list(base_func.penalty_transforms),
        )

    def create_suite(self, iid: int, dim: int) -> list[ComposedFunction]:
        """Create all 24 large-scale BBOB functions."""
        return [self.create_function(fid, iid, dim) for fid in range(1, 25)]
