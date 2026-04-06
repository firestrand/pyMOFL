"""BBOBMixintSuiteFactory — 24 mixed-integer BBOB functions.

Wraps standard BBOB functions with 80% discretized variables.
Dimension must be divisible by 5.
"""

from __future__ import annotations

from pyMOFL.factories.bbob_suite_factory import BBOBSuiteFactory
from pyMOFL.functions.transformations.composed import ComposedFunction
from pyMOFL.functions.transformations.discretize import DiscretizeTransform


class BBOBMixintSuiteFactory:
    """Factory for 24 mixed-integer BBOB functions.

    Each function is a standard BBOB function with a DiscretizeTransform
    prepended to the input transforms.

    Parameters
    ----------
    bbob_factory : BBOBSuiteFactory, optional
        Base BBOB factory. Created if not provided.
    """

    def __init__(self, bbob_factory: BBOBSuiteFactory | None = None):
        self._bbob = bbob_factory or BBOBSuiteFactory()

    def create_function(self, fid: int, iid: int, dim: int) -> ComposedFunction:
        """Create a single mixed-integer BBOB function.

        Parameters
        ----------
        fid : int
            Function ID (1-24).
        iid : int
            Instance ID.
        dim : int
            Dimension. Must be divisible by 5.
        """
        if dim % 5 != 0:
            raise ValueError(f"Dimension must be divisible by 5 for mixint, got {dim}.")
        if fid < 1 or fid > 24:
            raise ValueError(f"Unknown BBOB function ID: {fid}. Must be 1-24.")

        # Build the base BBOB function
        base_func = self._bbob.create_function(fid, iid, dim)

        # Create discretize transform (prepend to input transforms)
        discretize = DiscretizeTransform(dimension=dim)
        input_transforms = [discretize, *base_func.input_transforms]

        return ComposedFunction(
            base_function=base_func.base_function,
            input_transforms=input_transforms,
            output_transforms=list(base_func.output_transforms),
            penalty_transforms=list(base_func.penalty_transforms),
        )

    def create_suite(self, iid: int, dim: int) -> list[ComposedFunction]:
        """Create all 24 mixed-integer BBOB functions."""
        return [self.create_function(fid, iid, dim) for fid in range(1, 25)]
