"""Elementary absolute-value scalar transform."""

from typing import override

import numpy as np
from numpy.typing import NDArray

from .base import ScalarTransform


class AbsoluteTransform(ScalarTransform):
    """Map y to abs(y), usable after a target bias to report objective distance."""

    @override
    def __call__(self, y: float) -> float:
        """Return the absolute scalar value."""
        return abs(y)

    @override
    def transform_batch(
        self, Y: NDArray[np.float64], out: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Return absolute values, optionally using the supplied output buffer."""
        return np.absolute(Y, out=out)
