"""
Transformation decorators for optimization functions.

These decorators allow for the transformation of optimization functions,
such as shifting, rotating, scaling, and more.
"""

from .shifted import Shifted
from .rotated import Rotated
from .scaled import Scaled
from .biased import Biased
from .noise import Noise
from .matrix_transform import MatrixTransform
from .boundary_adjusted_shift import BoundaryAdjustedShift
from .max_absolute import MaxAbsolute

DECORATOR_REGISTRY = {
    "Shifted": Shifted,
    "Rotated": Rotated,
    "Biased": Biased,
    "Noise": Noise,
    "Scaled": Scaled,
    "MatrixTransform": MatrixTransform,
    "BoundaryAdjustedShift": BoundaryAdjustedShift,
}

def register_decorator(name):
    def decorator(cls):
        DECORATOR_REGISTRY[name] = cls
        return cls
    return decorator

@register_decorator("Shifted")
class _Shifted(Shifted):
    pass

@register_decorator("Rotated")
class _Rotated(Rotated):
    pass

@register_decorator("Biased")
class _Biased(Biased):
    pass

@register_decorator("Noise")
class _Noise(Noise):
    pass

@register_decorator("Scaled")
class _Scaled(Scaled):
    pass

__all__ = [
    "Shifted",
    "Rotated",
    "Scaled",
    "Biased",
    "Noise",
    "MatrixTransform",
    "BoundaryAdjustedShift",
    "MaxAbsolute",
    "DECORATOR_REGISTRY",
]

 