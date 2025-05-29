"""
Transformation decorators for optimization functions.

These decorators allow for the transformation of optimization functions,
such as shifting, rotating, scaling, and more.
"""

from .shifted import ShiftedFunction
from .rotated import RotatedFunction
from .biased import BiasedFunction
from .noise import NoiseDecorator
from .scaled import ScaledFunction
from .matrix_transform import MatrixTransformFunction
from .boundary_adjusted_shift import BoundaryAdjustedShiftFunction

DECORATOR_REGISTRY = {
    "Shifted": ShiftedFunction,
    "Rotated": RotatedFunction,
    "Biased": BiasedFunction,
    "Noise": NoiseDecorator,
    "Scaled": ScaledFunction,
    "MatrixTransform": MatrixTransformFunction,
    "BoundaryAdjustedShift": BoundaryAdjustedShiftFunction,
}

def register_decorator(name):
    def decorator(cls):
        DECORATOR_REGISTRY[name] = cls
        return cls
    return decorator

@register_decorator("Shifted")
class _Shifted(ShiftedFunction):
    pass

@register_decorator("Rotated")
class _Rotated(RotatedFunction):
    pass

@register_decorator("Biased")
class _Biased(BiasedFunction):
    pass

@register_decorator("Noise")
class _Noise(NoiseDecorator):
    pass

@register_decorator("Scaled")
class _Scaled(ScaledFunction):
    pass

 