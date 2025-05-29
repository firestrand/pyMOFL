"""
BiasedFunction decorator for adding a constant bias to an OptimizationFunction.
"""
from pyMOFL.core.function import OptimizationFunction

class BiasedFunction(OptimizationFunction):
    """
    Decorator that adds a constant bias to the output of an OptimizationFunction.
    Does not modify or enforce bounds; delegates bounds to the wrapped function.
    """
    def __init__(self, base_func: OptimizationFunction, bias: float = 0.0):
        self.base = base_func
        self.dimension = base_func.dimension
        self.bias = bias
        self.constraint_penalty = base_func.constraint_penalty

    def evaluate(self, x):
        return self.base(x) + self.bias

    def evaluate_batch(self, X):
        return self.base.evaluate_batch(X) + self.bias

    def violations(self, x):
        return self.base.violations(x)

    @property
    def initialization_bounds(self):
        return self.base.initialization_bounds

    @property
    def operational_bounds(self):
        return self.base.operational_bounds 