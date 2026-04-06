"""
Schaffer functions implementation.

This module contains all variants of Schaffer functions used in optimization benchmarks:
- Schaffer F6 (2D pair function, expanded to D dimensions)
- Schaffer F6 Expanded (CEC variant with wrap-around pairs)
- Schaffer N.1 (concentric rings, 2D only)
- Schaffer N.2 (deceptive oscillation, 2D only)
- Schaffer N.4 (cos-of-sin variant, 2D only)
- Schaffers F7 (adjacent-pair coupling, scalable, CEC 2013+)

References
----------
.. [1] Schaffer, J.D. (1985). "Multiple Objective Optimization with Vector Evaluated Genetic
       Algorithms". In Proceedings of the 1st International Conference on Genetic Algorithms.
       L. Erlbaum Associates Inc., pp. 93-100.
.. [2] Molga, M., & Smutnicki, C. (2005). "Test functions for optimization needs".
.. [3] Liang, J.J., et al. (2013). "Problem definitions and evaluation criteria
       for the CEC 2013 special session on real-parameter optimization."
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Schaffer_F6")
class Schaffer_F6(OptimizationFunction):
    """
    Schaffer's F6 function.

    The function is defined as:
        f(x) = 0.5 + (sin^2(sqrt(x1^2 + x2^2)) - 0.5) / (1 + 0.001 * (x1^2 + x2^2))^2

    This function is a challenging multimodal test function with a single global minimum
    surrounded by many local minima. The function is symmetric, and the global minimum
    is located at the origin.

    Domain: x_i ∈ [-100, 100] for i = 1, 2, ..., n
    Global minimum: f(0, 0, ..., 0) = 0

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for random initialization. If None, defaults to [-100, 100]^d.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. If None, defaults to [-100, 100]^d.

    References
    ----------
    .. [1] Schaffer, J.D. (1985). "Multiple Objective Optimization with Vector Evaluated Genetic
           Algorithms". In Proceedings of the 1st International Conference on Genetic Algorithms.
           L. Erlbaum Associates Inc., pp. 93-100.
    .. [2] Molga, M., & Smutnicki, C. (2005). "Test functions for optimization needs".
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        # Set default bounds: [-100, 100] for each dimension
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Schaffer F6 function at point x.

        For dimensions > 2, applies the 2D Schaffer F6 function to consecutive pairs
        of variables: (x1, x2), (x2, x3), ..., (xn-1, xn), (xn, x1).

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (dimension,).

        Returns
        -------
        float
            The function value at x.
        """
        x = self._validate_input(x)

        if self.dimension == 1:
            return 0.0  # Special case for 1D
        elif self.dimension == 2:
            return self._schaffer_f6_2d(x[0], x[1])
        else:
            # For higher dimensions, sum over consecutive pairs
            result = 0.0
            for i in range(self.dimension - 1):
                result += self._schaffer_f6_2d(x[i], x[i + 1])
            # Add the wrap-around pair
            result += self._schaffer_f6_2d(x[-1], x[0])
            return result

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Schaffer F6 function on a batch of points.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_points, dimension).

        Returns
        -------
        np.ndarray
            The function values for each point.
        """
        X = self._validate_batch_input(X)
        n_points, n_dims = X.shape
        results = np.zeros(n_points)

        if n_dims == 1:
            return results  # All zeros for 1D case
        elif n_dims == 2:
            for i in range(n_points):
                results[i] = self._schaffer_f6_2d(X[i, 0], X[i, 1])
        else:
            # For higher dimensions, sum over consecutive pairs
            for i in range(n_points):
                x = X[i]
                result = 0.0
                for j in range(n_dims - 1):
                    result += self._schaffer_f6_2d(x[j], x[j + 1])
                # Add the wrap-around pair
                result += self._schaffer_f6_2d(x[-1], x[0])
                results[i] = result

        return results

    @staticmethod
    def _schaffer_f6_2d(x1: float, x2: float) -> float:
        """
        Compute the 2D Schaffer F6 function.

        f(x1, x2) = 0.5 + (sin^2(sqrt(x1^2 + x2^2)) - 0.5) / (1 + 0.001 * (x1^2 + x2^2))^2

        Parameters
        ----------
        x1, x2 : float
            The two input variables.

        Returns
        -------
        float
            The function value.
        """
        sum_sq = x1**2 + x2**2
        sqrt_sum_sq = np.sqrt(sum_sq)
        numerator = np.sin(sqrt_sum_sq) ** 2 - 0.5
        denominator = (1.0 + 0.001 * sum_sq) ** 2
        return 0.5 + numerator / denominator

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """
        Get the global minimum point and value for the Schaffer F6 function.

        Parameters
        ----------
        dimension : int
            The dimensionality of the function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        global_min_point = np.zeros(self.dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value


@register("Schaffer_F6_Expanded")
@register("schaffer_f6_expanded")
class Schaffer_F6_Expanded(OptimizationFunction):
    """
    Expanded Schaffer F6 function.

    This function applies the 2D Schaffer F6 function to consecutive pairs
    of variables and sums the results. It's commonly used in CEC benchmarks.

    The function is defined as:
        f(x) = sum(F6(x_i, x_{i+1})) for i = 1 to D-1
             + F6(x_D, x_1)

    where:
        F6(a, b) = 0.5 + (sin^2(sqrt(a^2 + b^2)) - 0.5) / (1 + 0.001*(a^2 + b^2))^2

    Global minimum: f(0, 0, ..., 0) = 0

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, defaults to [-100, 100]^d.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, defaults to [-100, 100]^d.

    References
    ----------
    .. [1] Schaffer, J.D. (1985). "Multiple Objective Optimization with Vector Evaluated Genetic
           Algorithms". In Proceedings of the 1st International Conference on Genetic Algorithms.
           L. Erlbaum Associates Inc., pp. 93-100.
    .. [2] CEC 2005 Special Session on Real-Parameter Optimization
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        # Set default bounds: [-100, 100] for each dimension
        if initialization_bounds is None:
            initialization_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.INITIALIZATION,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        if operational_bounds is None:
            operational_bounds = Bounds(
                low=np.full(dimension, -100.0),
                high=np.full(dimension, 100.0),
                mode=BoundModeEnum.OPERATIONAL,
                qtype=QuantizationTypeEnum.CONTINUOUS,
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Expanded Schaffer F6 function at point x.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (dimension,).

        Returns
        -------
        float
            The function value at x.
        """
        x = self._validate_input(x)

        if self.dimension == 1:
            return 0.0

        result = 0.0
        # Apply F6 to consecutive pairs
        for i in range(self.dimension - 1):
            result += self._schaffer_f6_2d(x[i], x[i + 1])

        # Apply F6 to the wrap-around pair (x_n, x_1)
        result += self._schaffer_f6_2d(x[-1], x[0])

        return result

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Expanded Schaffer F6 function on a batch of points.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_points, dimension).

        Returns
        -------
        np.ndarray
            The function values for each point.
        """
        X = self._validate_batch_input(X)
        n_points, n_dims = X.shape
        results = np.zeros(n_points)

        if n_dims == 1:
            return results  # All zeros for 1D case

        for i in range(n_points):
            x = X[i]
            result = 0.0

            # Apply F6 to consecutive pairs
            for j in range(n_dims - 1):
                result += self._schaffer_f6_2d(x[j], x[j + 1])

            # Apply F6 to the wrap-around pair (x_n, x_1)
            result += self._schaffer_f6_2d(x[-1], x[0])

            results[i] = result

        return results

    @staticmethod
    def _schaffer_f6_2d(x1: float, x2: float) -> float:
        """
        Compute the 2D Schaffer F6 function.

        f(x1, x2) = 0.5 + (sin^2(sqrt(x1^2 + x2^2)) - 0.5) / (1 + 0.001 * (x1^2 + x2^2))^2

        Parameters
        ----------
        x1, x2 : float
            The two input variables.

        Returns
        -------
        float
            The function value.
        """
        sum_sq = x1**2 + x2**2
        sqrt_sum_sq = np.sqrt(sum_sq)
        numerator = np.sin(sqrt_sum_sq) ** 2 - 0.5
        denominator = (1.0 + 0.001 * sum_sq) ** 2
        return 0.5 + numerator / denominator

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """
        Get the global minimum point and value.

        Parameters
        ----------
        dimension : int
            The dimensionality of the function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        global_min_point = np.zeros(self.dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value


@register("Schaffer_1")
@register("Schaffer1")
class Schaffer1Function(OptimizationFunction):
    """
    Schaffer N.1 function — concentric ring landscape.

    f(x,y) = 0.5 + (sin²(sqrt(x² + y²)) - 0.5) / (1 + 0.001(x² + y²))²

    Properties: Continuous, Non-Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: -100 ≤ x_i ≤ 100
    Global minimum: f(0, 0) = 0

    Circularly symmetric — depends only on r² = x² + y².

    References:
        Schaffer, J.D., et al. (1989). "A study of control parameters for
        genetic algorithms." Proc. 3rd Intl. Conf. Genetic Algorithms.
        Jamil, M. & Yang, X.S. (2013). arXiv:1308.4008, #147.
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Schaffer 1 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-100.0, -100.0]),
            high=np.array([100.0, 100.0]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Schaffer N.1: 0.5 + (sin²(sqrt(x²+y²)) - 0.5) / (1+0.001(x²+y²))²."""
        x = self._validate_input(x)
        r2 = x[0] ** 2 + x[1] ** 2
        return float(0.5 + (np.sin(np.sqrt(r2)) ** 2 - 0.5) / (1 + 0.001 * r2) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schaffer N.1 for a batch of points."""
        X = self._validate_batch_input(X)
        r2 = X[:, 0] ** 2 + X[:, 1] ** 2
        return 0.5 + (np.sin(np.sqrt(r2)) ** 2 - 0.5) / (1 + 0.001 * r2) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([0.0, 0.0]), 0.0


@register("Schaffer_2")
@register("Schaffer2")
class Schaffer2Function(OptimizationFunction):
    """
    Schaffer N.2 function — deceptive oscillation variant.

    f(x,y) = 0.5 + (sin²(x² - y²) - 0.5) / (1 + 0.001(x² + y²))²

    Properties: Continuous, Non-Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: -100 ≤ x_i ≤ 100
    Global minimum: f(0, 0) = 0

    Unlike N.1, the numerator uses x²-y² instead of sqrt(x²+y²), making the
    landscape NOT circularly symmetric.

    References:
        Schaffer, J.D., et al. (1989). "A study of control parameters for
        genetic algorithms." Proc. 3rd Intl. Conf. Genetic Algorithms.
        Jamil, M. & Yang, X.S. (2013). arXiv:1308.4008, #148.
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Schaffer 2 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-100.0, -100.0]),
            high=np.array([100.0, 100.0]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Schaffer N.2: 0.5 + (sin²(x²-y²) - 0.5) / (1+0.001(x²+y²))²."""
        x = self._validate_input(x)
        r2 = x[0] ** 2 + x[1] ** 2
        return float(0.5 + (np.sin(x[0] ** 2 - x[1] ** 2) ** 2 - 0.5) / (1 + 0.001 * r2) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schaffer N.2 for a batch of points."""
        X = self._validate_batch_input(X)
        r2 = X[:, 0] ** 2 + X[:, 1] ** 2
        return 0.5 + (np.sin(X[:, 0] ** 2 - X[:, 1] ** 2) ** 2 - 0.5) / (1 + 0.001 * r2) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum."""
        return np.array([0.0, 0.0]), 0.0


@register("Schaffer_4")
@register("Schaffer4")
class Schaffer4Function(OptimizationFunction):
    """
    Schaffer N.4 function — cos-of-sin variant, hardest in the family.

    f(x,y) = 0.5 + (cos²(sin(|x² - y²|)) - 0.5) / (1 + 0.001(x² + y²))²

    Properties: Continuous, Non-Differentiable, Non-Separable, Non-Scalable, Multimodal
    Domain: -100 ≤ x_i ≤ 100
    Global minimum: f ≈ 0.29258 (NOT at the origin; f(0,0) = 1.0)

    The cos²(sin(·)) nesting produces a highly irregular landscape with many
    near-equal local minima.

    References:
        Schaffer, J.D., et al. (1989). "A study of control parameters for
        genetic algorithms." Proc. 3rd Intl. Conf. Genetic Algorithms.
        Jamil, M. & Yang, X.S. (2013). arXiv:1308.4008, #149.
    """

    def __init__(
        self,
        dimension: int = 2,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension != 2:
            raise ValueError("Schaffer 4 function is Non-Scalable and requires dimension=2")

        default_bounds = Bounds(
            low=np.array([-100.0, -100.0]),
            high=np.array([100.0, 100.0]),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Schaffer N.4: 0.5 + (cos²(sin(|x²-y²|)) - 0.5) / (1+0.001(x²+y²))²."""
        x = self._validate_input(x)
        r2 = x[0] ** 2 + x[1] ** 2
        return float(
            0.5 + (np.cos(np.sin(np.abs(x[0] ** 2 - x[1] ** 2))) ** 2 - 0.5) / (1 + 0.001 * r2) ** 2
        )

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schaffer N.4 for a batch of points."""
        X = self._validate_batch_input(X)
        r2 = X[:, 0] ** 2 + X[:, 1] ** 2
        return (
            0.5
            + (np.cos(np.sin(np.abs(X[:, 0] ** 2 - X[:, 1] ** 2))) ** 2 - 0.5)
            / (1 + 0.001 * r2) ** 2
        )

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get global minimum (analytical -- at |x^2-y^2| = pi/2, min r^2)."""
        # Optimum at (+-sqrt(pi/2), 0) or (0, +-sqrt(pi/2))
        return np.array([np.sqrt(np.pi / 2), 0.0]), 0.29258


@register("SchaffersF7")
@register("schaffers_f7")
class SchaffersF7Function(OptimizationFunction):
    """
    Schaffers F7 function.

    Uses adjacent-pair coupling through sqrt(xᵢ² + xᵢ₊₁²) with sinusoidal
    modulation. Not to be confused with Schaffer F6 (a different function).

    sᵢ = √(xᵢ² + xᵢ₊₁²),  i = 1..D-1
    f(x) = (1/(D-1) Σᵢ₌₁ᴰ⁻¹ √sᵢ · (1 + sin²(50·sᵢ^0.2)))²

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-100, 100]^D
    Global minimum: f(0, ..., 0) = 0
    Requires dimension >= 2.

    References
    ----------
    .. [1] Liang, J.J., et al. (2013). "Problem definitions and evaluation criteria
           for the CEC 2013 special session on real-parameter optimization."
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension < 2:
            raise ValueError("Schaffers F7 requires dimension >= 2")
        default_bounds = Bounds(
            low=np.full(dimension, -100.0),
            high=np.full(dimension, 100.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Schaffers F7: (1/(D-1) Σ √sᵢ(1+sin²(50sᵢ^0.2)))²."""
        x = self._validate_input(x)
        s = np.sqrt(x[:-1] ** 2 + x[1:] ** 2)
        terms = np.sqrt(s) * (1.0 + np.sin(50.0 * s**0.2) ** 2)
        return float((np.sum(terms) / (self.dimension - 1)) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schaffers F7 for a batch of points."""
        X = self._validate_batch_input(X)
        s = np.sqrt(X[:, :-1] ** 2 + X[:, 1:] ** 2)
        terms = np.sqrt(s) * (1.0 + np.sin(50.0 * s**0.2) ** 2)
        return (np.sum(terms, axis=1) / (self.dimension - 1)) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Schaffers F7 function.

        Returns
        -------
        tuple[np.ndarray, float]
            (global_min_point, global_min_value)
        """
        return np.zeros(self.dimension), 0.0


@register("SchaffersF7CEC")
@register("schaffers_f7_cec")
@register("schaffer_f7")
class SchaffersF7CECFunction(OptimizationFunction):
    """
    Schaffers F7 function (CEC variant).

    Uses (0.5 + sin²(...)) instead of (1 + sin²(...)). This matches the
    CEC 2013/2017/2020+ C reference implementations.

    sᵢ = √(xᵢ² + xᵢ₊₁²),  i = 1..D-1
    f(x) = (1/(D-1) Σᵢ₌₁ᴰ⁻¹ √sᵢ · (0.5 + sin²(50·sᵢ^0.2)))²

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Multimodal
    Domain: [-100, 100]^D
    Global minimum: f(0, ..., 0) = 0.25 * (D-1)²/(D-1)² = see note
    Requires dimension >= 2.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        if dimension < 2:
            raise ValueError("Schaffers F7 CEC requires dimension >= 2")
        default_bounds = Bounds(
            low=np.full(dimension, -100.0),
            high=np.full(dimension, 100.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Schaffers F7 CEC: (1/(D-1) Σ √sᵢ(0.5+sin²(50sᵢ^0.2)))²."""
        x = self._validate_input(x)
        s = np.sqrt(x[:-1] ** 2 + x[1:] ** 2)
        terms = np.sqrt(s) * (0.5 + np.sin(50.0 * s**0.2) ** 2)
        return float((np.sum(terms) / (self.dimension - 1)) ** 2)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schaffers F7 CEC for a batch of points."""
        X = self._validate_batch_input(X)
        s = np.sqrt(X[:, :-1] ** 2 + X[:, 1:] ** 2)
        terms = np.sqrt(s) * (0.5 + np.sin(50.0 * s**0.2) ** 2)
        return (np.sum(terms, axis=1) / (self.dimension - 1)) ** 2

    def get_global_minimum(self) -> tuple[np.ndarray, float]:
        """Get the global minimum of the Schaffers F7 CEC function."""
        return np.zeros(self.dimension), 0.0
