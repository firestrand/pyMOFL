"""
Schwefel functions implementation.

This module contains all variants of Schwefel functions used in optimization benchmarks.
Each variant is named consistently using the Schwefel_X_Y convention.

References
----------
.. [1] Schwefel, H. P. (1993). Evolution and optimum seeking. John Wiley & Sons, Inc.
.. [2] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
       "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
       optimization". Nanyang Technological University, Singapore, Tech. Rep.
.. [3] Jamil, M., & Yang, X.S. (2013). "A literature survey of benchmark functions for global
       optimization problems". International Journal of Mathematical Modelling and Numerical
       Optimisation, 4(2), 150-194. arXiv:1308.4008
       Local documentation: docs/literature_schwefel/jamil_yang_2013_literature_survey.md
.. [4] Ding, K., & Tan, Y. (2014). "A CUDA-Based Real Parameter Optimization Benchmark".
       arXiv:1407.7737
       Local documentation: docs/literature_schwefel/ding_tan_2014_cuda_benchmark.md
.. [5] Yang, X.S. (2023). "Ten New Benchmarks for Optimization". arXiv:2309.00644
       Local documentation: docs/literature_schwefel/yang_2023_ten_benchmarks.md
"""

import numpy as np

from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.function import OptimizationFunction
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.registry import register


@register("Schwefel_1_2")
class Schwefel_1_2(OptimizationFunction):
    """
    Schwefel 1.2 function.

    The function is defined as:
        f(x) = Σ_{i=1}^n (Σ_{j=1}^i x_j)^2

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
    .. [1] Schwefel, H. P. (1993). Evolution and optimum seeking. John Wiley & Sons, Inc.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        # Sensible defaults: [-100, 100] for each dimension
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
        """Compute the Schwefel 1.2 function value."""
        x = self._validate_input(x)
        cumsum = np.cumsum(x)
        return float(np.sum(cumsum**2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schwefel 1.2 function for batch."""
        X = self._validate_batch_input(X)
        cumsum = np.cumsum(X, axis=1)
        return np.sum(cumsum**2, axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        """
        Get the global minimum point and value for the Schwefel 1.2 function.

        Parameters
        ----------
        dimension : int
            The dimensionality of the function.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)
        """
        global_min_point = np.zeros(dimension)
        global_min_value = 0.0
        return global_min_point, global_min_value


@register("Schwefel_2_6")
class Schwefel_2_6(OptimizationFunction):
    """
    Schwefel 2.6 function.

    The function is defined as:
        f(x) = max_i |A_i @ x - B_i|

    where A is a matrix and B is a vector, often defined such that the global minimum is displaced
    from the origin.

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    A : np.ndarray
        The A matrix of shape (dimension, dimension).
    B : np.ndarray
        The B vector of shape (dimension,).
    initialization_bounds : Bounds, optional
        Bounds for random initialization. If None, defaults to [-100, 100]^d.
    operational_bounds : Bounds, optional
        Bounds for domain enforcement. If None, defaults to [-100, 100]^d.
    optimum_point : np.ndarray, optional
        The global optimum point for reference.

    References
    ----------
    .. [1] Schwefel, H. P. (1993). Evolution and optimum seeking. John Wiley & Sons, Inc.
    .. [2] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    """

    def __init__(
        self,
        dimension: int,
        A: np.ndarray,
        B: np.ndarray | None = None,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        optimum_point: np.ndarray | None = None,
        # F5 special parameters for B computation
        shift: np.ndarray | None = None,
        optimum_on_bounds: bool = False,
        compute_B: bool = False,
        **kwargs,
    ):
        # Default bounds: [-100, 100] for each dimension
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
        self.A = np.array(A)

        # Compute B if needed (F5 pattern)
        if B is not None:
            self.B = np.array(B)
        elif compute_B and shift is not None:
            if optimum_on_bounds:
                # F5: Use BoundsShiftOptimumPattern
                from pyMOFL.core.bounds_optimum_transform import BoundsShiftOptimumPattern

                pattern = BoundsShiftOptimumPattern()
                optimum = pattern.construct_optimum(dimension, shift)
                self.B = np.dot(self.A, optimum)
            else:
                # Standard: B = A @ shift
                self.B = np.dot(self.A, shift)
        else:
            raise ValueError(
                "Schwefel_2_6 requires either B parameter or (compute_B=True and shift)"
            )

        self.optimum_point = np.array(optimum_point) if optimum_point is not None else None

        # Validate dimensions
        if self.A.shape != (dimension, dimension):
            raise ValueError(f"A matrix shape {self.A.shape} doesn't match dimension {dimension}")
        if self.B.shape != (dimension,):
            raise ValueError(f"B vector shape {self.B.shape} doesn't match dimension {dimension}")

    def evaluate(self, x: np.ndarray) -> float:
        """Compute the Schwefel 2.6 function value."""
        x = self._validate_input(x)
        Ax = np.dot(self.A, x)
        diff = np.abs(Ax - self.B)
        return float(np.max(diff))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schwefel 2.6 function for batch."""
        X = self._validate_batch_input(X)
        Ax = np.dot(X, self.A.T)
        diff = np.abs(Ax - self.B[np.newaxis, :])
        return np.max(diff, axis=1)

    def get_global_minimum(self):
        """
        Get the global minimum point and value for the Schwefel 2.6 function.

        Returns
        -------
        tuple
            (optimum_point, 0.0) if optimum_point is set, else raises NotImplementedError.

        Raises
        ------
        NotImplementedError
            If the optimum point is not set.
        """
        if self.optimum_point is not None:
            return self.optimum_point, 0.0
        else:
            raise NotImplementedError("Global minimum is problem-dependent for Schwefel 2.6.")


@register("Schwefel_2_13")
class Schwefel_2_13(OptimizationFunction):
    """
    Schwefel's Problem 2.13 function: f(x) = sum((A_i - B_i(x))^2)

    Where:
    A_i = sum(a_ij * sin(alpha_j) + b_ij * cos(alpha_j))
    B_i(x) = sum(a_ij * sin(x_j) + b_ij * cos(x_j))

    This function is multimodal and non-separable.

    Global minimum: f(alpha) = 0, where alpha is the predefined optimum point.

    Parameters
    ----------
    dimension : int
        The dimensionality of the function.
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, defaults to [-π, π]^d.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, defaults to [-π, π]^d.
    a : np.ndarray, optional
        D×D coefficient matrix for sine terms. If None, random integers in [-100, 100] are used.
    b : np.ndarray, optional
        D×D coefficient matrix for cosine terms. If None, random integers in [-100, 100] are used.
    alpha : np.ndarray, optional
        Global optimum point. If None, random values in [-π, π] are used.

    References
    ----------
    .. [1] Suganthan, P. N., Hansen, N., Liang, J. J., Deb, K., Chen, Y. P., Auger, A., & Tiwari, S. (2005).
           "Problem definitions and evaluation criteria for the CEC 2005 special session on real-parameter
           optimization". Nanyang Technological University, Singapore, Tech. Rep.
    .. [2] Schwefel, H.-P. (1981). "Numerical optimization of computer models".
           John Wiley & Sons, Inc.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        a: np.ndarray | None = None,
        b: np.ndarray | None = None,
        alpha: np.ndarray | None = None,
    ):
        """
        Initialize the Schwefel's Problem 2.13 function.

        Parameters
        ----------
        dimension : int
            The dimensionality of the function.
        initialization_bounds : Bounds, optional
            Bounds for initialization. If None, defaults to [-π, π]^d.
        operational_bounds : Bounds, optional
            Bounds for operation. If None, defaults to [-π, π]^d.
        a : np.ndarray, optional
            D×D coefficient matrix for sine terms. If None, random integers in [-100, 100] are used.
        b : np.ndarray, optional
            D×D coefficient matrix for cosine terms. If None, random integers in [-100, 100] are used.
        alpha : np.ndarray, optional
            Global optimum point. If None, random values in [-π, π] are used.
        """
        default_bounds = Bounds(
            low=np.full(dimension, -np.pi),
            high=np.full(dimension, np.pi),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
        )

        # Initialize coefficient matrices if not provided
        if a is None:
            # Random integers in [-100, 100]
            a = np.random.randint(-100, 101, (dimension, dimension))

        if b is None:
            # Random integers in [-100, 100]
            b = np.random.randint(-100, 101, (dimension, dimension))

        # Initialize optimum point if not provided
        if alpha is None:
            alpha = np.random.uniform(-np.pi, np.pi, dimension)

        self.a = a
        self.b = b
        self.alpha = alpha

        # Precompute A_i values
        self.A = np.zeros(dimension)
        for i in range(dimension):
            for j in range(dimension):
                self.A[i] += a[i, j] * np.sin(alpha[j]) + b[i, j] * np.cos(alpha[j])

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Schwefel's Problem 2.13 function at point x.

        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (dimension,).
        Returns
        -------
        float
            The function value at x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)

        result = 0.0
        for i in range(self.dimension):
            # Calculate B_i(x)
            B_i = 0.0
            for j in range(self.dimension):
                B_i += self.a[i, j] * np.sin(x[j]) + self.b[i, j] * np.cos(x[j])

            # Add squared difference to result
            result += (self.A[i] - B_i) ** 2

        return float(result)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Schwefel's Problem 2.13 function on a batch of points.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_points, dimension).
        Returns
        -------
        np.ndarray
            The function values for each point.
        """
        # Validate the batch input
        X = self._validate_batch_input(X)

        n_points, n_dims = X.shape
        results = np.zeros(n_points)

        # For each point in the batch
        for p in range(n_points):
            x = X[p]
            result = 0.0

            for i in range(n_dims):
                # Calculate B_i(x)
                B_i = 0.0
                for j in range(n_dims):
                    B_i += self.a[i, j] * np.sin(x[j]) + self.b[i, j] * np.cos(x[j])

                # Add squared difference to result
                result += (self.A[i] - B_i) ** 2

            results[p] = result

        return results

    @staticmethod
    def get_global_minimum(
        dimension: int, a: np.ndarray, b: np.ndarray, alpha: np.ndarray
    ) -> tuple:
        """
        Get the global minimum of the function for given parameters.

        Parameters
        ----------
        dimension : int
            The dimension of the function.
        a : np.ndarray
            D×D coefficient matrix for sine terms.
        b : np.ndarray
            D×D coefficient matrix for cosine terms.
        alpha : np.ndarray
            Global optimum point.

        Returns
        -------
        tuple
            (global_min_point, global_min_value)

        Notes
        -----
        The global minimum is at alpha, with value 0.0 for the given a, b, alpha.
        For reproducibility, the same a, b, alpha must be used as in the function instance.
        """
        return alpha, 0.0


class SchwefelFunction(OptimizationFunction):
    """
    Schwefel function (offset form of Problem 2.26).

    f(x) = 418.9829*D - Σ x_i * sin(sqrt(|x_i|))

    This is the normalized form where the global minimum is near zero.
    For the pure form (f* ≈ -418.983*D), see Schwefel_2_26.

    Global minimum: f(420.9687, ..., 420.9687) ≈ 0

    Properties: Continuous, Differentiable, Separable, Scalable, Multimodal

    References:
        .. [1] Schwefel, H. P. (1993). Evolution and optimum seeking.
        .. [2] Jamil, M., & Yang, X.S. (2013). #128 in literature survey. arXiv:1308.4008
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -500.0),
            high=np.full(dimension, 500.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds or default_bounds,
            operational_bounds=operational_bounds or default_bounds,
            **kwargs,
        )
        self._offset = 418.9829 * dimension

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Schwefel function: 418.9829*D - Σ x_i sin(sqrt(|x_i|))."""
        x = self._validate_input(x)
        return float(self._offset - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schwefel function for a batch of points."""
        X = self._validate_batch_input(X)
        return self._offset - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        return np.full(dimension, 420.9687), 0.0


class Schwefel_2_4(OptimizationFunction):
    """
    Schwefel 2.4 function (Extended Rosenbrock with star dependency on x_1).

    f(x) = Σ_{i=2}^{D} [(x_1 - x_i²)² + (x_i - 1)²]

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Unimodal
    Domain: -10 ≤ x_i ≤ 10
    Global minimum: f(1, 1, ..., 1) = 0

    x_1 acts as the "anchor" — all other terms depend on it, creating a
    star-shaped dependency structure. Getting x_1 wrong amplifies error
    in all other dimensions.

    References:
        Schwefel, H.P. (1981). "Numerical Optimization of Computer Models".
    """

    def __init__(self, dimension: int, **kwargs):
        if dimension < 2:
            raise ValueError("Schwefel 2.4 requires dimension >= 2")
        default_bounds = Bounds(
            low=np.full(dimension, -10.0),
            high=np.full(dimension, 10.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=kwargs.pop("initialization_bounds", None) or default_bounds,
            operational_bounds=kwargs.pop("operational_bounds", None) or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Schwefel 2.4: Σ_{i=2}^{D} [(x_1 - x_i²)² + (x_i - 1)²]."""
        x = self._validate_input(x)
        x1 = x[0]
        x_rest = x[1:]
        return float(np.sum((x1 - x_rest**2) ** 2 + (x_rest - 1) ** 2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schwefel 2.4 for a batch of points."""
        X = self._validate_batch_input(X)
        x1 = X[:, 0:1]  # shape (N, 1)
        x_rest = X[:, 1:]  # shape (N, D-1)
        return np.sum((x1 - x_rest**2) ** 2 + (x_rest - 1) ** 2, axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        return np.ones(dimension), 0.0


class Schwefel_2_20(OptimizationFunction):
    """
    Schwefel 2.20 function: f(x) = Σ |x_i|

    The absolute value sum (Manhattan / L1 norm). Non-differentiable at the origin.

    Global minimum: f(0, ..., 0) = 0

    Properties: Continuous, Non-Differentiable, Separable, Scalable, Unimodal

    References:
        .. [1] Schwefel, H. P. (1981). "Numerical Optimization of Computer Models".
        .. [2] Jamil, M., & Yang, X.S. (2013). #122 in literature survey.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
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
        """Compute Schwefel 2.20: Σ |x_i|."""
        x = self._validate_input(x)
        return float(np.sum(np.abs(x)))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schwefel 2.20 for a batch of points."""
        X = self._validate_batch_input(X)
        return np.sum(np.abs(X), axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        return np.zeros(dimension), 0.0


class Schwefel_2_21(OptimizationFunction):
    """
    Schwefel 2.21 function: f(x) = max_i |x_i|

    The Chebyshev / L∞ norm. Isolates the worst-performing dimension.

    Global minimum: f(0, ..., 0) = 0

    Properties: Continuous, Non-Differentiable, Non-Separable, Scalable, Unimodal

    References:
        .. [1] Schwefel, H. P. (1981). "Numerical Optimization of Computer Models".
        .. [2] Jamil, M., & Yang, X.S. (2013). #123 in literature survey.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
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
        """Compute Schwefel 2.21: max_i |x_i|."""
        x = self._validate_input(x)
        return float(np.max(np.abs(x)))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schwefel 2.21 for a batch of points."""
        X = self._validate_batch_input(X)
        return np.max(np.abs(X), axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        return np.zeros(dimension), 0.0


class Schwefel_2_22(OptimizationFunction):
    """
    Schwefel 2.22 function: f(x) = Σ |x_i| + Π |x_i|

    Sum of absolute values plus product of absolute values.
    The product term makes this significantly harder than a simple L1 norm.

    Global minimum: f(0, ..., 0) = 0

    Properties: Continuous, Non-Differentiable, Non-Separable, Scalable, Unimodal

    References:
        .. [1] Schwefel, H. P. (1981). "Numerical Optimization of Computer Models".
        .. [2] Jamil, M., & Yang, X.S. (2013). #124 in literature survey.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
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
        """Compute Schwefel 2.22: Σ|x_i| + Π|x_i|."""
        x = self._validate_input(x)
        a = np.abs(x)
        return float(np.sum(a) + np.prod(a))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schwefel 2.22 for a batch of points."""
        X = self._validate_batch_input(X)
        A = np.abs(X)
        return np.sum(A, axis=1) + np.prod(A, axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        return np.zeros(dimension), 0.0


class Schwefel_2_23(OptimizationFunction):
    """
    Schwefel 2.23 function: f(x) = Σ x_i^10

    The 10th-power sum. Gradient is essentially zero for a large portion of
    the search space, trapping gradient-based solvers.

    Global minimum: f(0, ..., 0) = 0

    Properties: Continuous, Differentiable, Separable, Scalable, Unimodal

    References:
        .. [1] Schwefel, H. P. (1981). "Numerical Optimization of Computer Models".
        .. [2] Jamil, M., & Yang, X.S. (2013). #125/#126 in literature survey.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -10.0),
            high=np.full(dimension, 10.0),
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
        """Compute Schwefel 2.23: Σ x_i^10."""
        x = self._validate_input(x)
        return float(np.sum(x**10))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schwefel 2.23 for a batch of points."""
        X = self._validate_batch_input(X)
        return np.sum(X**10, axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        return np.zeros(dimension), 0.0


class Schwefel_2_25(OptimizationFunction):
    """
    Schwefel 2.25 function: f(x) = Σ_{i=1}^{D-1} (x_i² - x_{i+1})² + (x_i - 1)²

    A Rosenbrock-like variant without the 100x scaling factor.
    Tests navigation of a curved, narrow valley with variable coupling.

    Global minimum: f(1, 1, ..., 1) = 0

    Properties: Continuous, Differentiable, Non-Separable, Scalable, Unimodal

    References:
        .. [1] Schwefel, H. P. (1981). "Numerical Optimization of Computer Models".
        .. [2] Jamil, M., & Yang, X.S. (2013). #127 in literature survey.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, 0.0),
            high=np.full(dimension, 10.0),
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
        """Compute Schwefel 2.25: Σ (x_i² - x_{i+1})² + (x_i - 1)²."""
        x = self._validate_input(x)
        return float(np.sum((x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schwefel 2.25 for a batch of points."""
        X = self._validate_batch_input(X)
        return np.sum((X[:, :-1] ** 2 - X[:, 1:]) ** 2 + (X[:, :-1] - 1) ** 2, axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        return np.ones(dimension), 0.0


class Schwefel_2_26(OptimizationFunction):
    """
    Schwefel 2.26 function: f(x) = -Σ x_i sin(sqrt(|x_i|))

    The "standard" Schwefel function (pure form, no offset).
    Famous for its deceptive landscape where the second-best local
    optimum is geometrically far from the global one.

    Global minimum: f(420.9687, ..., 420.9687) ≈ -418.983 * D

    Properties: Continuous, Differentiable, Separable, Scalable, Multimodal

    References:
        .. [1] Schwefel, H. P. (1981). "Numerical Optimization of Computer Models".
        .. [2] Jamil, M., & Yang, X.S. (2013). #128 in literature survey.
    """

    def __init__(
        self,
        dimension: int,
        initialization_bounds: Bounds | None = None,
        operational_bounds: Bounds | None = None,
        **kwargs,
    ):
        default_bounds = Bounds(
            low=np.full(dimension, -500.0),
            high=np.full(dimension, 500.0),
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
        """Compute Schwefel 2.26: -Σ x_i sin(sqrt(|x_i|))."""
        x = self._validate_input(x)
        return float(-np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schwefel 2.26 for a batch of points."""
        X = self._validate_batch_input(X)
        return -np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        return np.full(dimension, 420.9687), -418.9829 * dimension


class Schwefel_2_36(OptimizationFunction):
    """
    Schwefel 2.36 function (sine-root sum on non-negative domain).

    f(x) = Σ_{i=1}^{D} -x_i * sin(sqrt(|x_i|))

    Same formula as Schwefel 2.26 but with domain restricted to [0, 500].

    Properties: Continuous, Differentiable, Separable, Scalable, Multimodal
    Domain: 0 ≤ x_i ≤ 500
    Global minimum: f(420.9687, ..., 420.9687) ≈ -418.9829 × D

    Note
    ----
    Jamil & Yang 2013 (#129) lists f* = -3456 at x* = (12,...,12), which is
    a known legacy error in the survey. The correct optimum is the sine-root
    form per Schwefel (1981).

    References:
        Schwefel, H.P. (1981). "Numerical Optimization of Computer Models".
    """

    def __init__(self, dimension: int, **kwargs):
        default_bounds = Bounds(
            low=np.full(dimension, 0.0),
            high=np.full(dimension, 500.0),
            mode=BoundModeEnum.OPERATIONAL,
            qtype=QuantizationTypeEnum.CONTINUOUS,
        )
        super().__init__(
            dimension=dimension,
            initialization_bounds=kwargs.pop("initialization_bounds", None) or default_bounds,
            operational_bounds=kwargs.pop("operational_bounds", None) or default_bounds,
            **kwargs,
        )

    def evaluate(self, x: np.ndarray) -> float:
        """Compute Schwefel 2.36: -Σ x_i sin(sqrt(|x_i|))."""
        x = self._validate_input(x)
        return float(-np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute Schwefel 2.36 for a batch of points."""
        X = self._validate_batch_input(X)
        return -np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)

    @staticmethod
    def get_global_minimum(dimension: int) -> tuple:
        return np.full(dimension, 420.9687), -418.9829 * dimension
