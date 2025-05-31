"""
Network function implementation.

A mixed-integer network design problem from SPSO 2007/2011 benchmarks.
The problem involves optimizing the placement of Base Station Controllers (BSC)
and their connections to Base Transceiver Stations (BTS) to minimize the 
total connection distance while ensuring each BTS is connected to exactly one BSC.

References:
    .. [1] Clerc, M. (2012). "Standard Particle Swarm Optimisation 2007/2011 - 
           Benchmark Suite and Descriptions". Technical Note, 21 pp. 
           Available at: https://www.researchgate.net/publication/255756848
    .. [2] Zambrano-Bigiarini, M., Clerc, M., & Rojas, R. (2013).
           "Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future 
           PSO improvements". In IEEE Congress on Evolutionary Computation (pp. 2337-2344).
"""

import numpy as np
from pyMOFL.core.bounds import Bounds
from pyMOFL.core.bound_mode_enum import BoundModeEnum
from pyMOFL.core.quantization_type_enum import QuantizationTypeEnum
from pyMOFL.core.function import OptimizationFunction


class NetworkFunction(OptimizationFunction):
    """
    Network design benchmark function (SPSO ID 11).
    
    This problem involves optimizing the placement of Base Station Controllers (BSC)
    and their connections to Base Transceiver Stations (BTS) in a telecom network.
    
    The objective is to minimize the total Euclidean distance between connected
    stations while ensuring that each BTS is connected to exactly one BSC.
    
    The search space includes:
    - Binary variables: Connection matrix between BTS and BSC (1 means there is a link)
    - Continuous variables: 2D positions of the BSCs
    
    Mathematical definition:
    f(x) = sum(distance(BTS_i, BSC_j) for each connected pair (i,j))
           + penalty * (number of constraint violations)
    
    where a constraint violation occurs if a BTS is not connected to exactly one BSC.
    
    Global minimum: Unknown (problem-dependent)
    
    Parameters
    ----------
    initialization_bounds : Bounds, optional
        Bounds for initialization. If None, defaults to [0,1] for binary variables and [0,20] for continuous variables.
    operational_bounds : Bounds, optional
        Bounds for operation. If None, defaults to [0,1] for binary variables and [0,20] for continuous variables.
    
    References
    ----------
    .. [1] Clerc, M. (2012). "Standard Particle Swarm Optimisation 2007/2011 - 
           Benchmark Suite and Descriptions". Technical Note, 21 pp.
    """
    
    def __init__(
        self,
        initialization_bounds: Bounds = None,
        operational_bounds: Bounds = None
    ):
        """
        Initialize the Network function.
        
        Parameters
        ----------
        initialization_bounds : Bounds, optional
            Bounds for initialization. If None, defaults to [0,1] for binary variables and [0,20] for continuous variables.
        operational_bounds : Bounds, optional
            Bounds for operation. If None, defaults to [0,1] for binary variables and [0,20] for continuous variables.
        """
        self.bts_count = 19  # Number of Base Transceiver Stations
        self.bsc_count = 2   # Number of Base Station Controllers
        dimension = (self.bts_count * self.bsc_count) + (2 * self.bsc_count)

        # Default bounds for binary and continuous variables
        binary_dim = self.bts_count * self.bsc_count
        cont_dim = 2 * self.bsc_count
        if initialization_bounds is None:
            # First binary_dim variables are binary (0/1), quantized as INTEGER (binary is a special case)
            init_low = np.concatenate([
                np.zeros(binary_dim),
                np.zeros(cont_dim)
            ])
            init_high = np.concatenate([
                np.ones(binary_dim),
                np.full(cont_dim, 20.0)
            ])
            initialization_bounds = Bounds(
                low=init_low,
                high=init_high,
                mode=BoundModeEnum.INITIALIZATION,
                qtype=np.array([
                    [QuantizationTypeEnum.INTEGER]*binary_dim + [QuantizationTypeEnum.CONTINUOUS]*cont_dim
                ]).flatten()
            )
        if operational_bounds is None:
            oper_low = np.concatenate([
                np.zeros(binary_dim),
                np.zeros(cont_dim)
            ])
            oper_high = np.concatenate([
                np.ones(binary_dim),
                np.full(cont_dim, 20.0)
            ])
            operational_bounds = Bounds(
                low=oper_low,
                high=oper_high,
                mode=BoundModeEnum.OPERATIONAL,
                qtype=np.array([
                    [QuantizationTypeEnum.INTEGER]*binary_dim + [QuantizationTypeEnum.CONTINUOUS]*cont_dim
                ]).flatten()
            )
        super().__init__(
            dimension=dimension,
            initialization_bounds=initialization_bounds,
            operational_bounds=operational_bounds
        )
        
        # Fixed BTS positions (from original implementation)
        self.bts_positions = np.array([
            [6, 9],
            [8, 7],
            [6, 5],
            [10, 5],
            [8, 3],
            [12, 2],
            [4, 7],
            [7, 3],
            [1, 6],
            [8, 2],
            [13, 12],
            [15, 7],
            [15, 11],
            [16, 6],
            [16, 8],
            [18, 9],
            [3, 7],
            [18, 2],
            [20, 17]
        ])
        
        # Penalty factor for constraint violations
        self.penalty = 100.0
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the Network function at point x.
        
        Parameters
        ----------
        x : np.ndarray
            Input vector of shape (dimension,).
            First self.bts_count * self.bsc_count elements are binary (connection matrix),
            last 2 * self.bsc_count elements are continuous (BSC positions).
        Returns
        -------
        float
            The function value at x.
        """
        # Validate and preprocess the input
        x = self._validate_input(x)
        
        # Initialize objective value
        f = 0.0
        
        # Get binary and continuous parts of the solution
        n_binary = self.bts_count * self.bsc_count
        
        # Convert the continuous values to binary for the connection matrix
        # (threshold at 0.5 as in the SPSO implementation)
        binary_part = (x[:n_binary] >= 0.5).astype(float)  
        bsc_positions = x[n_binary:].reshape(self.bsc_count, 2)
        
        # Check constraints: each BTS must connect to exactly one BSC
        for bts_idx in range(self.bts_count):
            # Calculate the sum of connections for this BTS
            connection_sum = 0.0
            
            for bsc_idx in range(self.bsc_count):
                connection_sum += binary_part[bts_idx + (bsc_idx * self.bts_count)]
            
            # Apply penalty if the sum is not 1
            if abs(connection_sum - 1.0) > 1e-10:
                f += self.penalty
        
        # Calculate the total distance between connected stations
        for bsc_idx in range(self.bsc_count):
            for bts_idx in range(self.bts_count):
                # Check if there is a connection between this BTS and BSC
                if binary_part[bts_idx + (bsc_idx * self.bts_count)] < 1.0:
                    continue
                
                # There is a connection - calculate the Euclidean distance
                dx = self.bts_positions[bts_idx, 0] - bsc_positions[bsc_idx, 0]
                dy = self.bts_positions[bts_idx, 1] - bsc_positions[bsc_idx, 1]
                
                # Add the distance to the objective value
                f += np.sqrt(dx*dx + dy*dy)
        
        return float(f)
    
    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the Network function on a batch of points.
        
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
        
        # Initialize results array
        results = np.zeros(X.shape[0])
        
        # Process each point individually
        for i in range(X.shape[0]):
            results[i] = self.evaluate(X[i])
        
        return results 