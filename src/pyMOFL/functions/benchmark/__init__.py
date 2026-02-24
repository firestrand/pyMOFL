"""
Benchmark functions for optimization algorithm testing.

This module contains core mathematical benchmark functions organized by families.
Functions include metadata indicating their properties (unimodal/multimodal,
separable/non-separable, etc.).

Function Families:
- Schwefel: Schwefel_1_2, Schwefel_2_6, Schwefel_2_13
- Schaffer: Schaffer_F6, Schaffer_F6_Expanded
- Sphere: SphereFunction
- Rosenbrock: RosenbrockFunction, GriewankOfRosenbrock
- Rastrigin: RastriginFunction
- Griewank: GriewankFunction
- Ackley: AckleyFunction
- Weierstrass: WeierstrassFunction
- Elliptic: HighConditionedElliptic
- CEC: BentCigarFunction, DiscusFunction, DifferentPowersFunction, etc.
- Engineering: CompressionSpringFunction, NetworkFunction, GearTrainFunction, LennardJonesFunction
- Simple: StepFunction, TripodFunction, PermFunction
"""

# Import function families - clean, consistent naming only
from .ackley import Ackley2Function, Ackley3Function, Ackley4Function, AckleyFunction
from .alpine import Alpine1Function, Alpine2Function
from .bent_cigar import BentCigarFunction, DiscusFunction
from .branin import Branin2Function, BraninFunction

# Engineering functions
from .compression_spring import CompressionSpringFunction
from .different_powers import DifferentPowersFunction
from .dixon_price import DixonPriceFunction
from .easom import EasomFunction
from .elliptic import HighConditionedElliptic
from .gear_train import GearTrainFunction
from .goldstein_price import GoldsteinPriceFunction
from .griewank import GriewankFunction
from .happycat import HappyCatFunction, HGBatFunction
from .himmelblau import HimmelblauFunction
from .katsuura import KatsuuraFunction
from .lennard_jones import LennardJonesFunction
from .levy import LevyFunction
from .lunacek import LunacekBiRastriginFunction
from .matyas import MatyasFunction
from .max_absolute import MaxAbsolute
from .mccormick import McCormickFunction
from .network import NetworkFunction
from .perm import PermFunction

# New function families
from .powell import PowellSingular2Function, PowellSingularFunction, PowellSumFunction
from .rastrigin import RastriginFunction
from .rosenbrock import GriewankOfRosenbrock, RosenbrockFunction
from .schaffer import (
    Schaffer1Function,
    Schaffer2Function,
    Schaffer4Function,
    Schaffer_F6,
    Schaffer_F6_Expanded,
    SchaffersF7Function,
)
from .schwefel import (
    Schwefel_1_2,
    Schwefel_2_4,
    Schwefel_2_6,
    Schwefel_2_13,
    Schwefel_2_20,
    Schwefel_2_21,
    Schwefel_2_22,
    Schwefel_2_23,
    Schwefel_2_25,
    Schwefel_2_26,
    Schwefel_2_36,
    SchwefelFunction,
)
from .sphere import SphereFunction

# Simple functions
from .step import StepFunction
from .sum_different_powers import SumDifferentPowersFunction
from .tripod import TripodFunction
from .weierstrass import WeierstrassFunction
from .zakharov import ZakharovFunction

__all__ = [
    "Ackley2Function",
    "Ackley3Function",
    "Ackley4Function",
    "AckleyFunction",
    "Alpine1Function",
    "Alpine2Function",
    "BentCigarFunction",
    "Branin2Function",
    "BraninFunction",
    "CompressionSpringFunction",
    "DifferentPowersFunction",
    "DiscusFunction",
    "DixonPriceFunction",
    "EasomFunction",
    "GearTrainFunction",
    "GoldsteinPriceFunction",
    "GriewankFunction",
    "GriewankOfRosenbrock",
    "HGBatFunction",
    "HappyCatFunction",
    "HighConditionedElliptic",
    "HimmelblauFunction",
    "KatsuuraFunction",
    "LennardJonesFunction",
    "LevyFunction",
    "LunacekBiRastriginFunction",
    "MatyasFunction",
    "MaxAbsolute",
    "McCormickFunction",
    "NetworkFunction",
    "PermFunction",
    "PowellSingular2Function",
    "PowellSingularFunction",
    "PowellSumFunction",
    "RastriginFunction",
    "RosenbrockFunction",
    "Schaffer1Function",
    "Schaffer2Function",
    "Schaffer4Function",
    "Schaffer_F6",
    "Schaffer_F6_Expanded",
    "SchaffersF7Function",
    "SchwefelFunction",
    "Schwefel_1_2",
    "Schwefel_2_4",
    "Schwefel_2_6",
    "Schwefel_2_13",
    "Schwefel_2_20",
    "Schwefel_2_21",
    "Schwefel_2_22",
    "Schwefel_2_23",
    "Schwefel_2_25",
    "Schwefel_2_26",
    "Schwefel_2_36",
    "SphereFunction",
    "StepFunction",
    "SumDifferentPowersFunction",
    "TripodFunction",
    "WeierstrassFunction",
    "ZakharovFunction",
]
