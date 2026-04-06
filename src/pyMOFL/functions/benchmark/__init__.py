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
- BBOB: LinearSlopeFunction, AttractiveSectorFunction, SharpRidgeFunction, SchwefelSinFunction, GallagherPeaksFunction
- CEC: BentCigarFunction, DiscusFunction, DifferentPowersFunction, etc.
- Engineering: CompressionSpringFunction, NetworkFunction, GearTrainFunction, LennardJonesFunction
- Simple: StepFunction, TripodFunction, PermFunction
- Generators: MultiBasinFunction (GNBG)
"""

# Import function families - clean, consistent naming only
from .ackley import Ackley2Function, Ackley3Function, Ackley4Function, AckleyFunction
from .adjiman import AdjimanFunction
from .alpine import Alpine1Function, Alpine2Function
from .attractive_sector import AttractiveSectorFunction
from .bartels_conn import BartelsConnFunction
from .beale import BealeFunction
from .bent_cigar import BentCigarFunction, DiscusFunction
from .biggs_exp import (
    BiggsExp02Function,
    BiggsExp03Function,
    BiggsExp04Function,
    BiggsExp05Function,
)
from .bird import BirdFunction
from .bohachevsky import Bohachevsky1Function, Bohachevsky2Function, Bohachevsky3Function
from .booth import BoothFunction
from .box_betts import BoxBettsFunction
from .branin import Branin2Function, BraninFunction
from .brent import BrentFunction
from .brown import BrownFunction
from .buche_rastrigin import BucheRastriginFunction
from .bukin import Bukin6Function
from .camel import SixHumpCamelFunction, ThreeHumpCamelFunction
from .chichinadze import ChichinadzeFunction
from .chung_reynolds import ChungReynoldsFunction
from .cola import ColaFunction
from .colville import ColvilleFunction

# Engineering functions
from .compression_spring import CompressionSpringFunction
from .corana import CoranaFunction
from .cosine_mixture import CosineMixtureFunction
from .cross_in_tray import CrossInTrayFunction
from .cross_leg_table import CrossLegTableFunction
from .crowned_cross import CrownedCrossFunction
from .csendes import CsendesFunction
from .damavandi import DamavandiFunction
from .deb01 import Deb01Function
from .deb03 import Deb03Function
from .decanomial import DecanomialFunction
from .deceptive import DeceptiveFunction
from .deckkers_aarts import DeckkersAartsFunction
from .deflected_corrugated_spring import DeflectedCorrugatedSpringFunction
from .devillers_glasser import DeVilliersGlasser01Function, DeVilliersGlasser02Function
from .different_powers import DifferentPowersFunction
from .dixon_price import DixonPriceFunction
from .dolan import DolanFunction
from .drop_wave import DropWaveFunction
from .easom import EasomFunction
from .egg_crate import EggCrateFunction
from .eggholder import EggholderFunction
from .el_attar import ElAttarVidyasagarDuttaFunction
from .elliptic import HighConditionedElliptic
from .exp2 import Exp2Function
from .exponential_function import ExponentialFunction
from .freudenstein_roth import FreudensteinRothFunction
from .gallagher_peaks import GallagherPeaksFunction
from .gear_train import GearTrainFunction
from .giunta import GiuntaFunction
from .goldstein_price import GoldsteinPriceFunction
from .griewank import GriewankFunction
from .gulf import GulfFunction
from .hansen import HansenFunction
from .happycat import HappyCatFunction, HGBatFunction
from .hartmann import Hartmann3Function, Hartmann6Function
from .helical_valley import HelicalValleyFunction
from .himmelblau import HimmelblauFunction
from .holder_table import HolderTableFunction
from .hosaki import HosakiFunction
from .jennrich_sampson import JennrichSampsonFunction
from .katsuura import KatsuuraFunction
from .keane import KeaneFunction
from .kowalik import KowalikFunction
from .langermann import LangermannFunction
from .lennard_jones import LennardJonesFunction
from .leon import LeonFunction
from .levy import LevyCEC2022Function, LevyCECFunction, LevyFunction  # noqa: F401
from .linear_slope import LinearSlopeFunction
from .lunacek import LunacekBiRastriginCECFunction, LunacekBiRastriginFunction
from .matyas import MatyasFunction
from .max_absolute import MaxAbsolute
from .mccormick import McCormickFunction
from .michalewicz import MichalewiczFunction
from .miele_cantrell import MieleCantrellFunction
from .mishra import (
    Mishra01Function,
    Mishra02Function,
    Mishra03Function,
    Mishra04Function,
    Mishra05Function,
    Mishra06Function,
    Mishra07Function,
    Mishra08Function,
    Mishra09Function,
    Mishra10Function,
    Mishra11Function,
)
from .multi_basin import MultiBasinFunction, create_basin_component
from .multi_modal_func import MultiModalFunction
from .needle_eye import NeedleEyeFunction
from .network import NetworkFunction
from .new_function import NewFunction01Function, NewFunction02Function
from .odd_square import OddSquareFunction
from .parsopoulos import ParsopoulosFunction
from .perm import PermFunction

# New function families
from .powell import PowellSingular2Function, PowellSingularFunction, PowellSumFunction
from .qing import QingFunction
from .quartic import QuarticFunction
from .quintic import QuinticFunction
from .rana import RanaFunction
from .rastrigin import RastriginFunction
from .rosenbrock import GriewankOfRosenbrock, RosenbrockFunction
from .salomon import SalomonFunction
from .schaffer import (
    Schaffer1Function,
    Schaffer2Function,
    Schaffer4Function,
    Schaffer_F6,
    Schaffer_F6_Expanded,
    SchaffersF7CECFunction,  # noqa: F401
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
from .schwefel_sin import SchwefelSinFunction
from .sharp_ridge import SharpRidgeFunction
from .sphere import SphereFunction

# Simple functions
from .step import StepFunction
from .step_ellipsoid import StepEllipsoidFunction
from .styblinski_tang import StyblinskiTangFunction
from .sum_different_powers import SumDifferentPowersFunction
from .test_tube_holder import TestTubeHolderFunction
from .tripod import TripodFunction
from .ursem01 import Ursem01Function
from .venter_sobieski import VenterSobieskiFunction
from .weierstrass import WeierstrassFunction
from .xin_she_yang01 import XinSheYang01Function
from .zakharov import ZakharovFunction
from .zero_sum import ZeroSumFunction
from .zettl import ZettlFunction
from .zimmerman import ZimmermanFunction
from .zirilli import ZirilliFunction

__all__ = [
    "Ackley2Function",
    "Ackley3Function",
    "Ackley4Function",
    "AckleyFunction",
    "AdjimanFunction",
    "Alpine1Function",
    "Alpine2Function",
    "AttractiveSectorFunction",
    "BartelsConnFunction",
    "BealeFunction",
    "BentCigarFunction",
    "BiggsExp02Function",
    "BiggsExp03Function",
    "BiggsExp04Function",
    "BiggsExp05Function",
    "BirdFunction",
    "Bohachevsky1Function",
    "Bohachevsky2Function",
    "Bohachevsky3Function",
    "BoothFunction",
    "BoxBettsFunction",
    "Branin2Function",
    "BraninFunction",
    "BrentFunction",
    "BrownFunction",
    "BucheRastriginFunction",
    "Bukin6Function",
    "ChichinadzeFunction",
    "ChungReynoldsFunction",
    "ColaFunction",
    "ColvilleFunction",
    "CompressionSpringFunction",
    "CoranaFunction",
    "CosineMixtureFunction",
    "CrossInTrayFunction",
    "CrossLegTableFunction",
    "CrownedCrossFunction",
    "CsendesFunction",
    "DamavandiFunction",
    "DeVilliersGlasser01Function",
    "DeVilliersGlasser02Function",
    "Deb01Function",
    "Deb03Function",
    "DecanomialFunction",
    "DeceptiveFunction",
    "DeckkersAartsFunction",
    "DeflectedCorrugatedSpringFunction",
    "DifferentPowersFunction",
    "DiscusFunction",
    "DixonPriceFunction",
    "DolanFunction",
    "DropWaveFunction",
    "EasomFunction",
    "EggCrateFunction",
    "EggholderFunction",
    "ElAttarVidyasagarDuttaFunction",
    "Exp2Function",
    "ExponentialFunction",
    "FreudensteinRothFunction",
    "GallagherPeaksFunction",
    "GearTrainFunction",
    "GiuntaFunction",
    "GoldsteinPriceFunction",
    "GriewankFunction",
    "GriewankOfRosenbrock",
    "GulfFunction",
    "HGBatFunction",
    "HansenFunction",
    "HappyCatFunction",
    "Hartmann3Function",
    "Hartmann6Function",
    "HelicalValleyFunction",
    "HighConditionedElliptic",
    "HimmelblauFunction",
    "HolderTableFunction",
    "HosakiFunction",
    "JennrichSampsonFunction",
    "KatsuuraFunction",
    "KeaneFunction",
    "KowalikFunction",
    "LangermannFunction",
    "LennardJonesFunction",
    "LeonFunction",
    "LevyCEC2022Function",
    "LevyFunction",
    "LinearSlopeFunction",
    "LunacekBiRastriginCECFunction",
    "LunacekBiRastriginFunction",
    "MatyasFunction",
    "MaxAbsolute",
    "McCormickFunction",
    "MichalewiczFunction",
    "MieleCantrellFunction",
    "Mishra01Function",
    "Mishra02Function",
    "Mishra03Function",
    "Mishra04Function",
    "Mishra05Function",
    "Mishra06Function",
    "Mishra07Function",
    "Mishra08Function",
    "Mishra09Function",
    "Mishra10Function",
    "Mishra11Function",
    "MultiBasinFunction",
    "MultiModalFunction",
    "NeedleEyeFunction",
    "NetworkFunction",
    "NewFunction01Function",
    "NewFunction02Function",
    "OddSquareFunction",
    "ParsopoulosFunction",
    "PermFunction",
    "PowellSingular2Function",
    "PowellSingularFunction",
    "PowellSumFunction",
    "QingFunction",
    "QuarticFunction",
    "QuinticFunction",
    "RanaFunction",
    "RastriginFunction",
    "RosenbrockFunction",
    "SalomonFunction",
    "Schaffer1Function",
    "Schaffer2Function",
    "Schaffer4Function",
    "Schaffer_F6",
    "Schaffer_F6_Expanded",
    "SchaffersF7Function",
    "SchwefelFunction",
    "SchwefelSinFunction",
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
    "SharpRidgeFunction",
    "SixHumpCamelFunction",
    "SphereFunction",
    "StepEllipsoidFunction",
    "StepFunction",
    "StyblinskiTangFunction",
    "SumDifferentPowersFunction",
    "TestTubeHolderFunction",
    "ThreeHumpCamelFunction",
    "TripodFunction",
    "Ursem01Function",
    "VenterSobieskiFunction",
    "WeierstrassFunction",
    "XinSheYang01Function",
    "ZakharovFunction",
    "ZeroSumFunction",
    "ZettlFunction",
    "ZimmermanFunction",
    "ZirilliFunction",
    "create_basin_component",
]
