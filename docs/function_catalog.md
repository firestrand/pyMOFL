# pyMOFL Benchmark Function Catalog

This catalog provides a comprehensive index of all **175 concrete benchmark function classes** and **336 registered component aliases** in `pyMOFL`.

## Quick Usage

Functions can be instantiated either by direct class import or dynamically via the component registry:

```python
# 1. Direct class import
from pyMOFL.functions.benchmark import SphereFunction, RastriginFunction
f1 = SphereFunction(dimension=10)

# 2. Dynamic lookup via registry alias
from pyMOFL.registry import get
SphereCls = get("sphere")
f2 = SphereCls(dimension=10)
```

## Categories

- [Scalable Functions (52 functions)](#scalable-functions)
- [BBOB Primitives (7 functions)](#bbob-primitives)
- [Fixed 2D Functions (69 functions)](#fixed-2d-functions)
- [Fixed Dimension (3D-6D) Functions (17 functions)](#fixed-dimension-3d-6d-functions)
- [Mishra Family (11 functions)](#mishra-family)
- [Schwefel Family (13 functions)](#schwefel-family)
- [Engineering & Special Benchmarks (6 functions)](#engineering-special-benchmarks)

---

## Scalable Functions

| Function Class | Registry Aliases | Dimension | Source File | Description |
|:---|:---|:---:|:---|:---|
| `Ackley4Function` | `Ackley4`, `ackley4`, `ackley_4` | Scalable (D ≥ 1) | [ackley.py](../src/pyMOFL/functions/benchmark/ackley.py) | Ackley 4 (Modified Ackley) function: |
| `AckleyFunction` | `Ackley`, `ackley` | Scalable (D ≥ 1) | [ackley.py](../src/pyMOFL/functions/benchmark/ackley.py) | Ackley function: f(x) = -20·exp(-0.2·sqrt(sum(x_i^2)/D)) - exp(sum(cos(2π·x_i))/D) + 20 + e |
| `Alpine2Function` | `Alpine2`, `Alpine_2` | Scalable (D ≥ 1) | [alpine.py](../src/pyMOFL/functions/benchmark/alpine.py) | Alpine 2 function. |
| `BentCigarFunction` | `BentCigar`, `bent_cigar`, `cigar` | Scalable (D ≥ 1) | [bent_cigar.py](../src/pyMOFL/functions/benchmark/bent_cigar.py) | Bent Cigar function. |
| `BrownFunction` | `Brown`, `brown` | Scalable (D ≥ 1) | [brown.py](../src/pyMOFL/functions/benchmark/brown.py) | Brown function. |
| `ChebyshevFunction` | `Chebyshev`, `chebyshev` | D=9 | [chebyshev.py](../src/pyMOFL/functions/benchmark/chebyshev.py) | Storn's Chebyshev Polynomial Fitting benchmark function (CEC 2019 F1). |
| `ChungReynoldsFunction` | `ChungReynolds`, `chung_reynolds` | Scalable (D ≥ 1) | [chung_reynolds.py](../src/pyMOFL/functions/benchmark/chung_reynolds.py) | Chung-Reynolds function. |
| `DiscusFunction` | `Discus`, `discus` | Scalable (D ≥ 1) | [bent_cigar.py](../src/pyMOFL/functions/benchmark/bent_cigar.py) | Discus (Tablet) function. |
| `DixonPriceFunction` | `DixonPrice`, `dixon_price` | Scalable (D ≥ 1) | [dixon_price.py](../src/pyMOFL/functions/benchmark/dixon_price.py) | Dixon-Price function. |
| `ExpandedDecreasingMinimaFunction` | `DecreasingMinima`, `ExpandedDecreasingMinima`, `decreasing_minima`, `expanded_decreasing_minima` | D=1 | [niching.py](../src/pyMOFL/functions/benchmark/niching.py) | Expanded Decreasing Minima benchmark function. |
| `ExpandedEqualMinimaFunction` | `EqualMinima`, `ExpandedEqualMinima`, `equal_minima`, `expanded_equal_minima` | D=1 | [niching.py](../src/pyMOFL/functions/benchmark/niching.py) | Expanded Equal Minima benchmark function. |
| `ExpandedFiveUnevenPeakTrapFunction` | `ExpandedFiveUnevenPeakTrap`, `FiveUnevenPeakTrap`, `expanded_five_uneven_peak_trap`, `five_uneven_peak_trap` | D=1 | [niching.py](../src/pyMOFL/functions/benchmark/niching.py) | Expanded Five-Uneven-Peak Trap benchmark function. |
| `ExpandedTwoPeakTrapFunction` | `ExpandedTwoPeakTrap`, `TwoPeakTrap`, `expanded_two_peak_trap`, `two_peak_trap` | D=1 | [niching.py](../src/pyMOFL/functions/benchmark/niching.py) | Expanded Two-Peak Trap benchmark function. |
| `ExpandedUnevenMinimaFunction` | `ExpandedUnevenMinima`, `UnevenMinima`, `expanded_uneven_minima`, `uneven_minima` | D=1 | [niching.py](../src/pyMOFL/functions/benchmark/niching.py) | Expanded Uneven Minima benchmark function. |
| `GriewankFunction` | `Griewank`, `griewank` | Scalable (D ≥ 1) | [griewank.py](../src/pyMOFL/functions/benchmark/griewank.py) | Griewank function. |
| `GriewankOfRosenbrock` | `GriewankOfRosenbrock`, `griewankOfRosenbrock`, `griewank_of_rosenbrock` | Scalable (D ≥ 1) | [rosenbrock.py](../src/pyMOFL/functions/benchmark/rosenbrock.py) | Griewank of Rosenbrock (F8F2) expanded function (multimodal). |
| `HGBatFunction` | `HGBat`, `hgbat` | Scalable (D ≥ 1) | [happycat.py](../src/pyMOFL/functions/benchmark/happycat.py) | HGBat function. |
| `HappyCatFunction` | `HappyCat`, `happycat` | Scalable (D ≥ 1) | [happycat.py](../src/pyMOFL/functions/benchmark/happycat.py) | HappyCat function. |
| `HighConditionedElliptic` | `HighConditionedElliptic`, `elliptic`, `high_conditioned_elliptic` | Scalable (D ≥ 1) | [elliptic.py](../src/pyMOFL/functions/benchmark/elliptic.py) | Core Elliptic function used inside CEC-2005 F3. |
| `HilbertFunction` | `Hilbert`, `hilbert` | D=16 | [hilbert.py](../src/pyMOFL/functions/benchmark/hilbert.py) | Inverse Hilbert Matrix benchmark function (CEC 2019 F2). |
| `KatsuuraFunction` | `Katsuura`, `katsuura` | Scalable (D ≥ 1) | [katsuura.py](../src/pyMOFL/functions/benchmark/katsuura.py) | Katsuura function (CEC form with normalization). |
| `LangermannFunction` | `Langermann`, `langermann` | Scalable (D ≥ 1) | [langermann.py](../src/pyMOFL/functions/benchmark/langermann.py) | Langermann function. |
| `LennardJonesCECFunction` | `LennardJonesCEC`, `lennard_jones_cec` | D=18 | [lennard_jones.py](../src/pyMOFL/functions/benchmark/lennard_jones.py) | Lennard-Jones atomic cluster potential energy function (CEC 2019 F3 variant). |
| `LevyCEC2022Function` | `LevyCEC2022`, `levy_cec2022` | Scalable (D ≥ 1) | [levy.py](../src/pyMOFL/functions/benchmark/levy.py) | Levy function (CEC 2022+ variant). |
| `LevyCECFunction` | `LevyCEC`, `levy_cec` | Scalable (D ≥ 1) | [levy.py](../src/pyMOFL/functions/benchmark/levy.py) | Levy function (CEC 2017 variant). |
| `LevyFunction` | `Levy`, `levy` | Scalable (D ≥ 1) | [levy.py](../src/pyMOFL/functions/benchmark/levy.py) | Levy function. |
| `LunacekBiRastriginCECFunction` | `LunacekBiRastriginCEC`, `lunacek_bi_rastrigin_cec` | Scalable (D ≥ 1) | [lunacek.py](../src/pyMOFL/functions/benchmark/lunacek.py) | Lunacek Bi-Rastrigin function (CEC variant). |
| `LunacekBiRastriginFunction` | `LunacekBiRastrigin`, `lunacek_bi_rastrigin` | Scalable (D ≥ 1) | [lunacek.py](../src/pyMOFL/functions/benchmark/lunacek.py) | Lunacek Bi-Rastrigin function. |
| `LunacekRotatedCosineFunction` | `LunacekRotatedCosine`, `lunacek_rotated_cosine` | Scalable (D ≥ 1) | [lunacek.py](../src/pyMOFL/functions/benchmark/lunacek.py) | Lunacek Bi-Rastrigin with rotation applied only to the cosine term. |
| `MaxAbsolute` | `yao_liu_04` | Scalable (D ≥ 1) | [max_absolute.py](../src/pyMOFL/functions/benchmark/max_absolute.py) |  |
| `MichalewiczFunction` | `Michalewicz`, `michalewicz` | Scalable (D ≥ 1) | [michalewicz.py](../src/pyMOFL/functions/benchmark/michalewicz.py) | Michalewicz function. |
| `MultiBasinFunction` | `GNBG`, `multi_basin` | Scalable (D ≥ 1) | [multi_basin.py](../src/pyMOFL/functions/benchmark/multi_basin.py) | Generalized Multi-Basin generator (GNBG Baseline). |
| `MultiModalFunction` | `MultiModal`, `multi_modal` | Scalable (D ≥ 1) | [multi_modal_func.py](../src/pyMOFL/functions/benchmark/multi_modal_func.py) | Multi-Modal function. |
| `NeedleEyeFunction` | `NeedleEye`, `needle_eye` | Scalable (D ≥ 1) | [needle_eye.py](../src/pyMOFL/functions/benchmark/needle_eye.py) | Needle Eye function. |
| `PermFunction` | `Perm`, `perm` | Scalable (D ≥ 1) | [perm.py](../src/pyMOFL/functions/benchmark/perm.py) | Perm (0,d,β) function with d=5, β=0.5 (SPSO ID-20). |
| `PowellSumFunction` | `PowellSum`, `Powell_Sum` | Scalable (D ≥ 1) | [powell.py](../src/pyMOFL/functions/benchmark/powell.py) | Powell Sum function. |
| `QingFunction` | `Qing`, `qing` | Scalable (D ≥ 1) | [qing.py](../src/pyMOFL/functions/benchmark/qing.py) | Qing function. |
| `QuarticFunction` | `Quartic`, `quartic` | Scalable (D ≥ 1) | [quartic.py](../src/pyMOFL/functions/benchmark/quartic.py) | Quartic (De Jong 4) function, without noise. |
| `RastriginFunction` | `Rastrigin`, `rastrigin` | Scalable (D ≥ 1) | [rastrigin.py](../src/pyMOFL/functions/benchmark/rastrigin.py) | Rastrigin function: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i)) |
| `RosenbrockFunction` | `Rosenbrock`, `rosenbrock` | Scalable (D ≥ 1) | [rosenbrock.py](../src/pyMOFL/functions/benchmark/rosenbrock.py) | Rosenbrock function (unimodal). |
| `SalomonFunction` | `Salomon`, `salomon` | Scalable (D ≥ 1) | [salomon.py](../src/pyMOFL/functions/benchmark/salomon.py) | Salomon function. |
| `Schaffer_F6` | `Schaffer_F6` | Scalable (D ≥ 1) | [schaffer.py](../src/pyMOFL/functions/benchmark/schaffer.py) | Schaffer's F6 function. |
| `Schaffer_F6_Expanded` | `Schaffer_F6_Expanded`, `schaffer_f6_expanded` | Scalable (D ≥ 1) | [schaffer.py](../src/pyMOFL/functions/benchmark/schaffer.py) | Expanded Schaffer F6 function. |
| `SchaffersF7CECFunction` | `SchaffersF7CEC`, `schaffer_f7`, `schaffers_f7_cec` | Scalable (D ≥ 1) | [schaffer.py](../src/pyMOFL/functions/benchmark/schaffer.py) | Schaffers F7 function (CEC variant). |
| `SchaffersF7Function` | `SchaffersF7`, `schaffers_f7` | Scalable (D ≥ 1) | [schaffer.py](../src/pyMOFL/functions/benchmark/schaffer.py) | Schaffers F7 function. |
| `SphereFunction` | `Sphere`, `sphere` | Scalable (D ≥ 1) | [sphere.py](../src/pyMOFL/functions/benchmark/sphere.py) | Sphere function. |
| `StepFunction` | `Step`, `step` | Scalable (D ≥ 1) | [step.py](../src/pyMOFL/functions/benchmark/step.py) | Step function (De Jong's Step function). |
| `StyblinskiTangFunction` | `StyblinskiTang`, `styblinski_tang` | Scalable (D ≥ 1) | [styblinski_tang.py](../src/pyMOFL/functions/benchmark/styblinski_tang.py) | Styblinski-Tang function. |
| `SumDifferentPowersFunction` | `SumDifferentPowers`, `sum_different_powers` | Scalable (D ≥ 1) | [sum_different_powers.py](../src/pyMOFL/functions/benchmark/sum_different_powers.py) | Sum of Different Powers function. |
| `WeierstrassFunction` | `Weierstrass`, `weierstrass` | Scalable (D ≥ 1) | [weierstrass.py](../src/pyMOFL/functions/benchmark/weierstrass.py) | Weierstrass function: |
| `ZakharovFunction` | `Zakharov`, `zakharov` | Scalable (D ≥ 1) | [zakharov.py](../src/pyMOFL/functions/benchmark/zakharov.py) | Zakharov function. |
| `ZeroSumFunction` | `ZeroSum`, `zero_sum` | Scalable (D ≥ 1) | [zero_sum.py](../src/pyMOFL/functions/benchmark/zero_sum.py) | Zero Sum function. |

## BBOB Primitives

| Function Class | Registry Aliases | Dimension | Source File | Description |
|:---|:---|:---:|:---|:---|
| `AttractiveSectorFunction` | `AttractiveSector`, `attractive_sector` | Scalable (D ≥ 1) | [attractive_sector.py](../src/pyMOFL/functions/benchmark/attractive_sector.py) | Attractive Sector function. |
| `BucheRastriginFunction` | `BucheRastrigin`, `buche_rastrigin` | Scalable (D ≥ 1) | [buche_rastrigin.py](../src/pyMOFL/functions/benchmark/buche_rastrigin.py) | Büche-Rastrigin function. |
| `DifferentPowersFunction` | `DifferentPowers`, `different_powers` | Scalable (D ≥ 1) | [different_powers.py](../src/pyMOFL/functions/benchmark/different_powers.py) | Different Powers function (CEC form with outer sqrt). |
| `GallagherPeaksFunction` | `GallagherPeaks`, `gallagher_peaks` | Scalable (D ≥ 1) | [gallagher_peaks.py](../src/pyMOFL/functions/benchmark/gallagher_peaks.py) | Gallagher's Gaussian Peaks function. |
| `LinearSlopeFunction` | `LinearSlope`, `linear_slope` | Scalable (D ≥ 1) | [linear_slope.py](../src/pyMOFL/functions/benchmark/linear_slope.py) | Linear Slope function. |
| `SharpRidgeFunction` | `SharpRidge`, `sharp_ridge` | Scalable (D ≥ 1) | [sharp_ridge.py](../src/pyMOFL/functions/benchmark/sharp_ridge.py) | Sharp Ridge function. |
| `StepEllipsoidFunction` | `StepEllipsoid`, `step_ellipsoid` | Scalable (D ≥ 1) | [step_ellipsoid.py](../src/pyMOFL/functions/benchmark/step_ellipsoid.py) | Step Ellipsoidal function. |

## Fixed 2D Functions

| Function Class | Registry Aliases | Dimension | Source File | Description |
|:---|:---|:---:|:---|:---|
| `Ackley2Function` | `Ackley2`, `ackley2`, `ackley_2` | D=2 | [ackley.py](../src/pyMOFL/functions/benchmark/ackley.py) | Ackley 2 function: f(x, y) = -200 * exp(-0.02 * sqrt(x_1^2 + x_2^2)) |
| `Ackley3Function` | `Ackley3`, `ackley3`, `ackley_3` | D=2 | [ackley.py](../src/pyMOFL/functions/benchmark/ackley.py) | Ackley 3 function: f(x,y) = -200*exp(-0.02*sqrt(x^2+y^2)) + 5*exp(cos(3x)+sin(3y)) |
| `AdjimanFunction` | `adjiman` | D=2 | [adjiman.py](../src/pyMOFL/functions/benchmark/adjiman.py) | Adjiman function (2D). |
| `Alpine1Function` | `Alpine1`, `Alpine_1` | D=2 | [alpine.py](../src/pyMOFL/functions/benchmark/alpine.py) | Alpine 1 function. |
| `BartelsConnFunction` | `bartels_conn` | D=2 | [bartels_conn.py](../src/pyMOFL/functions/benchmark/bartels_conn.py) | Bartels Conn function (2D). |
| `BealeFunction` | `Beale`, `beale` | D=2 | [beale.py](../src/pyMOFL/functions/benchmark/beale.py) | Beale function (2D). |
| `BiggsExp02Function` | `BiggsExp02`, `biggs_exp02` | D=2 | [biggs_exp.py](../src/pyMOFL/functions/benchmark/biggs_exp.py) | Biggs EXP02 function. |
| `BirdFunction` | `bird` | D=2 | [bird.py](../src/pyMOFL/functions/benchmark/bird.py) | Bird function (2D). |
| `Bohachevsky1Function` | `Bohachevsky1`, `bohachevsky1` | D=2 | [bohachevsky.py](../src/pyMOFL/functions/benchmark/bohachevsky.py) | Bohachevsky function variant 1 (2D). |
| `Bohachevsky2Function` | `Bohachevsky2`, `bohachevsky2` | D=2 | [bohachevsky.py](../src/pyMOFL/functions/benchmark/bohachevsky.py) | Bohachevsky function variant 2 (2D). |
| `Bohachevsky3Function` | `Bohachevsky3`, `bohachevsky3` | D=2 | [bohachevsky.py](../src/pyMOFL/functions/benchmark/bohachevsky.py) | Bohachevsky function variant 3 (2D). |
| `BoothFunction` | `Booth`, `booth` | D=2 | [booth.py](../src/pyMOFL/functions/benchmark/booth.py) | Booth function (2D). |
| `Branin2Function` | `Branin_2`, `Branin_RCOS_2` | D=2 | [branin.py](../src/pyMOFL/functions/benchmark/branin.py) | Branin RCOS 2 function. |
| `BraninFunction` | `Branin`, `Branin_RCOS` | D=2 | [branin.py](../src/pyMOFL/functions/benchmark/branin.py) | Branin RCOS function. |
| `BrentFunction` | `brent` | D=2 | [brent.py](../src/pyMOFL/functions/benchmark/brent.py) | Brent function (2D). |
| `Bukin6Function` | `Bukin6`, `bukin6` | D=2 | [bukin.py](../src/pyMOFL/functions/benchmark/bukin.py) | Bukin N.6 function (2D). |
| `ChichinadzeFunction` | `chichinadze` | D=2 | [chichinadze.py](../src/pyMOFL/functions/benchmark/chichinadze.py) | Chichinadze function (2D). |
| `CosineMixtureFunction` | `CosineMixture`, `cosine_mixture` | D=2 | [cosine_mixture.py](../src/pyMOFL/functions/benchmark/cosine_mixture.py) | Cosine Mixture function. |
| `CrossInTrayFunction` | `CrossInTray`, `cross_in_tray` | D=2 | [cross_in_tray.py](../src/pyMOFL/functions/benchmark/cross_in_tray.py) | Cross-in-Tray function (2D). |
| `CrossLegTableFunction` | `cross_leg_table` | D=2 | [cross_leg_table.py](../src/pyMOFL/functions/benchmark/cross_leg_table.py) | Cross-Leg Table function (2D). |
| `CrownedCrossFunction` | `crowned_cross` | D=2 | [crowned_cross.py](../src/pyMOFL/functions/benchmark/crowned_cross.py) | Crowned Cross function (2D). |
| `CsendesFunction` | `Csendes`, `csendes`, `infinity` | D=2 | [csendes.py](../src/pyMOFL/functions/benchmark/csendes.py) | Csendes (Infinity) function. |
| `DamavandiFunction` | `damavandi` | D=2 | [damavandi.py](../src/pyMOFL/functions/benchmark/damavandi.py) | Damavandi function (2D). |
| `Deb01Function` | `Deb01`, `deb01` | D=2 | [deb01.py](../src/pyMOFL/functions/benchmark/deb01.py) | Deb's First function (Deb01). |
| `Deb03Function` | `Deb03`, `deb03` | D=2 | [deb03.py](../src/pyMOFL/functions/benchmark/deb03.py) | Deb's Third function (Deb03). |
| `DecanomialFunction` | `Decanomial`, `decanomial` | D=2 | [decanomial.py](../src/pyMOFL/functions/benchmark/decanomial.py) | Decanomial function. |
| `DeceptiveFunction` | `Deceptive`, `deceptive` | D=2 | [deceptive.py](../src/pyMOFL/functions/benchmark/deceptive.py) | Deceptive function. |
| `DeckkersAartsFunction` | `deckkers_aarts` | D=2 | [deckkers_aarts.py](../src/pyMOFL/functions/benchmark/deckkers_aarts.py) | Deckkers-Aarts function (2D). |
| `DeflectedCorrugatedSpringFunction` | `DeflectedCorrugatedSpring`, `deflected_corrugated_spring` | D=2 | [deflected_corrugated_spring.py](../src/pyMOFL/functions/benchmark/deflected_corrugated_spring.py) | Deflected Corrugated Spring function. |
| `DropWaveFunction` | `DropWave`, `drop_wave` | D=2 | [drop_wave.py](../src/pyMOFL/functions/benchmark/drop_wave.py) | Drop-Wave function (2D). |
| `EasomFunction` | `Easom` | D=2 | [easom.py](../src/pyMOFL/functions/benchmark/easom.py) | Easom function. |
| `EggCrateFunction` | `egg_crate` | D=2 | [egg_crate.py](../src/pyMOFL/functions/benchmark/egg_crate.py) | Egg Crate function (2D). |
| `EggholderFunction` | `Eggholder`, `eggholder` | D=2 | [eggholder.py](../src/pyMOFL/functions/benchmark/eggholder.py) | Eggholder function (2D). |
| `ElAttarVidyasagarDuttaFunction` | `el_attar` | D=2 | [el_attar.py](../src/pyMOFL/functions/benchmark/el_attar.py) | El-Attar-Vidyasagar-Dutta function (2D). |
| `Exp2Function` | `exp2` | D=2 | [exp2.py](../src/pyMOFL/functions/benchmark/exp2.py) | Exp2 function (2D). |
| `ExpandedHimmelblauFunction` | `ExpandedHimmelblau`, `expanded_himmelblau` | D=2 | [niching.py](../src/pyMOFL/functions/benchmark/niching.py) | Expanded Himmelblau multimodal benchmark function. |
| `ExpandedSixHumpCamelFunction` | `ExpandedSixHumpCamel`, `expanded_six_hump_camel` | D=2 | [niching.py](../src/pyMOFL/functions/benchmark/niching.py) | Expanded Six-Hump Camel benchmark function. |
| `ExponentialFunction` | `Exponential`, `exponential` | D=2 | [exponential_function.py](../src/pyMOFL/functions/benchmark/exponential_function.py) | Exponential function. |
| `FreudensteinRothFunction` | `freudenstein_roth` | D=2 | [freudenstein_roth.py](../src/pyMOFL/functions/benchmark/freudenstein_roth.py) | Freudenstein Roth function (2D). |
| `GiuntaFunction` | `giunta` | D=2 | [giunta.py](../src/pyMOFL/functions/benchmark/giunta.py) | Giunta function (2D). |
| `GoldsteinPriceFunction` | `GoldsteinPrice`, `Goldstein_Price` | D=2 | [goldstein_price.py](../src/pyMOFL/functions/benchmark/goldstein_price.py) | Goldstein-Price function. |
| `HansenFunction` | `hansen` | D=2 | [hansen.py](../src/pyMOFL/functions/benchmark/hansen.py) | Hansen function (2D). |
| `HimmelblauFunction` | `Himmelblau` | D=2 | [himmelblau.py](../src/pyMOFL/functions/benchmark/himmelblau.py) | Himmelblau function. |
| `HolderTableFunction` | `HolderTable`, `holder_table` | D=2 | [holder_table.py](../src/pyMOFL/functions/benchmark/holder_table.py) | Holder Table function (2D). |
| `HosakiFunction` | `hosaki` | D=2 | [hosaki.py](../src/pyMOFL/functions/benchmark/hosaki.py) | Hosaki function (2D). |
| `JennrichSampsonFunction` | `jennrich_sampson` | D=2 | [jennrich_sampson.py](../src/pyMOFL/functions/benchmark/jennrich_sampson.py) | Jennrich-Sampson function (2D). |
| `KeaneFunction` | `Keane`, `keane` | D=2 | [keane.py](../src/pyMOFL/functions/benchmark/keane.py) | Keane function. |
| `LeonFunction` | `leon` | D=2 | [leon.py](../src/pyMOFL/functions/benchmark/leon.py) | Leon function (2D). |
| `MatyasFunction` | `Matyas` | D=2 | [matyas.py](../src/pyMOFL/functions/benchmark/matyas.py) | Matyas function. |
| `McCormickFunction` | `McCormick` | D=2 | [mccormick.py](../src/pyMOFL/functions/benchmark/mccormick.py) | McCormick function. |
| `ModifiedVincentFunction` | `ModifiedVincent`, `modified_vincent` | D=2 | [niching.py](../src/pyMOFL/functions/benchmark/niching.py) | Modified Vincent multimodal benchmark function. |
| `NewFunction01Function` | `NewFunction01`, `new_function01` | D=2 | [new_function.py](../src/pyMOFL/functions/benchmark/new_function.py) | New Function 01. |
| `NewFunction02Function` | `NewFunction02`, `new_function02` | D=2 | [new_function.py](../src/pyMOFL/functions/benchmark/new_function.py) | New Function 02. |
| `OddSquareFunction` | `OddSquare`, `odd_square` | D=2 | [odd_square.py](../src/pyMOFL/functions/benchmark/odd_square.py) | Odd Square function. |
| `ParsopoulosFunction` | `parsopoulos` | D=2 | [parsopoulos.py](../src/pyMOFL/functions/benchmark/parsopoulos.py) | Parsopoulos function (2D). |
| `QuinticFunction` | `Quintic`, `quintic` | D=2 | [quintic.py](../src/pyMOFL/functions/benchmark/quintic.py) | Quintic function. |
| `RanaFunction` | `Rana`, `rana` | D=2 | [rana.py](../src/pyMOFL/functions/benchmark/rana.py) | Rana function. |
| `Schaffer1Function` | `Schaffer1`, `Schaffer_1` | D=2 | [schaffer.py](../src/pyMOFL/functions/benchmark/schaffer.py) | Schaffer N.1 function — concentric ring landscape. |
| `Schaffer2Function` | `Schaffer2`, `Schaffer_2` | D=2 | [schaffer.py](../src/pyMOFL/functions/benchmark/schaffer.py) | Schaffer N.2 function — deceptive oscillation variant. |
| `Schaffer4Function` | `Schaffer4`, `Schaffer_4` | D=2 | [schaffer.py](../src/pyMOFL/functions/benchmark/schaffer.py) | Schaffer N.4 function — cos-of-sin variant, hardest in the family. |
| `SixHumpCamelFunction` | `SixHumpCamel`, `six_hump_camel` | D=2 | [camel.py](../src/pyMOFL/functions/benchmark/camel.py) | Six-Hump Camel function (2D). |
| `TestTubeHolderFunction` | `test_tube_holder` | D=2 | [test_tube_holder.py](../src/pyMOFL/functions/benchmark/test_tube_holder.py) |  |
| `ThreeHumpCamelFunction` | `ThreeHumpCamel`, `three_hump_camel` | D=2 | [camel.py](../src/pyMOFL/functions/benchmark/camel.py) | Three-Hump Camel function (2D). |
| `Ursem01Function` | `ursem01` | D=2 | [ursem01.py](../src/pyMOFL/functions/benchmark/ursem01.py) | Ursem01 function (2D). |
| `VenterSobieskiFunction` | `venter_sobieski` | D=2 | [venter_sobieski.py](../src/pyMOFL/functions/benchmark/venter_sobieski.py) | Venter Sobiezcczanski-Sobieski function (2D). |
| `XinSheYang01Function` | `XinSheYang01`, `xin_she_yang01` | D=2 | [xin_she_yang01.py](../src/pyMOFL/functions/benchmark/xin_she_yang01.py) | Xin-She Yang's First function. |
| `ZettlFunction` | `zettl` | D=2 | [zettl.py](../src/pyMOFL/functions/benchmark/zettl.py) | Zettl function (2D). |
| `ZimmermanFunction` | `Zimmerman`, `zimmerman` | D=2 | [zimmerman.py](../src/pyMOFL/functions/benchmark/zimmerman.py) | Zimmerman function (2D). |
| `ZirilliFunction` | `Zirilli`, `zirilli` | D=2 | [zirilli.py](../src/pyMOFL/functions/benchmark/zirilli.py) | Zirilli (Aluffi-Pentini) function (2D). |

## Fixed Dimension (3D-6D) Functions

| Function Class | Registry Aliases | Dimension | Source File | Description |
|:---|:---|:---:|:---|:---|
| `BiggsExp03Function` | `BiggsExp03`, `biggs_exp03` | D=3 | [biggs_exp.py](../src/pyMOFL/functions/benchmark/biggs_exp.py) | Biggs EXP03 function. |
| `BiggsExp04Function` | `BiggsExp04`, `biggs_exp04` | D=4 | [biggs_exp.py](../src/pyMOFL/functions/benchmark/biggs_exp.py) | Biggs EXP04 function. |
| `BiggsExp05Function` | `BiggsExp05`, `biggs_exp05` | D=5 | [biggs_exp.py](../src/pyMOFL/functions/benchmark/biggs_exp.py) | Biggs EXP05 function. |
| `BoxBettsFunction` | `BoxBetts`, `box_betts` | D=3 | [box_betts.py](../src/pyMOFL/functions/benchmark/box_betts.py) | Box-Betts exponential quadratic sum function (3D). |
| `ColvilleFunction` | `Colville`, `colville` | D=4 | [colville.py](../src/pyMOFL/functions/benchmark/colville.py) | Colville function (4D). |
| `CoranaFunction` | `Corana`, `corana` | D=4 | [corana.py](../src/pyMOFL/functions/benchmark/corana.py) | Corana function (4D). |
| `DeVilliersGlasser01Function` | `DeVilliersGlasser01`, `devillers_glasser01` | D=4 | [devillers_glasser.py](../src/pyMOFL/functions/benchmark/devillers_glasser.py) | De Villiers-Glasser 01 function (4D). |
| `DeVilliersGlasser02Function` | `DeVilliersGlasser02`, `devillers_glasser02` | D=5 | [devillers_glasser.py](../src/pyMOFL/functions/benchmark/devillers_glasser.py) | De Villiers-Glasser 02 function (5D). |
| `DolanFunction` | `Dolan`, `dolan` | D=5 | [dolan.py](../src/pyMOFL/functions/benchmark/dolan.py) | Dolan function (5D). |
| `GulfFunction` | `Gulf`, `gulf` | D=3 | [gulf.py](../src/pyMOFL/functions/benchmark/gulf.py) | Gulf research function (3D). |
| `Hartmann3Function` | `Hartmann3`, `hartmann3` | D=3 | [hartmann.py](../src/pyMOFL/functions/benchmark/hartmann.py) | Hartmann 3D function. |
| `Hartmann6Function` | `Hartmann6`, `hartmann6` | D=6 | [hartmann.py](../src/pyMOFL/functions/benchmark/hartmann.py) | Hartmann 6D function. |
| `HelicalValleyFunction` | `HelicalValley`, `helical_valley` | D=3 | [helical_valley.py](../src/pyMOFL/functions/benchmark/helical_valley.py) | Helical Valley (Fletcher-Powell) function (3D). |
| `KowalikFunction` | `Kowalik`, `kowalik` | D=4 | [kowalik.py](../src/pyMOFL/functions/benchmark/kowalik.py) | Kowalik function (4D). |
| `MieleCantrellFunction` | `MieleCantrell`, `miele_cantrell` | D=4 | [miele_cantrell.py](../src/pyMOFL/functions/benchmark/miele_cantrell.py) | Miele-Cantrell function (4D). |
| `PowellSingular2Function` | `PowellSingular2`, `Powell_Singular_2` | D=4 | [powell.py](../src/pyMOFL/functions/benchmark/powell.py) | Powell Singular 2 function. |
| `PowellSingularFunction` | `PowellSingular`, `Powell_Singular` | D=4 | [powell.py](../src/pyMOFL/functions/benchmark/powell.py) | Powell Singular function. |

## Mishra Family

| Function Class | Registry Aliases | Dimension | Source File | Description |
|:---|:---|:---:|:---|:---|
| `Mishra01Function` | `Mishra01`, `mishra01` | Scalable (D ≥ 1) | [mishra.py](../src/pyMOFL/functions/benchmark/mishra.py) | Mishra 01 function (scalable). |
| `Mishra02Function` | `Mishra02`, `mishra02` | Scalable (D ≥ 1) | [mishra.py](../src/pyMOFL/functions/benchmark/mishra.py) | Mishra 02 function (scalable). |
| `Mishra03Function` | `Mishra03`, `mishra03` | D=2 | [mishra.py](../src/pyMOFL/functions/benchmark/mishra.py) | Mishra 03 function (2D fixed). |
| `Mishra04Function` | `Mishra04`, `mishra04` | D=2 | [mishra.py](../src/pyMOFL/functions/benchmark/mishra.py) | Mishra 04 function (2D fixed). |
| `Mishra05Function` | `Mishra05`, `mishra05` | D=2 | [mishra.py](../src/pyMOFL/functions/benchmark/mishra.py) | Mishra 05 function (2D fixed). |
| `Mishra06Function` | `Mishra06`, `mishra06` | D=2 | [mishra.py](../src/pyMOFL/functions/benchmark/mishra.py) | Mishra 06 function (2D fixed). |
| `Mishra07Function` | `Mishra07`, `mishra07` | Scalable (D ≥ 1) | [mishra.py](../src/pyMOFL/functions/benchmark/mishra.py) | Mishra 07 function (scalable). |
| `Mishra08Function` | `Mishra08`, `mishra08` | D=2 | [mishra.py](../src/pyMOFL/functions/benchmark/mishra.py) | Mishra 08 function (2D fixed), also known as Mishra-Decanomial. |
| `Mishra09Function` | `Mishra09`, `mishra09` | D=3 | [mishra.py](../src/pyMOFL/functions/benchmark/mishra.py) | Mishra 09 function (3D fixed). |
| `Mishra10Function` | `Mishra10`, `mishra10` | D=2 | [mishra.py](../src/pyMOFL/functions/benchmark/mishra.py) | Mishra 10 function (2D fixed). |
| `Mishra11Function` | `Mishra11`, `mishra11` | Scalable (D ≥ 1) | [mishra.py](../src/pyMOFL/functions/benchmark/mishra.py) | Mishra 11 function (scalable). |

## Schwefel Family

| Function Class | Registry Aliases | Dimension | Source File | Description |
|:---|:---|:---:|:---|:---|
| `SchwefelFunction` | `Schwefel`, `schwefel` | Scalable (D ≥ 1) | [schwefel.py](../src/pyMOFL/functions/benchmark/schwefel.py) | Schwefel function (offset form of Problem 2.26). |
| `SchwefelSinFunction` | `SchwefelSin`, `schwefel_sin` | Scalable (D ≥ 1) | [schwefel_sin.py](../src/pyMOFL/functions/benchmark/schwefel_sin.py) | Schwefel x*sin(sqrt(\|x\|)) function. |
| `Schwefel_1_2` | `Schwefel_1_2` | Scalable (D ≥ 1) | [schwefel.py](../src/pyMOFL/functions/benchmark/schwefel.py) | Schwefel 1.2 function. |
| `Schwefel_2_13` | `Schwefel_2_13` | Scalable (D ≥ 1) | [schwefel.py](../src/pyMOFL/functions/benchmark/schwefel.py) | Schwefel's Problem 2.13 function: f(x) = sum((A_i - B_i(x))^2) |
| `Schwefel_2_20` | `Schwefel_2_20`, `schwefel_2_20` | Scalable (D ≥ 1) | [schwefel.py](../src/pyMOFL/functions/benchmark/schwefel.py) | Schwefel 2.20 function: f(x) = Σ \|x_i\| |
| `Schwefel_2_21` | `Schwefel_2_21`, `schwefel_2_21` | Scalable (D ≥ 1) | [schwefel.py](../src/pyMOFL/functions/benchmark/schwefel.py) | Schwefel 2.21 function: f(x) = max_i \|x_i\| |
| `Schwefel_2_22` | `Schwefel_2_22`, `schwefel_2_22` | Scalable (D ≥ 1) | [schwefel.py](../src/pyMOFL/functions/benchmark/schwefel.py) | Schwefel 2.22 function: f(x) = Σ \|x_i\| + Π \|x_i\| |
| `Schwefel_2_23` | `Schwefel_2_23`, `schwefel_2_23` | Scalable (D ≥ 1) | [schwefel.py](../src/pyMOFL/functions/benchmark/schwefel.py) | Schwefel 2.23 function: f(x) = Σ x_i^10 |
| `Schwefel_2_25` | `Schwefel_2_25`, `schwefel_2_25` | Scalable (D ≥ 1) | [schwefel.py](../src/pyMOFL/functions/benchmark/schwefel.py) | Schwefel 2.25 function: f(x) = Σ_{i=1}^{D-1} (x_i² - x_{i+1})² + (x_i - 1)² |
| `Schwefel_2_26` | `Schwefel_2_26`, `schwefel_2_26` | Scalable (D ≥ 1) | [schwefel.py](../src/pyMOFL/functions/benchmark/schwefel.py) | Schwefel 2.26 function: f(x) = -Σ x_i sin(sqrt(\|x_i\|)) |
| `Schwefel_2_36` | `Schwefel_2_36`, `schwefel_2_36` | Scalable (D ≥ 1) | [schwefel.py](../src/pyMOFL/functions/benchmark/schwefel.py) | Schwefel 2.36 function (sine-root sum on non-negative domain). |
| `Schwefel_2_4` | `Schwefel_2_4`, `schwefel_2_4` | Scalable (D ≥ 1) | [schwefel.py](../src/pyMOFL/functions/benchmark/schwefel.py) | Schwefel 2.4 function (Extended Rosenbrock with star dependency on x_1). |
| `Schwefel_2_6` | `Schwefel_2_6` | Scalable (D ≥ 1) | [schwefel.py](../src/pyMOFL/functions/benchmark/schwefel.py) | Schwefel 2.6 function. |

## Engineering & Special Benchmarks

| Function Class | Registry Aliases | Dimension | Source File | Description |
|:---|:---|:---:|:---|:---|
| `ColaFunction` | `Cola`, `cola` | D=17 | [cola.py](../src/pyMOFL/functions/benchmark/cola.py) | Cola function. |
| `CompressionSpringFunction` | `CompressionSpring`, `compression_spring` | Scalable (D ≥ 1) | [compression_spring.py](../src/pyMOFL/functions/benchmark/compression_spring.py) | Compression Spring Function (SPSO ID-21). |
| `GearTrainFunction` | `GearTrain`, `gear_train` | Scalable (D ≥ 1) | [gear_train.py](../src/pyMOFL/functions/benchmark/gear_train.py) | Gear Train function (SPSO ID-18). |
| `LennardJonesFunction` | `LennardJones`, `lennard_jones` | Scalable (D ≥ 1) | [lennard_jones.py](../src/pyMOFL/functions/benchmark/lennard_jones.py) | Lennard-Jones n-atom cluster potential energy function (SPSO ID-17). |
| `NetworkFunction` | `Network`, `network` | Scalable (D ≥ 1) | [network.py](../src/pyMOFL/functions/benchmark/network.py) | Network design benchmark function (SPSO ID 11). |
| `TripodFunction` | `Tripod`, `tripod` | Scalable (D ≥ 1) | [tripod.py](../src/pyMOFL/functions/benchmark/tripod.py) | Tripod benchmark function. |
