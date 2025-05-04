# SPSO 2007 & SPSO 2011 - **Benchmark-Function Handbook**

> Standard Particle Swarm Optimisation (SPSO) is released with two reference
> benchmark suites—**SPSO 2007** and **SPSO 2011**—so that new PSO variants
> can be compared on a *fixed*, *traceable* set of problems.
> The tables below collect every function in those suites, give its basic
> properties, and point you to the *primary* literature where the analytic
> form, bounds and optima are defined.

---

## 1 Standalone / Engineering Test Problems

| SPSO ID | Function                   | Dimension   | Category                  | Short description                                                            | Canonical source                                                                                                            |
| ------- | -------------------------- | ----------- | ------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 4       | **Tripod**                 | 2           | multimodal, non-separable | Piece-wise quadratic surface with three “legs”.                              | Molga & Smutnicki, *Test Functions for Optimization Needs* (2005) ([robertmarks.org][1])                                    |
| 11      | **Network**                | 42 (mixed)  | hybrid, partly binary     | Cost minimisation of a simplified tele-com network; 35 real + 7 binary vars. | Clerc, *SPSO Benchmark Doc* (2012 tech. note) and Zambrano-Bigiarini et al., *CEC-2013 baseline* (2013) ([ResearchGate][2]) |
| 15      | **Step** (biased)          | 10          | discontinuous             | De Jong’s discontinuous step surface, additional bias term per SPSO spec.    | De Jong, PhD thesis (1975) ([CiteSeerX][3])                                                                                 |
| 17      | **Lennard-Jones (6-atom)** | 18          | physics, non-convex       | 12-6 potential energy of a six-atom cluster; many local minima.              | Lennard-Jones (1924) ([Royal Society Publishing][4])                                                                        |
| 18      | **Gear Train**             | 4 (integer) | engineering, discrete     | Select numbers of teeth to approximate a target ratio 1 ∶ 6.931.             | Sandgren, *J. Mech. Des.* 112 (2):223–229 (1990) ([ASME Digital Collection][5])                                             |
| 20      | **Perm (0,d,β)**           | 5 (integer) | multimodal                | Bowl-shaped *Perm* function with β = 0.5.                                    | Surjanovic & Bingham, *VLSE Library* (2013) ([Simon Fraser University][6])                                                  |
| 21      | **Compression Spring**     | 3 (mixed)   | engineering, constrained  | Minimise spring weight subject to stress & deflection limits.                | Deb, *Efficient Constraint Handling for GAs* (2000) ([ScienceDirect][7])                                                    |

---

## 2 Shifted Classical Functions (all inherited from **CEC 2005**)

The SPSO authors adopted six shifted problems from the CEC 2005 real-parameter
suite, keeping the original shift vectors and biases but *fixing the
dimensions* as shown below.

| SPSO ID | Base         | Dim. | CEC-2005 ID | Notes                              |
| ------- | ------------ | ---- | ----------- | ---------------------------------- |
| 100     | Sphere       | 30   | F1          | global minimum at -450 after shift |
| 102     | Rosenbrock   | 10   | F6          | narrow curved valley, shifted      |
| 103     | Rastrigin    | 30   | F9          | highly multimodal, shifted         |
| 104     | Schwefel-2.6 | 10   | F5          | deceptive global min on boundary   |
| 105     | Griewank     | 10   | F7          | shifted (rotation omitted)         |
| 106     | Ackley       | 30   | F8          | shifted (rotation omitted)         |

All six are defined in the CEC technical report by Liang et al. (2005)
([ResearchGate][8]) and, for the Rastrigin original, Rastrigin (1974) ([SCIRP][9]).

---

## 3 Relationship between the 2007 & 2011 suites

* **SPSO 2007** uses *all* functions in Tables 1 and 2.
* **SPSO 2011** **drops** the low-dimensional Tripod (ID 4) and adds adaptive,
  rotational-invariant sampling, but the *function* list is otherwise
  unchanged. Benchmarks for SPSO 2011 are documented in the Zambrano-Bigiarini
  **CEC-2013 baseline** paper ([ResearchGate][2]).

---

## 4 Implementing in **pyMOFL**

```python
from pymofl.functions.unimodal   import SphereFunction
from pymofl.decorators           import ShiftedFunction
from pymofl.functions.engineering import GearTrainFunction

# SPSO-100 : shifted 30-D Sphere
shift_vec  = load_cec_shift('F1_30D.dat')
sphere100  = ShiftedFunction(SphereFunction(dimension=30), shift_vec)

# SPSO-18 : integer Gear-train design
gear18 = GearTrainFunction()

# evaluate
print(sphere100.evaluate(np.zeros(30)))
print(gear18.evaluate([12, 28, 16, 18]))
```

*Every SPSO member can be reproduced via either*
(1) a dedicated `*Function` class (engineering cases) *or*
(2) **base + decorator** (`ShiftedFunction`, `RotatedFunction`, …) for the CEC derivatives.

---

## 5 Reference List

.. \[1] Molga M., Smutnicki C. (2005). *Test Functions for Optimization Needs*. ([robertmarks.org][1])
.. \[2] Clerc M. (2012). *Standard PSO 2007/2011 Benchmark Documentation*.
See also Zambrano-Bigiarini M. et al. (2013). *Standard PSO-2011 at CEC-2013*. ([ResearchGate][2])
.. \[3] De Jong K.A. (1975). *An Analysis of the Behavior of a Class of Genetic Adaptive Systems*. PhD thesis. ([CiteSeerX][3])
.. \[4] Lennard-Jones J.E. (1924). “On the Determination of Molecular Fields II.” *Proc. R. Soc. A* 106, 463-477. ([Royal Society Publishing][4])
.. \[5] Sandgren E. (1990). “Nonlinear Integer and Discrete Programming in Mechanical Design Optimization.” *J. Mech. Des.* 112(2), 223-229. ([ASME Digital Collection][5])
.. \[6] Surjanovic S., Bingham D. (2013). *Virtual Library of Simulation Experiments: Test Functions and Datasets*. ([Simon Fraser University][6])
.. \[7] Deb K. (2000). “An Efficient Constraint Handling Method for Genetic Algorithms.” *Comput. Methods Appl. Mech. Eng.* 186, 311-338. ([ScienceDirect][7])
.. \[8] Liang J.J. et al. (2005). *Problem Definitions and Evaluation Criteria for the CEC-2005 Special Session on Real-Parameter Optimization*. ([ResearchGate][8])
.. \[9] Rastrigin L.A. (1974). *Systems of Extremal Control*. Mir, Moscow. ([SCIRP][9])

---

**File placement:** save this file as `docs/suites/sPSO_benchmarks.md` so that
developers can open it alongside the codebase and immediately see (i) which
functions to call, (ii) how to reproduce SPSO figures, and (iii) where every
formula came from.

[1]: https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf?utm_source=chatgpt.com "[PDF] Test functions for optimization needs - Robert Marks.org"
[2]: https://www.researchgate.net/publication/255756848_Standard_Particle_Swarm_Optimisation_2011_at_CEC-2013_A_baseline_for_future_PSO_improvements "(PDF) Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future PSO improvements"
[3]: https://citeseerx.ist.psu.edu/document?doi=7b2ea6ffdb72c9c0d30389c8e8d720c6e9041b6c&repid=rep1&type=pdf&utm_source=chatgpt.com "De Jong, K. A. (1975). An analysis of the behavior of a ... - CiteSeerX"
[4]: https://royalsocietypublishing.org/doi/10.1098/rspa.1924.0082?utm_source=chatgpt.com "On the determination of molecular fields. —II. From the equation of ..."
[5]: https://asmedigitalcollection.asme.org/mechanicaldesign/issue/112/2?utm_source=chatgpt.com "Volume 112 Issue 2 | J. Mech. Des. - ASME Digital Collection"
[6]: https://www.sfu.ca/~ssurjano/permdb.html?utm_source=chatgpt.com "Perm Function d, beta"
[7]: https://www.sciencedirect.com/science/article/abs/pii/S0045782599003898?utm_source=chatgpt.com "An efficient constraint handling method for genetic algorithms"
[8]: https://www.researchgate.net/publication/235710019_Problem_Definitions_and_Evaluation_Criteria_for_the_CEC_2005_Special_Session_on_Real-Parameter_Optimization?utm_source=chatgpt.com "(PDF) Problem Definitions and Evaluation Criteria for the CEC 2005 ..."
[9]: https://www.scirp.org/reference/referencespapers?referenceid=610558&utm_source=chatgpt.com "L. A. Rastrigin, “Systems of Extreme Control,” Nauka, Moscow, 1974."
