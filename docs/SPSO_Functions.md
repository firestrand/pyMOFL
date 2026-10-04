# SPSO 2007 & SPSO 2011 - **Benchmark-Function Handbook**

The supported pyMOFL suites select F04 Tripod, F11 Network, F18 Gear and F21
Spring from pinned 2007/2011 distributions. [Source review](spso-reference-review.md)
records the verified definitions, source captures, bounds and version differences.
The broader tables below are historical research notes; entries outside that
selected scope have not been independently source-validated by this work.

---

## 1 Standalone / Engineering Test Problems

| SPSO ID | Function                   | Dimension   | Category                  | Short description                                                            | Canonical source                                                                                                            |
| ------- | -------------------------- | ----------- | ------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 4       | **Tripod**                 | 2           | multimodal, non-separable | Piecewise linear source definition; sign(0)=0. | [Pinned source review](spso-reference-review.md) |
| 11      | **Network**                | 42 (mixed)  | hybrid, partly binary     | 38 binary links followed by four continuous BSC coordinates. | [Pinned source review](spso-reference-review.md) |
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

Both pinned distributions define Tripod (ID4). The selected Spring objectives
differ: 2007 retains the historical g2 penalty multiplier bug; 2011 corrects it.
The library exposes these source variants explicitly. Neither the selected
catalog nor this review claims implementation of the entire source suite or
its optimizer algorithms.

---

## 4 Implementing in **pyMOFL**

```python
import pyMOFL

suite = pyMOFL.get_suite("spso2011")
print([function.dimension for function in suite])  # [2, 42, 4, 3]
spring = pyMOFL.load("spso2011_f21")
print(spring.dimension)  # 3; source coordinates are [N, D, d]
```

Selected source suites use the existing config-driven function/transform
pipeline. Omit a suite-wide dimension for these heterogeneous fixed entries.
Existing generic engineering aliases retain their definitions. Spring has no
certified optimum point in the acquired source; metadata does not clip or enforce
bounds. [API notes](api-compatibility.md) explain loading and quantization limits.

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

This handbook lives at `docs/SPSO_Functions.md`; the pinned source review is the
authority for the currently selected library scope.

[1]: https://robertmarks.org/Classes/ENGR5358/Papers/functions.pdf?utm_source=chatgpt.com "[PDF] Test functions for optimization needs - Robert Marks.org"
[2]: https://www.researchgate.net/publication/255756848_Standard_Particle_Swarm_Optimisation_2011_at_CEC-2013_A_baseline_for_future_PSO_improvements "(PDF) Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future PSO improvements"
[3]: https://citeseerx.ist.psu.edu/document?doi=7b2ea6ffdb72c9c0d30389c8e8d720c6e9041b6c&repid=rep1&type=pdf&utm_source=chatgpt.com "De Jong, K. A. (1975). An analysis of the behavior of a ... - CiteSeerX"
[4]: https://royalsocietypublishing.org/doi/10.1098/rspa.1924.0082?utm_source=chatgpt.com "On the determination of molecular fields. —II. From the equation of ..."
[5]: https://asmedigitalcollection.asme.org/mechanicaldesign/issue/112/2?utm_source=chatgpt.com "Volume 112 Issue 2 | J. Mech. Des. - ASME Digital Collection"
[6]: https://www.sfu.ca/~ssurjano/permdb.html?utm_source=chatgpt.com "Perm Function d, beta"
[7]: https://www.sciencedirect.com/science/article/abs/pii/S0045782599003898?utm_source=chatgpt.com "An efficient constraint handling method for genetic algorithms"
[8]: https://www.researchgate.net/publication/235710019_Problem_Definitions_and_Evaluation_Criteria_for_the_CEC_2005_Special_Session_on_Real-Parameter_Optimization?utm_source=chatgpt.com "(PDF) Problem Definitions and Evaluation Criteria for the CEC 2005 ..."
[9]: https://www.scirp.org/reference/referencespapers?referenceid=610558&utm_source=chatgpt.com "L. A. Rastrigin, “Systems of Extreme Control,” Nauka, Moscow, 1974."
