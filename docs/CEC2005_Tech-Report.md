# Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session on Real-Parameter Optimization

P. N. Suganthan<sup>1</sup>, N. Hansen<sup>2</sup>, J. J. Liang<sup>1</sup>, K. Deb<sup>3</sup>, Y.-P. Chen<sup>4</sup>, A. Auger<sup>2</sup>, S. Tiwari<sup>3</sup>

<sup>1</sup>School of EEE, Nanyang Technological University, Singapore, 639798 [cite: 1]
<sup>2</sup>ETH Zurich, Switzerland [cite: 1]
<sup>3</sup>Kanpur Genetic Algorithms Laboratory (KanGAL), Indian Institute of Technology, Kanpur, PIN 208 016, India [cite: 1]
<sup>4</sup>Natural Computing Laboratory, Department of Computer Science, National Chiao Tung University, Taiwan [cite: 1]

*Emails:* epnsugan@ntu.edu.sg, Nikolaus.Hansen@inf.ethz.ch, liangjing@pmail.ntu.edu.sg, deb@iitk.ac.in, ypchen@csie.nctu.edu.tw, Anne.Auger@inf.ethz.ch, tiwaris@iitk.ac.in [cite: 1]

**Technical Report, Nanyang Technological University, Singapore**
**And**
**KanGAL Report Number 2005005 (Kanpur Genetic Algorithms Laboratory, IIT Kanpur)** [cite: 1]

**May 2005** [cite: 1]

## Acknowledgement

We also acknowledge the contributions by Drs / Professors Maurice Clerc (Maurice.Clerc@WriteMe.com), Bogdan Filipic (bogdan.filipic@ijs.si), William Hart (wehart@sandia.gov), Marc Schoenauer (Marc.Schoenauer@lri.fr), Hans-Paul Schwefel (hans-paul.schwefel@cs.uni-dortmund.de), Aristin Pedro Ballester (p.ballester@imperial.ac.uk) and Darrell Whitley (whitley@CS.ColoState.EDU). [cite: 1]

---

## Problem Definitions and Evaluation Criteria for the CEC 2005 Special Session on Real-Parameter Optimization

In the past two decades, different kinds of optimization algorithms have been designed and applied to solve real-parameter function optimization problems. [cite: 4, 5] Some of the popular approaches are real-parameter EAs, evolution strategies (ES), differential evolution (DE), particle swarm optimization (PSO), evolutionary programming (EP), classical methods such as quasi-Newton method (QN), hybrid evolutionary-classical methods, other non-evolutionary methods such as simulated annealing (SA), tabu search (TS) and others. [cite: 5] Under each category, there exist many different methods varying in their operators and working principles, such as correlated ES and CMA-ES. [cite: 6] In most such studies, a subset of the standard test problems (Sphere, Schwefel's, Rosenbrock's, Rastrigin's, etc.) is considered. [cite: 7] Although some comparisons are made in some research studies, often they are confusing and limited to the test problems used in the study. [cite: 8] In some occasions, the test problem and chosen algorithm are complementary to each other and the same algorithm may not work in other problems that well. [cite: 9] There is definitely a need of evaluating these methods in a more systematic manner by specifying a common termination criterion, size of problems, initialization scheme, linkages/rotation, etc. [cite: 10] There is also a need to perform a scalability study demonstrating how the running time/evaluations increase with an increase in the problem size. [cite: 10] We would also like to include some real world problems in our standard test suite with codes/executables. [cite: 11]

In this report, 25 benchmark functions are given and experiments are conducted on some real-parameter optimization algorithms. [cite: 12] The codes in Matlab, C and Java for them could be found at [http://www.ntu.edu.sg/home/EPNSugan/](http://www.ntu.edu.sg/home/EPNSugan/). [cite: 13] The mathematical formulas and properties of these functions are described in Section 2. [cite: 14] In Section 3, the evaluation criteria are given. [cite: 14] Some notes are given in Section 4. [cite: 15]

## 1. Summary of the 25 CEC'05 Test Functions

### Unimodal Functions (5):

- $F_1$: Shifted Sphere Function [cite: 15]
- $F_2$: Shifted Schwefel's Problem 1.2 [cite: 15]
- $F_3$: Shifted Rotated High Conditioned Elliptic Function [cite: 15]
- $F_4$: Shifted Schwefel's Problem 1.2 with Noise in Fitness [cite: 15]
- $F_5$: Schwefel's Problem 2.6 with Global Optimum on Bounds [cite: 15]

### Multimodal Functions (20):

#### Basic Functions (7):

- $F_6$: Shifted Rosenbrock's Function [cite: 15]
- $F_7$: Shifted Rotated Griewank's Function without Bounds [cite: 15]
- $F_8$: Shifted Rotated Ackley's Function with Global Optimum on Bounds [cite: 15]
- $F_9$: Shifted Rastrigin's Function [cite: 15]
- $F_{10}$: Shifted Rotated Rastrigin's Function [cite: 15]
- $F_{11}$: Shifted Rotated Weierstrass Function [cite: 15]
- $F_{12}$: Schwefel's Problem 2.13 [cite: 15]

#### Expanded Functions (2):
- $F_{13}$: Expanded Extended Griewank's plus Rosenbrock's Function (F8F2) [cite: 16]
- $F_{14}$: Shifted Rotated Expanded Scaffer's F6 [cite: 16]

#### Hybrid Composition Functions (11):
- $F_{15}$: Hybrid Composition Function [cite: 16]
- $F_{16}$: Rotated Hybrid Composition Function [cite: 16]
- $F_{17}$: Rotated Hybrid Composition Function with Noise in Fitness [cite: 16]
- $F_{18}$: Rotated Hybrid Composition Function [cite: 16]
- $F_{19}$: Rotated Hybrid Composition Function with a Narrow Basin for the Global Optimum [cite: 16]
- $F_{20}$: Rotated Hybrid Composition Function with the Global Optimum on the Bounds [cite: 16]
- $F_{21}$: Rotated Hybrid Composition Function [cite: 16]
- $F_{22}$: Rotated Hybrid Composition Function with High Condition Number Matrix [cite: 16]
- $F_{23}$: Non-Continuous Rotated Hybrid Composition Function [cite: 16]
- $F_{24}$: Rotated Hybrid Composition Function [cite: 16]
- $F_{25}$: Rotated Hybrid Composition Function without Bounds [cite: 16]

#### Pseudo-Real Problems:
Available from [http://www.cs.colostate.edu/~genitor/functions.html](http://www.cs.colostate.edu/~genitor/functions.html). [cite: 16] If you have any queries on these problems, please contact Professor Darrell Whitley. Email: whitley@CS.ColoState.EDU [cite: 17]

---

## 2. Definitions of the 25 CEC'05 Test Functions [cite: 18]

### 2.1 Unimodal Functions:

#### 2.1.1. $F_1$: Shifted Sphere Function [cite: 18]

$F_1(x) = \sum_{i=1}^{D} z_i^2 + f_{\text{bias}_1}$. [cite: 18]

$z = x - o$, where $x = [x_1, x_2, ..., x_D]$. [cite: 18]
D: dimensions. [cite: 18]
$o = [o_1, o_2, ..., o_D]$: the shifted global optimum. [cite: 19]

**Properties:** [cite: 19]
- Unimodal [cite: 19]
- Shifted [cite: 19]
- Separable [cite: 19]
- Scalable [cite: 19]

Search range: $x \in [-100, 100]^D$. [cite: 19]
Global optimum: $x^* = o$. [cite: 19]
$F_1(x^*) = f_{\text{bias}_1} = -450$. [cite: 19]

*Figure 2-1 3-D map for 2-D function* [cite: 19]

**Associated Data files:** [cite: 20]

| File Name              | Variable     | Description                                                                 | Notes for Usage              |
|------------------------|--------------|-----------------------------------------------------------------------------|------------------------------|
| `f01/shift_D50.txt`    | `o`          | $1 \times 50$ vector, the shifted global optimum                            | When using, cut $o = o(1:D)$ |
| `fbias_data.mat`       | `f_bias`     | $1 \times 25$ vector, records all the 25 function's $f_{\text{bias}_i}$ values | Function biases handled in code |

#### 2.1.2. $F_2$: Shifted Schwefel's Problem 1.2 [cite: 22]

$F_2(x) = \sum_{i=1}^{D} \left( \sum_{j=1}^{i} z_j \right)^2 + f_{\text{bias}_2}$. [cite: 22]

D: dimensions. [cite: 22]
$z = x - o$, where $x = [x_1, x_2, ..., x_D]$. [cite: 22]
$o = [o_1, o_2, ..., o_D]$: the shifted global optimum. [cite: 22]

**Properties:** [cite: 22]
- Unimodal [cite: 22]
- Shifted [cite: 22]
- Non-separable [cite: 22]
- Scalable [cite: 22]

Search range: $x \in [-100, 100]^D$. [cite: 22]
Global optimum $x^* = o$. [cite: 22]
$F_2(x^*) = f_{\text{bias}_2} = -450$. [cite: 22]

*Figure 2-2 3-D map for 2-D function* [cite: 22]

**Associated Data files:** [cite: 22]

| File Name                 | Variable | Description                               | Notes for Usage              |
|---------------------------|----------|-------------------------------------------|------------------------------|
| `f02/shift_D50.txt`       | `o`      | $1 \times 50$ vector, the shifted global optimum | When using, cut $o = o(1:D)$ |

#### 2.1.3. $F_3$: Shifted Rotated High Conditioned Elliptic Function [cite: 23]

$F_3(x) = \sum_{i=1}^{D} (10^6)^{\frac{i-1}{D-1}} z_i^2 + f_{\text{bias}_3}$. [cite: 23]

D: dimensions. [cite: 23]
$z = (x - o) \cdot M$, where $x = [x_1, x_2, ..., x_D]$. [cite: 23]
$o = [o_1, o_2, ..., o_D]$: the shifted global optimum. [cite: 23]
M: orthogonal matrix. [cite: 23]

**Properties:** [cite: 23]
- Unimodal [cite: 23]
- Shifted [cite: 23]
- Rotated [cite: 23]
- Non-separable [cite: 23]
- Scalable [cite: 23]

Search range: $x \in [-100, 100]^D$. [cite: 23]
Global optimum $x^* = o$. [cite: 23]
$F_3(x^*) = f_{\text{bias}_3} = -450$. [cite: 23]

*Figure 2-3 3-D map for 2-D function* [cite: 23]

**Associated Data files:** [cite: 24]

| File Name                      | Variable | Description                               | Notes for Usage              |
|--------------------------------|----------|-------------------------------------------|------------------------------|
| `f03/shift_D50.txt`            | `o`      | $1 \times 50$ vector, the shifted global optimum | When using, cut $o = o(1:D)$ |
| `f03/rot_D2.txt`               | `M`      | $2 \times 2$ matrix                        |                              |
| `f03/rot_D10.txt`              | `M`      | $10 \times 10$ matrix                      |                              |
| `f03/rot_D30.txt`              | `M`      | $30 \times 30$ matrix                      |                              |
| `f03/rot_D50.txt`              | `M`      | $50 \times 50$ matrix                      |                              |

#### 2.1.4. $F_4$: Shifted Schwefel's Problem 1.2 with Noise in Fitness

$F_4(x) = \left( \sum_{i=1}^{D} \left( \sum_{j=1}^{i} z_j \right)^2 \right) \cdot (1 + 0.4 \cdot |N(0,1)|) + f_{\text{bias}_4}$.

$z = x - o$, where $x = [x_1, x_2, ..., x_D]$.
D: dimensions.
$o = [o_1, o_2, ..., o_D]$: the shifted global optimum.
$N(0,1)$ is a normally distributed random number with mean 0 and standard deviation 1.

**Properties:**
- Unimodal
- Shifted
- Non-separable
- Scalable
- Noise in fitness

Search range: $x \in [-100, 100]^D$.
Global optimum $x^* = o$.
$F_4(x^*) = f_{\text{bias}_4} = -450$.

*Figure 2-4 3-D map for 2-D function*

**Associated Data file:**

| File Name               | Variable | Description                               | Notes for Usage              |
|-------------------------|----------|-------------------------------------------|------------------------------|
| `f04/shift_D50.txt`     | `o`      | $1 \times 50$ vector, the shifted global optimum | When using, cut $o = o(1:D)$ |

#### 2.1.5. $F_5$: Schwefel's Problem 2.6 with Global Optimum on Bounds

For 2-D, the original function is: $f(x_1, x_2) = \max\{|x_1 + 2x_2 - 7|, |2x_1 + x_2 - 5|\}$. For $x^* = [1,3]$, $f(x^*) = 0$.

Extended to D dimensions:
$F_5(x) = \max_{i=1,...,D} \{|A_i x - B_i|\} + f_{\text{bias}_5}$.

Where $x = [x_1, x_2, ..., x_D]$.
D: dimensions.
A is a $D \times D$ matrix, where $a_{ij}$ are integer random numbers in the range [-500, 500], and $\det(A) \neq 0$. $A_i$ is the $i^{th}$ row of A.
$B_i = A_i \cdot o$.
$o = [o_1, o_2, ..., o_D]^T$ is a $D \times 1$ vector, where $o_i$ are random numbers in the range [-100, 100].
After loading the data file, set $o_i = -100$ for $i=1, 2, ..., \lceil D/4 \rceil$, and $o_i = 100$ for $i=\lfloor 3D/4 \rfloor, ..., D$.

**Properties:**
- Unimodal
- Non-separable
- Scalable
- If the initialization procedure initializes the population at the bounds, this problem will be solved easily.

Search range: $x \in [-100, 100]^D$.
Global optimum $x^* = o$.
$F_5(x^*) = f_{\text{bias}_5} = -310$.

*Figure 2-5 3-D map for 2-D function*

**Associated Data file:**

| File Name                 | Variable | Description                                                                                                | Notes for Usage                     |
|---------------------------|----------|------------------------------------------------------------------------------------------------------------|-------------------------------------|
| `f05/shift_D50.txt`       | `o, A`   | File containing the shifted global optimum and linear transformation matrix                                | When using, cut $o = o(1:D)$, $A = A(1:D, 1:D)$ |

### 2.2 Basic Multimodal Functions

#### 2.2.1. $F_6$: Shifted Rosenbrock's Function

$F_6(x) = \sum_{i=1}^{D-1} \left( 100(z_i^2 - z_{i+1})^2 + (z_i - 1)^2 \right) + f_{\text{bias}_6}$.

$z = x - o + 1$, where $x = [x_1, x_2, ..., x_D]$.
D: dimensions.
$o = [o_1, o_2, ..., o_D]$: the shifted global optimum.

**Properties:**
- Multi-modal
- Shifted
- Non-separable
- Scalable
- Having a very narrow valley from local optimum to global optimum

Search range: $x \in [-100, 100]^D$.
Global optimum $x^* = o$.
$F_6(x^*) = f_{\text{bias}_6} = 390$.

*Figure 2-6 3-D map for 2-D function*

**Associated Data file:**

| File Name                   | Variable | Description                               | Notes for Usage              |
|-----------------------------|----------|-------------------------------------------|------------------------------|
| `f06/shift_D50.txt`         | `o`      | $1 \times 50$ vector, the shifted global optimum | When using, cut $o = o(1:D)$ |

#### 2.2.2. $F_7$: Shifted Rotated Griewank's Function without Bounds

$F_7(x) = \sum_{i=1}^{D} \frac{z_i^2}{4000} - \prod_{i=1}^{D} \cos\left(\frac{z_i}{\sqrt{i}}\right) + 1 + f_{\text{bias}_7}$.

$z = (x - o) \cdot M$, where $x = [x_1, x_2, ..., x_D]$.
D: dimensions.
$o = [o_1, o_2, ..., o_D]$: the shifted global optimum.
$M'$: linear transformation matrix, condition number = 3.
$M = M' \cdot (1 + 0.3 \cdot |N(0,1)|)$.

**Properties:**
- Multi-modal
- Rotated
- Shifted
- Non-separable
- Scalable
- No bounds for variables $x$.
- Initialize population in $[0, 600]^D$. Global optimum $x^* = o$ is outside of the initialization range.
  $F_7(x^*) = f_{\text{bias}_7} = -180$.

*Figure 2-7 3-D map for 2-D function*

**Associated Data file:**

| File Name                 | Variable | Description                               | Notes for Usage              |
|---------------------------|----------|-------------------------------------------|------------------------------|
| `f07/shift_D50.txt`       | `o`      | $1 \times 50$ vector, the shifted global optimum | When using, cut $o = o(1:D)$ |
| `f07/rot_D2.txt`          | `M`      | $2 \times 2$ matrix                        |                              |
| `f07/rot_D10.txt`         | `M`      | $10 \times 10$ matrix                      |                              |
| `f07/rot_D30.txt`         | `M`      | $30 \times 30$ matrix                      |                              |
| `f07/rot_D50.txt`         | `M`      | $50 \times 50$ matrix                      |                              |

#### 2.2.3. $F_8$: Shifted Rotated Ackley's Function with Global Optimum on Bounds

$F_8(x) = -20 \exp\left(-0.2 \sqrt{\frac{1}{D}\sum_{i=1}^{D} z_i^2}\right) - \exp\left(\frac{1}{D}\sum_{i=1}^{D} \cos(2\pi z_i)\right) + 20 + e + f_{\text{bias}_8}$.

$z = (x - o) \cdot M$, where $x = [x_1, x_2, ..., x_D]$.
D: dimensions.
$o = [o_1, o_2, ..., o_D]$: the shifted global optimum.
After loading the data file, set $o_{2j-1} = -32$ (the PDF states "$o_{2j-1}=-32o_{2j}$ are randomly distributed", which is ambiguous. Assuming it means some coordinates of $o$ are set to the bound or near it. The text "$o_{2j}$ are randomly distributed in the search range, for $j=1,2,...,\lfloor D/2\rfloor$" also appears in the PDF. For clarity and to reflect the "Global Optimum on Bounds" property, typically some components of $o$ are set to the boundary values, -32 or +32. The exact procedure for setting $o$ would depend on the specific data file implementation details. The file `ackley_func_data.mat` or `.txt` contains the actual $o$ values.)
M: linear transformation matrix, condition number = 100.

**Properties:**
- Multi-modal
- Rotated
- Shifted
- Non-separable
- Scalable
- Global optimum on the bound.
- If the initialization procedure initializes the population at the bounds, this problem will be solved easily.

Search range: $x \in [-32, 32]^D$.
Global optimum $x^* = o$.
$F_8(x^*) = f_{\text{bias}_8} = -140$.

*Figure 2-8 3-D map for 2-D function*

**Associated Data file:**

| File Name             | Variable | Description                               | Notes for Usage              |
|-----------------------|----------|-------------------------------------------|------------------------------|
| `f08/shift_D50.txt`   | `o`      | $1 \times 50$ vector, the shifted global optimum | When using, cut $o = o(1:D)$ |
| `f08/rot_D2.txt`      | `M`      | $2 \times 2$ matrix                        |                              |
| `f08/rot_D10.txt`     | `M`      | $10 \times 10$ matrix                      |                              |
| `f08/rot_D30.txt`     | `M`      | $30 \times 30$ matrix                      |                              |
| `f08/rot_D50.txt`     | `M`      | $50 \times 50$ matrix                      |                              |

#### 2.2.4. $F_9$: Shifted Rastrigin's Function

$F_9(x) = \sum_{i=1}^{D} (z_i^2 - 10 \cos(2\pi z_i) + 10) + f_{\text{bias}_9}$.

$z = x - o$, where $x = [x_1, x_2, ..., x_D]$.
D: dimensions.
$o = [o_1, o_2, ..., o_D]$: the shifted global optimum.

**Properties:**
- Multi-modal
- Shifted
- Separable
- Scalable
- Local optima's number is huge.

Search range: $x \in [-5, 5]^D$.
Global optimum $x^* = o$.
$F_9(x^*) = f_{\text{bias}_9} = -330$.

*Figure 2-9 3-D map for 2-D function*

**Associated Data file:**

| File Name                 | Variable | Description                               | Notes for Usage              |
|---------------------------|----------|-------------------------------------------|------------------------------|
| `f09/shift_D50.txt`       | `o`      | $1 \times 50$ vector, the shifted global optimum | When using, cut $o = o(1:D)$ |

#### 2.2.5. $F_{10}$: Shifted Rotated Rastrigin's Function

$F_{10}(x) = \sum_{i=1}^{D} (z_i^2 - 10 \cos(2\pi z_i) + 10) + f_{\text{bias}_{10}}$.

$z = (x - o) \cdot M$, where $x = [x_1, x_2, ..., x_D]$.
D: dimensions.
$o = [o_1, o_2, ..., o_D]$: the shifted global optimum.
M: linear transformation matrix, condition number = 2.

**Properties:**
- Multi-modal
- Shifted
- Rotated
- Non-separable
- Scalable
- Local optima's number is huge.

Search range: $x \in [-5, 5]^D$.
Global optimum $x^* = o$.
$F_{10}(x^*) = f_{\text{bias}_{10}} = -330$.

*Figure 2-10 3-D map for 2-D function*

**Associated Data file:**

| File Name                 | Variable | Description                               | Notes for Usage              |
|---------------------------|----------|-------------------------------------------|------------------------------|
| `f10/shift_D50.txt`       | `o`      | $1 \times 50$ vector, the shifted global optimum | When using, cut $o = o(1:D)$ |
| `f10/rot_D2.txt`          | `M`      | $2 \times 2$ matrix                        |                              |
| `f10/rot_D10.txt`         | `M`      | $10 \times 10$ matrix                      |                              |
| `f10/rot_D30.txt`         | `M`      | $30 \times 30$ matrix                      |                              |
| `f10/rot_D50.txt`         | `M`      | $50 \times 50$ matrix                      |                              |

#### 2.2.6. $F_{11}$: Shifted Rotated Weierstrass Function

$F_{11}(x) = \sum_{i=1}^{D} \left( \sum_{k=0}^{k_{\text{max}}} [a^k \cos(2\pi b^k (z_i + 0.5))] \right) - D \sum_{k=0}^{k_{\text{max}}} [a^k \cos(2\pi b^k \cdot 0.5)] + f_{\text{bias}_{11}}$.

$a = 0.5$, $b = 3$, $k_{\text{max}} = 20$.
$z = (x - o) \cdot M$, where $x = [x_1, x_2, ..., x_D]$.
D: dimensions.
$o = [o_1, o_2, ..., o_D]$: the shifted global optimum.
M: linear transformation matrix, condition number = 5.

**Properties:**
- Multi-modal
- Shifted
- Rotated
- Non-separable
- Scalable
- Continuous but differentiable only on a set of points.

Search range: $x \in [-0.5, 0.5]^D$.
Global optimum $x^* = o$.
$F_{11}(x^*) = f_{\text{bias}_{11}} = 90$.

*Figure 2-11 3-D map for 2-D function*

**Associated Data file:**

| File Name                 | Variable | Description                               | Notes for Usage              |
|---------------------------|----------|-------------------------------------------|------------------------------|
| `f11/shift_D50.txt`       | `o`      | $1 \times 50$ vector, the shifted global optimum | When using, cut $o = o(1:D)$ |
| `f11/rot_D2.txt`          | `M`      | $2 \times 2$ matrix                        |                              |
| `f11/rot_D10.txt`         | `M`      | $10 \times 10$ matrix                      |                              |
| `f11/rot_D30.txt`         | `M`      | $30 \times 30$ matrix                      |                              |
| `f11/rot_D50.txt`         | `M`      | $50 \times 50$ matrix                      |                              |

#### 2.2.7. $F_{12}$: Schwefel's Problem 2.13

$F_{12}(x) = \sum_{i=1}^{D} (A_i - B_i(x))^2 + f_{\text{bias}_{12}}$.

Where $x = [x_1, x_2, ..., x_D]$.
D: dimensions.
$A_i = \sum_{j=1}^{D} (a_{ij} \sin(\alpha_j) + b_{ij} \cos(\alpha_j))$.
$B_i(x) = \sum_{j=1}^{D} (a_{ij} \sin(x_j) + b_{ij} \cos(x_j))$, for $i=1,...,D$.

A, B are two $D \times D$ matrices; $a_{ij}, b_{ij}$ are integer random numbers in the range [-100, 100].
$\alpha = [\alpha_1, \alpha_2, ..., \alpha_D]$; $\alpha_j$ are random numbers in the range $[-\pi, \pi]$.

**Properties:**
- Multi-modal
- Shifted
- Non-separable
- Scalable

Search range: $x \in [-\pi, \pi]^D$.
Global optimum $x^* = \alpha$.
$F_{12}(x^*) = f_{\text{bias}_{12}} = -460$.

*Figure 2-12 3-D map for 2-D function*

**Associated Data file:**

| File Name                 | Variable | Description                                                                                                                               | Notes for Usage                                                                |
|---------------------------|----------|-------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `f12/bias_D50.txt`        | `alpha, a, b` | File containing alpha, a and b matrices                                                                                                    | When using, cut to appropriate dimensions                                       |

### 2.3 Expanded Functions

Using a 2-D function $F(x,y)$ as a starting function, the corresponding expanded function is:
$EF(x_1, x_2, ..., x_D) = F(x_1, x_2) + F(x_2, x_3) + ... + F(x_{D-1}, x_D) + F(x_D, x_1)$.

#### 2.3.1. $F_{13}$: Shifted Expanded Griewank's plus Rosenbrock's Function (F8F2)

F8: Griewank's Function: $F_8(x) = \sum_{i=1}^{D} \frac{x_i^2}{4000} - \prod_{i=1}^{D} \cos\left(\frac{x_i}{\sqrt{i}}\right) + 1$.
F2: Rosenbrock's Function: $F_2(x) = \sum_{i=1}^{D-1} (100(x_i^2 - x_{i+1})^2 + (x_i - 1)^2)$.

$F8F2(x_1, x_2, ..., x_D) = F8(F2(x_1, x_2)) + F8(F2(x_2, x_3)) + ... + F8(F2(x_{D-1}, x_D)) + F8(F2(x_D, x_1))$.

Shifted to:
$F_{13}(x) = F8(F2(z_1, z_2)) + F8(F2(z_2, z_3)) + ... + F8(F2(z_{D-1}, z_D)) + F8(F2(z_D, z_1)) + f_{\text{bias}_{13}}$.

$z = x - o + 1$, where $x = [x_1, x_2, ..., x_D]$.
D: dimensions.
$o = [o_1, o_2, ..., o_D]$: the shifted global optimum.

**Properties:**
- Multi-modal
- Shifted
- Non-separable
- Scalable

Search range: $x \in [-5, 5]^D$.
Global optimum $x^* = o$.
$F_{13}(x^*) = f_{\text{bias}_{13}} = -130$. (Note: PDF says $f_{\text{bias}_{13}}(13)$, this is likely a typo and should be $f_{\text{bias}_{13}}$ or referencing the value for function 13).

*Figure 2-13 3-D map for 2-D function*

**Associated Data file:**

| File Name                 | Variable | Description                               | Notes for Usage              |
|---------------------------|----------|-------------------------------------------|------------------------------|
| `f13/shift_D50.txt`       | `o`      | $1 \times 50$ vector, the shifted global optimum | When using, cut $o = o(1:D)$ |

#### 2.3.2. $F_{14}$: Shifted Rotated Expanded Scaffer's F6 Function

Original 2-D Scaffer's F6 function: $F(x,y) = 0.5 + \frac{\sin^2(\sqrt{x^2+y^2}) - 0.5}{(1 + 0.001(x^2+y^2))^2}$.

Expanded to:
$F_{14}(x) = EF(z_1, z_2, ..., z_D) = F(z_1, z_2) + F(z_2, z_3) + ... + F(z_{D-1}, z_D) + F(z_D, z_1) + f_{\text{bias}_{14}}$.

$z = (x - o) \cdot M$, where $x = [x_1, x_2, ..., x_D]$.
D: dimensions.
$o = [o_1, o_2, ..., o_D]$: the shifted global optimum.
M: linear transformation matrix, condition number = 3.

**Properties:**
- Multi-modal
- Shifted
- Non-separable
- Scalable

Search range: $x \in [-100, 100]^D$.
Global optimum $x^* = o$.
$F_{14}(x^*) = f_{\text{bias}_{14}} = -300$. (Note: PDF says $f_{\text{bias}_{14}}(14)$, typo, should be $f_{\text{bias}_{14}}$).

*Figure 2-14 3-D map for 2-D function*

**Associated Data file:**

| File Name                     | Variable | Description                               | Notes for Usage              |
|-------------------------------|----------|-------------------------------------------|------------------------------|
| `f14/shift_D50.txt`           | `o`      | $1 \times 50$ vector, the shifted global optimum | When using, cut $o = o(1:D)$ |
| `f14/rot_D2.txt`              | `M`      | $2 \times 2$ matrix                        |                              |
| `f14/rot_D10.txt`             | `M`      | $10 \times 10$ matrix                      |                              |
| `f14/rot_D30.txt`             | `M`      | $30 \times 30$ matrix                      |                              |
| `f14/rot_D50.txt`             | `M`      | $50 \times 50$ matrix                      |                              |

### 2.4 Composition Functions

Let $F(x)$ be the new composition function.
$f_i(x)$: $i^{th}$ basic function used to construct the composition function.
n: number of basic functions.
D: dimensions.
$M_i$: linear transformation matrix for each $f_i(x)$.
$o_i$: new shifted optimum position for each $f_i(x)$.

The general form of a composition function is:
$F(x) = \sum_{i=1}^{n} \{w_i \cdot [f'_i(((x-o_i)/\lambda_i) \cdot M_i) + \text{bias}_i]\} + f_{\text{g_bias}}$
(Note: The PDF uses $f_{\text{_bias}}$ at the end, I'm using $f_{\text{g_bias}}$ for global bias to distinguish from $\text{bias}_i$).

$w_i$: weight value for each $f_i(x)$, calculated as below:
$w_i = \exp\left(-\frac{\sum_{k=1}^{D}(x_k - o_{ik})^2}{2D\sigma_i^2}\right)$.
(The PDF has $o_{ik}$ as $\sigma_{ik}$ in the formula, but based on context and typical usage, $o_{ik}$ as the k-th component of the $i$-th optimum $o_i$ makes more sense for calculating distance to the center of the function's influence).

The weights $w_i$ are then adjusted:
If $w_j = \max(w_1, ..., w_n)$, then $w_j = w_j$.
Otherwise, $w_j = w_j \cdot (1 - (\max(w_1, ..., w_n))^{10})$.
(The PDF notation is $w_i = \begin{cases} w_i & w_i = \max(w_i) \\ w_i \cdot (1-\max(w_i).\textasciicircum10) & w_i \neq \max(w_i) \end{cases}$. The condition $w_i=\max(w_i)$ should refer to $w_i$ being the maximum *among all* weights for that $x$).

Then, normalize the weights: $w_i = \frac{w_i}{\sum_{j=1}^{n} w_j}$.

$\sigma_i$: used to control each $f_i(x)$'s coverage range; a small $\sigma_i$ gives a narrow range for that $f_i(x)$.
$\lambda_i$: used to stretch or compress the function; $\lambda_i > 1$ means stretch, $\lambda_i < 1$ means compress.
$o_i$: define the global and local optima's position.
$\text{bias}_i$: define which optimum is global optimum. Using $o_i$ and $\text{bias}_i$, a global optimum can be placed anywhere.

If $f_i(x)$ are different functions, they have different properties and heights. To get a better mixture, estimate the maximum function value $f_{\text{max}_i}$ for each basic function $f_i(x)$. Then, normalize each basic function to similar heights:
$f'_i(x) = C \cdot \frac{f_i(x)}{|f_{\text{max}_i}|}$, where C is a predefined constant.
$|f_{\text{max}_i}|$ is estimated using $f_{\text{max}_i} = f_i((x'/\lambda_i) \cdot M_i)$, where $x' = [5,5,...,5]$.

In the following composition functions:
- Number of basic functions $n=10$.
- D: dimensions.
- $o$: $n \times D$ matrix, defines $f_i(x)$'s global optimal positions (This is $o_i$ for each function).
- $\text{bias} = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]$. Hence, the first function $f_1(x)$ always has $\text{bias}_1 = 0$, making its associated optimum (after considering $f'_i$) the target global optimum of the composition function before adding $f_{\text{g_bias}}$.
- $C=2000$.

---
**Pseudo Code:**

```text
Define f1-f10 (basic functions)
Define σ_params (array of σ for each f_i)
Define λ_params (array of λ for each f_i)
Define bias_coefficients (array of bias for each f_i, e.g., [0, 100,...])
Define C_norm (normalization constant, e.g., 2000)
Define f_global_bias (final bias for the composed function)

Load o_optima_data (n x D matrix of optima o_i for each f_i)
Load M_rotation_data (rotation matrices M_i for each f_i, can be identity)

FUNCTION Calculate_Composed_Fitness(x_input_vector):
    n_functions = 10 // Number of basic functions
    D_dimensions = length(x_input_vector)

    raw_weights = array of size n_functions
    normalized_fitness_contributions = array of size n_functions

    // Fixed point for f_max estimation
    y_ref_point = create_vector(D_dimensions, 5.0) 

    // --- Step 1: Calculate normalized fitness and raw weights ---
    for i from 0 to n_functions-1:
        // Calculate w_i (gaussian-like weight)
        current_optimum_oi = o_optima_data[i]
        sum_sq_dist = 0
        for k from 0 to D_dimensions-1:
            sum_sq_dist += (x_input_vector[k] - current_optimum_oi[k])^2
        end for
        raw_weights[i] = exp(-sum_sq_dist / (2 * D_dimensions * σ_params[i]^2))

        // Transform input for current basic function f_i
        // z_i = ((x_input_vector - o_i) / λ_i) * M_i
        x_shifted = vector_subtract(x_input_vector, current_optimum_oi)
        x_scaled = vector_divide_scalar(x_shifted, λ_params[i])
        z_i = matrix_multiply_vector(M_rotation_data[i], x_scaled)
        
        // Calculate raw fitness of f_i at z_i
        fit_i_raw = evaluate_basic_function_by_type(function_type[i], z_i)
        
        // Estimate f_max_i for normalization
        y_ref_scaled = vector_divide_scalar(y_ref_point, λ_params[i])
        z_ref_transformed = matrix_multiply_vector(M_rotation_data[i], y_ref_scaled)
        f_max_i = evaluate_basic_function_by_type(function_type[i], z_ref_transformed)
        
        // Normalize fitness contribution
        if abs(f_max_i) < 1.0e-20: // Avoid division by zero
            normalized_fitness_contributions[i] = C_norm * fit_i_raw
        else:
            normalized_fitness_contributions[i] = C_norm * fit_i_raw / abs(f_max_i)
        end if
    end for

    // --- Step 2: Adjust and Normalize Weights ---
    current_max_raw_weight = 0.0
    for i from 0 to n_functions-1:
        if raw_weights[i] > current_max_raw_weight:
            current_max_raw_weight = raw_weights[i]
        end if
    end for

    final_weights = array of size n_functions
    sum_adjusted_weights = 0.0

    for i from 0 to n_functions-1:
        if abs(raw_weights[i] - current_max_raw_weight) < 1.0e-20: // Check if it's the max weight
            adjusted_w_i = raw_weights[i]
        else:
            adjusted_w_i = raw_weights[i] * (1.0 - power(current_max_raw_weight, 10))
        end if
        final_weights[i] = adjusted_w_i 
        sum_adjusted_weights += adjusted_w_i
    end for

    // Final normalization of weights
    if abs(sum_adjusted_weights) < 1.0e-20: // Avoid division by zero
         for i from 0 to n_functions-1:
            final_weights[i] = 1.0 / n_functions 
        end for
    else:
        for i from 0 to n_functions-1:
            final_weights[i] = final_weights[i] / sum_adjusted_weights
        end for
    end if

    // --- Step 3: Calculate Composed Function Value ---
    Composed_F_x = 0.0
    for i from 0 to n_functions-1:
        Composed_F_x += final_weights[i] * (normalized_fitness_contributions[i] + bias_coefficients[i])
    end for

    Composed_F_x += f_global_bias

    return Composed_F_x
END FUNCTION
```
#### 2.4.1. $F_{15}$: Hybrid Composition Function

This function is a composition of 10 basic functions:
- $f_1, f_2$: Rastrigin's Function
  $f(x_{vec}) = \sum_{j=1}^{D} (x_{vec,j}^2 - 10 \cos(2\pi x_{vec,j}) + 10)$
- $f_3, f_4$: Weierstrass Function
  $f(x_{vec}) = \sum_{j=1}^{D} \left( \sum_{k=0}^{k_{\text{max}}} [a^k \cos(2\pi b^k (x_{vec,j} + 0.5))] \right) - D \sum_{k=0}^{k_{\text{max}}} [a^k \cos(2\pi b^k \cdot 0.5)]$, with $a=0.5, b=3, k_{\text{max}}=20$. [cite: 1]
- $f_5, f_6$: Griewank's Function
  $f(x_{vec}) = \sum_{j=1}^{D} \frac{x_{vec,j}^2}{4000} - \prod_{j=1}^{D} \cos\left(\frac{x_{vec,j}}{\sqrt{j}}\right) + 1$ [cite: 1]
- $f_7, f_8$: Ackley's Function
  $f(x_{vec}) = -20 \exp\left(-0.2 \sqrt{\frac{1}{D}\sum_{j=1}^{D} x_{vec,j}^2}\right) - \exp\left(\frac{1}{D}\sum_{j=1}^{D} \cos(2\pi x_{vec,j})\right) + 20 + e$ [cite: 1]
- $f_9, f_{10}$: Sphere Function
  $f(x_{vec}) = \sum_{j=1}^{D} x_{vec,j}^2$ [cite: 1]

*(Note: I've used $x_{vec}$ in the basic function definitions to clearly denote the input vector to these functions, which would be $z_i$ in the context of the composition.)*

**Parameters for $F_{15}$:**
- $\sigma = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]$ (i.e., $\sigma_i=1$ for all $i=1, ..., 10$). [cite: 1]
- $\lambda = [1, 1, 10, 10, 5/60, 5/60, 5/32, 5/32, 5/100, 5/100]$. [cite: 1]
- $M_i$ are all Identity matrices for $F_{15}$. [cite: 1]
- $\text{bias_coefficients} = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]$. [cite: 1]
- $f_{\text{global_bias}} = f_{\text{bias}_{15}} = 120$. [cite: 1]

*(Important Note from PDF: The formulas above for $f_1$ to $f_{10}$ are for the basic functions themselves. When used in the composition, no shift or rotation is included in *these base expressions*; the shift by $o_i$ and rotation by $M_i$ (if $M_i$ is not identity) are applied as per the general composition formula when calculating $z_i = ((x-o_i)/\lambda_i) \cdot M_i$. For $F_{15}$, $M_i$ is identity.)* [cite: 1]

*Example from PDF for $f_1$ (Rastrigin) in composition:*
When calculating $f'_1(((x-o_1)/\lambda_1) \cdot M_1)$, we first calculate $z_1 = ((x-o_1)/\lambda_1) \cdot M_1$.
Then, the Rastrigin function is evaluated as $f_1(z_1) = \sum_{j=1}^{D} (z_{1,j}^2 - 10 \cos(2\pi z_{1,j}) + 10)$. This $f_1(z_1)$ is then normalized to $f'_1$. [cite: 1]

**Properties of $F_{15}$:**
- Multi-modal [cite: 1]
- Separable near the global optimum (due to Rastrigin $f_1$ having $\text{bias_coefficients}_1=0$ and $M_1$ being identity) [cite: 1]
- Scalable [cite: 1]
- A huge number of local optima [cite: 1]
- Different function properties are mixed together [cite: 1]
- Sphere Functions ($f_9, f_{10}$) contribute to flat areas in the landscape. [cite: 1]

Search range: $x \in [-5, 5]^D$. [cite: 1]
Global optimum is designed to be at $o_1$ (the optimum of the first basic function).
$F_{15}(x^*) = f_{\text{bias}_{15}} = 120$. [cite: 1]

*Figure 2-15 3-D map for 2-D function* [cite: 1]

**Associated Data file for $F_{15}$ (and other Hybrid Functions unless specified):**
(Generally, files like `hybrid_func1_data.mat/.txt` provide optima $o_i$, and `hybrid_func1_M_D<dims>.mat/.txt` would provide rotation matrices $M_i$ if they are not identity.)

| File Name                 | Variable | Description                                            | Notes for Usage                                     |
|---------------------------|----------|--------------------------------------------------------|-----------------------------------------------------|
| `f15/shift_D50.txt`       | `o`      | $10 \times 50$ matrix (optima $o_i$ for 10 functions) | Use $o_i = o(i, 1:D)$ for the $i$-th function. [cite: 1] |

*(For $F_{15}$, $M_i$ are identity matrices, so specific $M$ files are not strictly needed for its calculation beyond knowing they are identity.)*

#### 2.4.2. $F_{16}$: Rotated Version of Hybrid Composition Function $F_{15}$

All settings are the same as $F_{15}$ **except** $M_i$ are different linear transformation matrices with a condition number of 2. [cite: 1] The basic functions, $\sigma_i$, $\lambda_i$, $\text{bias_coefficients}$, and $f_{\text{global_bias}} = f_{\text{bias}_{16}} = 120$ remain the same as for $F_{15}$.

**Properties of $F_{16}$:**
- Multi-modal [cite: 1]
- Rotated [cite: 1]
- Non-Separable [cite: 1]
- Scalable [cite: 1]
- A huge number of local optima [cite: 1]
- Different function properties are mixed together [cite: 1]
- Sphere Functions give two flat areas for the function. [cite: 1]

Search range: $x \in [-5, 5]^D$. [cite: 1]
Global optimum $x^* \approx o_1$ (transformed by $M_1^{-1}$ if $o_1$ is the target in the unrotated space).
$F_{16}(x^*) = f_{\text{bias}_{16}} = 120$. [cite: 1]

*Figure 2-16 3-D map for 2-D function* [cite: 1]

**Associated Data file for $F_{16}$:**
Uses `f16/shift_D50.txt` for optima $o_i$.
Rotation matrices $M_i$ are provided in separate files:

| File Name                   | Variable      | Description                                                                                                                               |
|-----------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `f16/rot_D2.txt`            | `M1`...`M10`  | Ten $2 \times 2$ matrices sequentially                                                                                                    |
| `f16/rot_D10.txt`           | `M1`...`M10`  | Ten $10 \times 10$ matrices sequentially                                                                                                  |
| `f16/rot_D30.txt`           | `M1`...`M10`  | Ten $30 \times 30$ matrices sequentially                                                                                                  |
| `f16/rot_D50.txt`           | `M1`...`M10`  | Ten $50 \times 50$ matrices sequentially                                                                                                  |

*(When using, cut $o = o(:, 1:D)$ for the optima. The matrices $M_i$ are used in $z_i = ((x-o_i)/\lambda_i) \cdot M_i$.)*

#### 2.4.3. $F_{17}$: $F_{16}$ with Noise in Fitness

Let $G(x) = F_{16}(x) - f_{\text{bias}_{16}}$ (where $f_{\text{bias}_{16}}$ is the bias term for $F_{16}$, which is 120).
Then, $F_{17}(x) = G(x) \cdot (1 + 0.2 \cdot |N(0,1)|) + f_{\text{bias}_{17}}$.

All other settings (basic functions, $\sigma_i$, $\lambda_i$, $M_i$, $\text{bias_coefficients}$) are the same as for $F_{16}$.
$f_{\text{bias}_{17}} = 120$.
$N(0,1)$ is a normally distributed random number with mean 0 and standard deviation 1.

**Properties of $F_{17}$:**
- Multi-modal
- Rotated
- Non-Separable
- Scalable
- A huge number of local optima
- Different function properties are mixed together
- Sphere Functions give two flat areas for the function.
- With Gaussian noise in fitness.

Search range: $x \in [-5, 5]^D$.
Global optimum $x^* \approx o_1$ (transformed by $M_1^{-1}$).
$F_{17}(x^*) = f_{\text{bias}_{17}} = 120$.

*Figure 2-17 3-D map for 2-D function*

**Associated Data file for $F_{17}$:**
Same as $F_{16}$ (uses `f17/shift_D50.txt` for optima $o_i$ and `f17/rot_D<dims>.txt` for rotation matrices $M_i$).

#### 2.4.4. $F_{18}$: Rotated Hybrid Composition Function

This function uses a different set of 10 basic functions:
- $f_1, f_2$: Ackley's Function
  $f(x_{vec}) = -20 \exp\left(-0.2 \sqrt{\frac{1}{D}\sum_{j=1}^{D} x_{vec,j}^2}\right) - \exp\left(\frac{1}{D}\sum_{j=1}^{D} \cos(2\pi x_{vec,j})\right) + 20 + e$
- $f_3, f_4$: Rastrigin's Function
  $f(x_{vec}) = \sum_{j=1}^{D} (x_{vec,j}^2 - 10 \cos(2\pi x_{vec,j}) + 10)$
- $f_5, f_6$: Sphere Function
  $f(x_{vec}) = \sum_{j=1}^{D} x_{vec,j}^2$
- $f_7, f_8$: Weierstrass Function
  $f(x_{vec}) = \sum_{j=1}^{D} \left( \sum_{k=0}^{k_{\text{max}}} [a^k \cos(2\pi b^k (x_{vec,j} + 0.5))] \right) - D \sum_{k=0}^{k_{\text{max}}} [a^k \cos(2\pi b^k \cdot 0.5)]$, with $a=0.5, b=3, k_{\text{max}}=20$.
- $f_9, f_{10}$: Griewank's Function
  $f(x_{vec}) = \sum_{j=1}^{D} \frac{x_{vec,j}^2}{4000} - \prod_{j=1}^{D} \cos\left(\frac{x_{vec,j}}{\sqrt{j}}\right) + 1$

**Parameters for $F_{18}$:**
- $\sigma = [1, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2]$.
- $\lambda = [2 \cdot 5/32, 5/32, 2 \cdot 1, 1, 2 \cdot 5/100, 5/100, 2 \cdot 10, 10, 2 \cdot 5/60, 5/60]$.
  (Note: The PDF shows $2*5/32;$. Assuming `*` means multiplication).
- $M_i$ are all rotation matrices. Condition numbers for $M_1, ..., M_{10}$ are $[2, 3, 2, 3, 2, 3, 20, 30, 200, 300]$ respectively.
- $o_{10} = [0,0,...,0]$ (The optimum for the 10th basic function is set to the origin). Optima $o_1, ..., o_9$ are loaded from data files.
- $\text{bias_coefficients} = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]$.
- $f_{\text{global_bias}} = f_{\text{bias}_{18}} = 10$.

**Properties of $F_{18}$:**
- Multi-modal
- Rotated
- Non-Separable
- Scalable
- A huge number of local optima
- Different function properties are mixed together
- Sphere Functions give two flat areas for the function.
- A local optimum (from $f_{10}$) is set on the origin.

Search range: $x \in [-5, 5]^D$.
Global optimum $x^* \approx o_1$ (transformed by $M_1^{-1}$).
$F_{18}(x^*) = f_{\text{bias}_{18}} = 10$.

*Figure 2-18 3-D map for 2-D function*

**Associated Data file for $F_{18}$:**
Uses `f18/shift_D50.txt` for optima $o_i$.
Rotation matrices $M_i$ are provided in files:

| File Name                   | Variable      | Description                                                                                                                               |
|-----------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `f18/rot_D2.txt`            | `M1`...`M10`  | Ten $2 \times 2$ matrices sequentially                                                                                                    |
| `f18/rot_D10.txt`           | `M1`...`M10`  | Ten $10 \times 10$ matrices sequentially                                                                                                  |
| `f18/rot_D30.txt`           | `M1`...`M10`  | Ten $30 \times 30$ matrices sequentially                                                                                                  |
| `f18/rot_D50.txt`           | `M1`...`M10`  | Ten $50 \times 50$ matrices sequentially                                                                                                  |

#### 2.4.5. $F_{19}$: Rotated Hybrid Composition Function with a Narrow Basin for the Global Optimum

All settings are the same as $F_{18}$ **except** for $\sigma$ and $\lambda$ for the first basic function ($f_1$, Ackley's) to create a narrow basin for the global optimum:
- $\sigma = [\textbf{0.1}, 2, 1.5, 1.5, 1, 1, 1.5, 1.5, 2, 2]$ (First element changed from 1 to 0.1).
- $\lambda = [\textbf{0.1} \cdot 5/32, 5/32, 2 \cdot 1, 1, 2 \cdot 5/100, 5/100, 2 \cdot 10, 10, 2 \cdot 5/60, 5/60]$ (First element's scaling factor for $5/32$ changed from 2 to 0.1).
- All other parameters ($M_i$, their condition numbers, other $\sigma_i$ and $\lambda_i$, $o_i$, $\text{bias_coefficients}$, $f_{\text{global_bias}} = f_{\text{bias}_{19}} = 10$) remain the same as for $F_{18}$.

**Properties of $F_{19}$:**
- Multi-modal
- Rotated
- Non-Separable
- Scalable
- A huge number of local optima
- Different function properties are mixed together
- Sphere Functions give two flat areas for the function.
- A local optimum is set on the origin.
- **A narrow basin for the global optimum.**

Search range: $x \in [-5, 5]^D$.
Global optimum $x^* \approx o_1$ (transformed by $M_1^{-1}$).
$F_{19}(x^*) = f_{\text{bias}_{19}} = 10$. (PDF says $f_{\text{bias}_{19}}(19)$, assumed typo).

*Figure 2-19 3-D map for 2-D function*

**Associated Data file for $F_{19}$:**
Same as $F_{18}$ (uses `f19/shift_D50.txt` for optima $o_i$ and `f19/rot_D<dims>.txt` for rotation matrices $M_i$).

#### 2.4.6. $F_{20}$: Rotated Hybrid Composition Function with the Global Optimum on the Bounds

All settings are the same as $F_{18}$ **except** the location of the global optimum $o_1$ is modified.
After loading the data file for $o_i$ (from `f20/shift_D50.txt`), the coordinates of $o_1$ (the optimum for the first basic function) are adjusted:
Set $o_{1,j} = 5$ for $j = 1, 2, ..., \lfloor D/2 \rfloor$. (The PDF states "$o_{1(2,j)}=5$", which could be a typo. Interpreted as setting the first $\lfloor D/2 \rfloor$ components of $o_1$ to the upper bound, 5). This forces the global optimum onto the boundary of the search space.

- All other parameters (basic functions, $\sigma_i$, $\lambda_i$, $M_i$, their condition numbers, other $o_i$ (for $i=2..10$), $\text{bias_coefficients}$, $f_{\text{global_bias}} = f_{\text{bias}_{20}} = 10$) remain the same as for $F_{18}$.

**Properties of $F_{20}$:**
- Multi-modal
- Rotated
- Non-Separable
- Scalable
- A huge number of local optima
- Different function properties are mixed together
- Sphere Functions give two flat areas for the function.
- A local optimum is set on the origin.
- **Global optimum is on the bound.**
- If the initialization procedure initializes the population at the bounds, this problem might be solved more easily.

Search range: $x \in [-5, 5]^D$.
Global optimum $x^*$ is at the modified $o_1$ (transformed by $M_1^{-1}$).
$F_{20}(x^*) = f_{\text{bias}_{20}} = 10$.

*Figure 2-20 3-D map for 2-D function*

**Associated Data file for $F_{20}$:**
Same as $F_{18}$ (uses `f20/shift_D50.txt` for base optima $o_i$ and `f20/rot_D<dims>.txt` for rotation matrices $M_i$), with the specific modification to $o_1$ applied after loading.

#### 2.4.7. $F_{21}$: Rotated Hybrid Composition Function

This function uses a different set of 10 basic functions:
- $f_1, f_2$: Rotated Expanded Scaffer's F6 Function (as defined for $F_{14}$ but here it's a basic component, so the rotation $M_i$ for $F_{21}$ will be applied to the $z_i$ that goes into this basic Scaffer's F6 structure)
  The basic Scaffer's F6 form is $F(x,y) = 0.5 + \frac{\sin^2(\sqrt{x^2+y^2}) - 0.5}{(1 + 0.001(x^2+y^2))^2}$.
  The expanded form used here is $f(x_{vec}) = F(x_{vec,1}, x_{vec,2}) + F(x_{vec,2}, x_{vec,3}) + ... + F(x_{vec,D}, x_{vec,1})$.
- $f_3, f_4$: Rastrigin's Function
  $f(x_{vec}) = \sum_{j=1}^{D} (x_{vec,j}^2 - 10 \cos(2\pi x_{vec,j}) + 10)$
- $f_5, f_6$: F8F2 Function (Expanded Griewank's plus Rosenbrock's, as defined for $F_{13}$, used as a basic component here)
  Base F8 (Griewank): $F_8(y_{vec}) = \sum_{k=1}^{D'} \frac{y_{vec,k}^2}{4000} - \prod_{k=1}^{D'} \cos\left(\frac{y_{vec,k}}{\sqrt{k}}\right) + 1$.
  Base F2 (Rosenbrock): $F_2(y_{vec}) = \sum_{k=1}^{D'-1} (100(y_{vec,k}^2 - y_{vec,k+1})^2 + (y_{vec,k} - 1)^2)$.
  (Here $D'$ would be 2 as F8F2 applies F8 to the output of F2 which takes 2D inputs at a time in the expansion).
  The expanded F8F2 form is $f(x_{vec}) = F8(F2(x_{vec,1}, x_{vec,2})) + ... + F8(F2(x_{vec,D}, x_{vec,1}))$.
- $f_7, f_8$: Weierstrass Function
  $f(x_{vec}) = \sum_{j=1}^{D} \left( \sum_{k=0}^{k_{\text{max}}} [a^k \cos(2\pi b^k (x_{vec,j} + 0.5))] \right) - D \sum_{k=0}^{k_{\text{max}}} [a^k \cos(2\pi b^k \cdot 0.5)]$, with $a=0.5, b=3, k_{\text{max}}=20$.
- $f_9, f_{10}$: Griewank's Function
  $f(x_{vec}) = \sum_{j=1}^{D} \frac{x_{vec,j}^2}{4000} - \prod_{j=1}^{D} \cos\left(\frac{x_{vec,j}}{\sqrt{j}}\right) + 1$

**Parameters for $F_{21}$:**
- $\sigma = [1,1,1,1,1,2,2,2,2,2]$.
- $\lambda = [5 \cdot 5/100, 5/100, 5 \cdot 1, 1, 5 \cdot 1, 1, 5 \cdot 10, 10, 5 \cdot 5/200, 5/200]$.
  (The PDF format is "5\*5/100; 5/100; ...". Assuming `*` means multiplication).
- $M_i$ are all orthogonal matrices. (Condition numbers for these are not specified for $F_{21}$ but are for $F_{22}$).
- $\text{bias_coefficients} = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]$.
- $f_{\text{global_bias}} = f_{\text{bias}_{21}} = 360$.

**Properties of $F_{21}$:**
- Multi-modal
- Rotated
- Non-Separable
- Scalable
- A huge number of local optima
- Different function properties are mixed together.

Search range: $x \in [-5, 5]^D$.
Global optimum $x^* \approx o_1$ (transformed).
$F_{21}(x^*) = f_{\text{bias}_{21}} = 360$.

*Figure 2-21 3-D map for 2-D function*

**Associated Data file for $F_{21}$:**
Uses `f21/shift_D50.txt` for optima $o_i$.
Rotation matrices $M_i$ are provided in files:

| File Name                   | Variable      | Description                                                                                                                               |
|-----------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `f21/rot_D2.txt`            | `M1`...`M10`  | Ten $2 \times 2$ orthogonal matrices sequentially                                                                                         |
| `f21/rot_D10.txt`           | `M1`...`M10`  | Ten $10 \times 10$ matrices sequentially                                                                                                  |
| `f21/rot_D30.txt`           | `M1`...`M10`  | Ten $30 \times 30$ matrices sequentially                                                                                                  |
| `f21/rot_D50.txt`           | `M1`...`M10`  | Ten $50 \times 50$ matrices sequentially                                                                                                  |

#### 2.4.8. $F_{22}$: Rotated Hybrid Composition Function with High Condition Number Matrix

All settings are the same as $F_{21}$ **except** the rotation matrices $M_i$ have high condition numbers.
- The condition numbers for $M_1, ..., M_{10}$ are $[10, 20, 50, 100, 200, 1000, 2000, 3000, 4000, 5000]$ respectively.
- Basic functions, $\sigma_i$, $\lambda_i$, $o_i$, $\text{bias_coefficients}$, and $f_{\text{global_bias}} = f_{\text{bias}_{22}} = 360$ remain the same as for $F_{21}$.

**Properties of $F_{22}$:**
- Multi-modal
- Rotated (with high condition number matrices)
- Non-Separable
- Scalable
- A huge number of local optima
- Different function properties are mixed together.
- The PDF also states "Global optimum is on the bound", which seems like a copy-paste error from another function description as $F_{21}$ (its base) doesn't state this, and no parameters are changed to force this. This property might be incorrect here.

Search range: $x \in [-5, 5]^D$.
Global optimum $x^* \approx o_1$ (transformed).
$F_{22}(x^*) = f_{\text{bias}_{22}} = 360$.

*Figure 2-22 3-D map for 2-D function*

**Associated Data file for $F_{22}$:**
Uses `f22/shift_D50.txt` for optima $o_i$ (same as $F_{21}$).
Rotation matrices $M_i$ with high condition numbers are provided in files:

| File Name                    | Variable      | Description                                                                                                                               |
|------------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `f22/rot_sub_D2.txt`         | `M1`...`M10`  | Ten $2 \times 2$ high condition matrices sequentially                                                                                     |
| `f22/rot_sub_D10.txt`        | `M1`...`M10`  | Ten $10 \times 10$ matrices sequentially                                                                                                  |
| `f22/rot_sub_D30.txt`        | `M1`...`M10`  | Ten $30 \times 30$ matrices sequentially                                                                                                  |
| `f22/rot_sub_D50.txt`        | `M1`...`M10`  | Ten $50 \times 50$ matrices sequentially                                                                                                  |

#### 2.4.9. $F_{23}$: Non-Continuous Rotated Hybrid Composition Function

All settings are the same as $F_{21}$ (basic functions, $\sigma_i$, $\lambda_i$, $M_i$ which are orthogonal, $o_i$, $\text{bias_coefficients}$, $f_{\text{global_bias}} = f_{\text{bias}_{23}} = 360$) **except** for a non-continuous transformation applied to the input $x$ before it's used in the composition logic (or more precisely, as it's used to calculate distance for $w_i$ and as input to the first basic function $f_1$).

The non-continuous transformation on $x_j$ for $j=1, ..., D$:
$x'_j = x_j$ if $|x_j - o_{1j}| < 0.5$
$x'_j = \text{round}(2x_j)/2$ if $|x_j - o_{1j}| \ge 0.5$
where $o_{1j}$ is the $j$-th component of the optimum $o_1$ for the first basic function.
The `round(y)` function is defined as:
Let $a = \text{integer_part}(y)$ and $b = \text{decimal_part}(y)$.
$\text{round}(y) = a-1$ if $y \le 0$ and $b \ge 0.5$
$\text{round}(y) = a$ if $b < 0.5$
$\text{round}(y) = a+1$ if $y > 0$ and $b \ge 0.5$
(This is a specific rounding schedule, often to the nearest integer, with halves rounded based on sign or to even/odd, but here it is explicitly defined. The PDF formula $round(2x_j)/2$ means values are rounded to the nearest $0.5$).
The PDF says "All 'round' operators in this document use the same schedule." And the condition on $x_j$ depends on $o_{1j}$. This implies the discontinuity is centered around the global optimum's location $o_1$.

**Properties of $F_{23}$:**
- Multi-modal
- Rotated
- Non-Separable
- Scalable
- A huge number of local optima
- Different function properties are mixed together.
- **Non-continuous.**
- The PDF also states "Global optimum is on the bound", which, like for $F_{22}$, seems to be a copy-paste error. The discontinuity is around $o_1$, not necessarily pushing the true optimum to the bounds.

Search range: $x \in [-5, 5]^D$.
Global optimum $x^* \approx o_1$ (transformed). The function value at optimum is approximate due to discontinuity.
$F_{23}(x^*) \approx f_{\text{bias}_{23}} = 360$.

*Figure 2-23 3-D map for 2-D function*

**Associated Data file for $F_{23}$:**
Same as $F_{21}$ (uses `f23/shift_D50.txt` for optima $o_i$ and `f23/rot_D<dims>.txt` for rotation matrices $M_i$). The non-continuous transformation is an additional processing step.

#### 2.4.10. $F_{24}$: Rotated Hybrid Composition Function

This function uses yet another set of 10 basic functions:
1.  $f_1$: Weierstrass Function (as in $F_{21}$)
2.  $f_2$: Rotated Expanded Scaffer's F6 Function (as in $F_{21}$)
3.  $f_3$: F8F2 Function (as in $F_{21}$)
4.  $f_4$: Ackley's Function (as in $F_{15}$)
5.  $f_5$: Rastrigin's Function (as in $F_{15}$)
6.  $f_6$: Griewank's Function (as in $F_{15}$)
7.  $f_7$: Non-Continuous Expanded Scaffer's F6 Function.
    Base Scaffer's F6: $F(x,y) = 0.5 + \frac{\sin^2(\sqrt{x^2+y^2}) - 0.5}{(1 + 0.001(x^2+y^2))^2}$.
    Expanded form: $f(x_{vec})=F(y_1,y_2)+F(y_2,y_3)+...+F(y_D,y_1)$.
    With $y_j = x_{vec,j}$ if $|x_{vec,j}| < 0.5$, else $y_j = \text{round}(2x_{vec,j})/2$.
8.  $f_8$: Non-Continuous Rastrigin's Function.
    $f(x_{vec}) = \sum_{j=1}^{D} (y_j^2 - 10 \cos(2\pi y_j) + 10)$.
    With $y_j = x_{vec,j}$ if $|x_{vec,j}| < 0.5$, else $y_j = \text{round}(2x_{vec,j})/2$.
9.  $f_9$: High Conditioned Elliptic Function (this is $F_3$'s form, but as a basic function $f(x_{vec}) = \sum_{j=1}^{D} (10^6)^{\frac{j-1}{D-1}} x_{vec,j}^2$)
10. $f_{10}$: Sphere Function with Noise in Fitness.
    $f(x_{vec}) = \left(\sum_{j=1}^{D} x_{vec,j}^2\right) (1 + 0.1 \cdot |N(0,1)|)$

**Parameters for $F_{24}$:**
- $\sigma_i = 2$ for all $i=1, ..., 10$.
- $\lambda = [10, 5/20, 1, 5/32, 1, 5/100, 5/50, 1, 5/100, 5/100]$.
- $M_i$ are all rotation matrices. Condition numbers for $M_1, ..., M_{10}$ are $[100, 50, 30, 10, 5, 5, 4, 3, 2, 2]$ respectively.
- $\text{bias_coefficients} = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]$.
- $f_{\text{global_bias}} = f_{\text{bias}_{24}} = 260$.

**Properties of $F_{24}$:**
- Multi-modal
- Rotated
- Non-Separable
- Scalable
- A huge number of local optima
- Different function properties are mixed together, including non-continuity and noise.
- Unimodal functions (Sphere, Elliptic) contribute to flat/smoother areas.

Search range: $x \in [-5, 5]^D$.
Global optimum $x^* \approx o_1$ (transformed).
$F_{24}(x^*) = f_{\text{bias}_{24}} = 260$.

*Figure 2-24 3-D map for 2-D function*

**Associated Data file for $F_{24}$:**
Uses `f24/shift_D50.txt` for optima $o_i$.
Rotation matrices $M_i$ are provided in files:

| File Name                   | Variable      | Description                                                                                                                               |
|-----------------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `f24/rot_D2.txt`            | `M1`...`M10`  | Ten $2 \times 2$ matrices sequentially                                                                                                    |
| `f24/rot_D10.txt`           | `M1`...`M10`  | Ten $10 \times 10$ matrices sequentially                                                                                                  |
| `f24/rot_D30.txt`           | `M1`...`M10`  | Ten $30 \times 30$ matrices sequentially                                                                                                  |
| `f24/rot_D50.txt`           | `M1`...`M10`  | Ten $50 \times 50$ matrices sequentially                                                                                                  |

#### 2.4.11. $F_{25}$: Rotated Hybrid Composition Function without bounds

All settings are the same as $F_{24}$ (basic functions, $\sigma_i$, $\lambda_i$, $M_i$, their condition numbers, $o_i$, $\text{bias_coefficients}$, $f_{\text{global_bias}} = f_{\text{bias}_{25}} = 260$) **except** no exact search range is set for this test function.

**Properties of $F_{25}$:**
- Multi-modal
- Rotated
- Non-Separable
- Scalable
- A huge number of local optima
- Different function properties are mixed together.
- Unimodal functions give flat areas for the function.
- The PDF states "Global optimum is on the bound". This seems to conflict with "No bounds". The key is that the *initialization range is different from where the global optimum $o_1$ is located*.
- **No bounds** (for search, though an initialization range is given).
- Initialize population in $[2, 5]^D$. The global optimum $x^* \approx o_1$ (which is loaded from `f25/shift_D50.txt` and is likely centered around 0 before transformation by $M_1$) is outside of this initialization range.

$F_{25}(x^*) = f_{\text{bias}_{25}} = 260$.

**Associated Data file for $F_{25}$:**
Same as $F_{24}$ (uses `f25/shift_D50.txt` for optima $o_i$ and `f25/rot_D<dims>.txt` for rotation matrices $M_i$).

---
### 2.5 Comparisons Pairs

This section outlines pairs of functions for specific comparisons.

**Different Condition Numbers:**
- $F_1$: Shifted Sphere Function (implicitly, condition number 1)
- $F_2$: Shifted Schwefel's Problem 1.2 (non-separable, but not explicitly rotated by a controlled condition matrix in its basic form)
- $F_3$: Shifted Rotated High Conditioned Elliptic Function (explicit high condition number)

**Function With Noise Vs Without Noise:**
- **Pair 1:**
    - $F_2$: Shifted Schwefel's Problem 1.2
    - $F_4$: Shifted Schwefel's Problem 1.2 with Noise in Fitness
- **Pair 2:**
    - $F_{16}$: Rotated Hybrid Composition Function
    - $F_{17}$: $F_{16}$ with Noise in Fitness

**Function without Rotation Vs With Rotation:**
- **Pair 1:**
    - $F_9$: Shifted Rastrigin's Function (separable)
    - $F_{10}$: Shifted Rotated Rastrigin's Function (non-separable due to rotation)
- **Pair 2:**
    - $F_{15}$: Hybrid Composition Function (all $M_i$ are identity)
    - $F_{16}$: Rotated Hybrid Composition Function (all $M_i$ are rotation matrices)

**Continuous Vs Non-continuous:**
- $F_{21}$: Rotated Hybrid Composition Function
- $F_{23}$: Non-Continuous Rotated Hybrid Composition Function

**Global Optimum on Bounds Vs Not Necessarily on Bounds:**
(The PDF titles this "Global Optimum on Bounds Vs Global Optimum on Bounds", which is likely a typo. Should be "Global Optimum on Bounds Vs Global Optimum *Not* on Bounds" or similar contrast).
- $F_{18}$: Rotated Hybrid Composition Function (global optimum $o_1$ typically not on bound unless data makes it so)
- $F_{20}$: Rotated Hybrid Composition Function with the Global Optimum on the Bounds

**Wide Global Optimum Basin Vs Narrow Global Optimum Basin:**
- $F_{18}$: Rotated Hybrid Composition Function
- $F_{19}$: Rotated Hybrid Composition Function with a Narrow Basin for the Global Optimum

**Orthogonal Matrix Vs High Condition Number Matrix:**
(Applied to component functions in a hybrid composition)
- $F_{21}$: Rotated Hybrid Composition Function (component $M_i$ are orthogonal)
- $F_{22}$: Rotated Hybrid Composition Function with High Condition Number Matrix (component $M_i$ have high condition numbers)

**Global Optimum in the Initialization Range Vs outside of the Initialization Range:**
- $F_{24}$: Rotated Hybrid Composition Function (global optimum expected within typical $[-5,5]$ search range, initialization likely also covers this)
- $F_{25}$: Rotated Hybrid Composition Function without Bounds (global optimum at $o_1$ (likely near origin), but initialization is $[2,5]^D$)

---
### 2.6 Similar Groups:

**Unimodal Functions:**
- Functions $F_1 - F_5$

**Multi-modal Functions:**
- Functions $F_6 - F_{25}$
    - **Single Functions (Basic Multimodal):** Functions $F_6 - F_{12}$
    - **Expanded Functions:** Functions $F_{13} - F_{14}$
    - **Hybrid Composition Functions:** Functions $F_{15} - F_{25}$

**Functions with Global Optimum outside of the Initialization Range:**
- $F_7$: Shifted Rotated Griewank's Function without Bounds
- $F_{25}$: Rotated Hybrid Composition Function without Bounds

**Functions with Global Optimum on Bounds:**
- $F_5$: Schwefel's Problem 2.6 with Global Optimum on Bounds
- $F_8$: Shifted Rotated Ackley's Function with Global Optimum on Bounds
- $F_{20}$: Rotated Hybrid Composition Function with the Global Optimum on the Bounds

---
## 3. Evaluation Criteria

### 3.1 Description of the Evaluation Criteria

-   **Problems:** 25 minimization problems
-   **Dimensions:** $D = 10, 30, 50$
-   **Runs/problem:** 25 (Do not run many 25 runs to pick the best run)
-   **Max_FES (Maximum Function Evaluations):** $10000 \times D$
    -   For $D=10$: Max_FES = 100,000
    -   For $D=30$: Max_FES = 300,000
    -   For $D=50$: Max_FES = 500,000
-   **Initialization:** Uniform random initialization within the defined search space for each problem, except for problems $F_7$ and $F_{25}$, for which specific initialization ranges are specified.
    -   It is recommended to use the same set of initial populations for fair comparison across different algorithms and for the comparison pairs (problems 1, 2, 3 & 4; problems 9 & 10; problems 15, 16 & 17; problems 18, 19 & 20; problems 21, 22 & 23; problems 24 & 25). One way to achieve this is by using a fixed seed for the random number generator for generating these initial populations.
-   **Global Optimum:** All problems, except $F_7$ and $F_{25}$, have their global optimum $x^*$ located within their specified search bounds. There is no need to search outside these bounds for these problems. $F_7$ and $F_{25}$ are exceptions as they do not have predefined search bounds, and their global optima are outside their specified initialization ranges.
-   **Termination Error (Ter_Err):** $10^{-8}$.
-   **Termination Condition:** Terminate an algorithm run before reaching Max_FES if the error in function value, $f(x) - f(x^*)$, is less than or equal to Ter_Err ($10^{-8}$).

**1) Recording Function Error Values:**
-   For each run, record the function error value $(f(x) - f(x^*))$ at specified checkpoints: $1 \times 10^3$, $1 \times 10^4$, $1 \times 10^5$ FES, and also at the point of termination (either due to meeting Ter_Err or reaching Max_FES).
-   For each function and dimension, sort the recorded error values from the 25 runs in ascending order (best to worst).
-   Present the $1^{st}$ (best), $7^{th}$, $13^{th}$ (median), $19^{th}$, and $25^{th}$ (worst) error values.
-   Also, present the Mean and Standard Deviation (STD) of the error values over the 25 runs.

**2) Recording FES for Fixed Accuracy:**
-   For each run, record the number of FES required to reach a predefined fixed accuracy level (target error value). These accuracy levels are specified for each function (see Table 3-1 below).
-   The Max_FES limit applies; if the accuracy is not met within Max_FES, it's considered not achieved for that run in terms of FES count for this metric.

    **Table 3-1: Fixed Accuracy Level (Target Error $f(x) - f(x^*)$) for Each Function**

    | Function | Target Error Value $f(x)-f(x^*)$ | Function | Target Error Value $f(x)-f(x^*)$ |
    |----------|--------------------------------|----------|--------------------------------|
    | $F_1$    | $1 \times 10^{-6}$             | $F_{14}$ | $1 \times 10^{-2}$             |
    | $F_2$    | $1 \times 10^{-6}$             | $F_{15}$ | $1 \times 10^{-2}$             |
    | $F_3$    | $1 \times 10^{-6}$             | $F_{16}$ | $1 \times 10^{-2}$             |
    | $F_4$    | $1 \times 10^{-6}$             | $F_{17}$ | $1 \times 10^{-1}$             |
    | $F_5$    | $1 \times 10^{-6}$             | $F_{18}$ | $1 \times 10^{-1}$             |
    | $F_6$    | $1 \times 10^{-2}$             | $F_{19}$ | $1 \times 10^{-1}$             |
    | $F_7$    | $1 \times 10^{-2}$             | $F_{20}$ | $1 \times 10^{-1}$             |
    | $F_8$    | $1 \times 10^{-2}$             | $F_{21}$ | $1 \times 10^{-1}$             |
    | $F_9$    | $1 \times 10^{-2}$             | $F_{22}$ | $1 \times 10^{-1}$             |
    | $F_{10}$ | $1 \times 10^{-2}$             | $F_{23}$ | $1 \times 10^{-1}$             |
    | $F_{11}$ | $1 \times 10^{-2}$             | $F_{24}$ | $1 \times 10^{-1}$             |
    | $F_{12}$ | $1 \times 10^{-2}$             | $F_{25}$ | $1 \times 10^{-1}$             |
    | $F_{13}$ | $1 \times 10^{-2}$             |          |                                |
    *(Note: The PDF shows $f(x^*)$ values + accuracy, e.g., $-450 + 1e-6$. This table reflects the target error $f(x)-f(x^*)$ directly.)*

    -   **Successful Run:** A run is considered successful if the algorithm achieves the specified fixed accuracy level within the Max_FES for that dimension.
    -   For each function and dimension, sort the FES values from the 25 runs (only for successful runs if not all are successful, or note Max_FES if not achieved) in ascending order.
    -   Present the $1^{st}$ (best), $7^{th}$, $13^{th}$ (median), $19^{th}$, and $25^{th}$ (worst) FES values.
    -   Also, present the Mean and STD of the FES values over the 25 runs (or over successful runs, with clarification).

**3) Success Rate & Success Performance:**
-   For each problem (function/dimension):
-   **Success Rate** = (Number of successful runs) / (Total number of runs, i.e., 25)
-   **Success Performance** = Mean(FES for successful runs) $\times$ (Total runs) / (Number of successful runs). If no runs are successful, this metric is typically reported as N/A or $\infty$. (This simplifies to Mean(FES for successful runs) if there's at least one successful run).

**4) Convergence Graphs (or Run-length distribution graphs):**
-   For each problem, provide convergence graphs for $D=30$.
-   The graph should show the median performance (e.g., $13^{th}$ best run's error value) of the total runs versus FES.
-   The semi-log graphs should plot $\log_{10}(f(x) - f(x^*))$ (y-axis) against FES (x-axis).
-   Termination occurs either by meeting Ter_Err or reaching Max_FES.

**5) Algorithm Complexity:**
-   **a) Baseline Time $T_0$:** Execute the following test program and record its computing time as $T_0$.
```matlab
for i=1:1000000
    x_test = 5.55; % cast to double if necessary
    x_test = x_test + x_test; 
    x_test = x_test / 2.0;
    x_test = x_test * x_test;
    x_test = sqrt(x_test);
    x_test = log(x_test);   % natural logarithm
    x_test = exp(x_test);
    y_test = x_test / x_test; % results in 1 or NaN
end
```
-   **b) Function Evaluation Time $T_1$:** Evaluate the computing time for $200,000$ evaluations of benchmark function $F_3$ (Shifted Rotated High Conditioned Elliptic Function) for a specific dimension D. Record this time as $T_1$.
-   **c) Algorithm Runtime $T_2$:** Measure the complete computing time for your algorithm running for $200,000$ evaluations on the same D-dimensional benchmark function $F_3$. Execute this step 5 times and record the five $T_2$ values. Calculate $\overline{T2} = \text{Mean}(T_2 \text{ values})$.
-   This step is done to accommodate variations in execution time, especially for adaptive algorithms.
-   **Complexity Metrics:** The complexity of the algorithm is reflected by $\overline{T2}$, $T_1$, $T_0$, and the ratio $(\overline{T2} - T_1) / T_0$.
-   Calculate these complexity metrics for $D=10, 30,$ and $50$ to show the relationship between algorithm complexity and dimension.
-   Provide sufficient details on the computing system (CPU, RAM, OS) and the programming language/version used.

**6) Parameters:**
-   Discourage searching for a distinct set of parameters for each problem/dimension. Aim for a robust set of parameters.
-   Provide details on the following whenever applicable:
-   a) All parameters that need to be adjusted by the user.
-   b) Corresponding dynamic ranges for these parameters.
-   c) Guidelines on how to set or adjust these parameters.
-   d) Estimated cost of parameter tuning (e.g., in terms of FES).
-   e) Actual parameter values used for the experiments.

**7) Encoding:**
-   If the algorithm requires encoding of solutions (e.g., for genetic algorithms operating on binary strings for continuous parameters), the encoding scheme should be independent of the specific problems and governed by generic factors such as the search ranges. Describe the encoding scheme used.

### 3.2 Example (Illustrative - data will vary based on actual algorithm and runs)

System: Windows XP (SP1)
CPU: Pentium(R) 4 3.00GHz
RAM: 1 GB
Language: Matlab 6.5
Algorithm: Particle Swarm Optimizer (PSO)

**Results for $D=10$ (Max_FES = 100,000)**

**Table 3-2: Error Values Achieved When $FES=1 \times 10^3, 1 \times 10^4, 1 \times 10^5$ for Problems 1-8**
*(The PDF contains extensive tables here. I will represent the structure. "T" indicates termination error was met before the FES count.)*

| FES      | Statistic     | Prob 1      | Prob 2      | Prob 3      | Prob 4      | Prob 5      | Prob 6 | Prob 7 | Prob 8 |
|----------|---------------|-------------|-------------|-------------|-------------|-------------|--------|--------|--------|
| $10^3$   | $1^{st}$ (Best) | 4.8672e+2   | 2.2037e+6   | 4.7296e+2   | 4.6617e+2   | 2.3522e+3   | ...    | ...    | ...    |
|          | $7^{th}$        | 8.0293e+2   | 8.5141e+6   | 9.8091e+2   | 1.2900e+3   | 4.0573e+3   | ...    | ...    | ...    |
|          | Median        | 9.2384e+2   | 1.4311e+7   | 1.5293e+3   | 1.9769e+3   | 4.6308e+3   | ...    | ...    | ...    |
|          | $19^{th}$       | 1.3393e+3   | 1.9298e+7   | 1.7615e+3   | 2.9175e+3   | 4.8015e+3   | ...    | ...    | ...    |
|          | $25^{th}$ (Worst)| 1.9151e+3   | 4.4688e+7   | 3.2337e+3   | 6.5038e+3   | 5.6701e+3   | ...    | ...    | ...    |
|          | Mean          | 1.0996e+3   | 1.5156e+7   | 1.5107e+3   | 2.3669e+3   | 4.4857e+3   | ...    | ...    | ...    |
|          | Std           | 4.0575e+2   | 9.3002e+6   | 7.2503e+2   | 1.5082e+3   | 7.0081e+2   | ...    | ...    | ...    |
| $10^4$   | $1^{st}$ (Best) | 3.1984e-3   | 1.3491e+5   | 1.0413e+0   | 6.7175e+0   | 1.6584e+3   | ...    | ...    | ...    |
|          | ...           | ...         | ...         | ...         | ...         | ...         | ...    | ...    | ...    |
| $10^5$   | $1^{st}$ (Best) | 4.7434e-9T  | 4.2175e+4   | 5.1782e-9T  | 1.7070e-5   | 1.1864e+3   | ...    | ...    | ...    |
|          | ...           | ...         | ...         | ...         | ...         | ...         | ...    | ...    | ...    |

*(Similar tables would follow for problems 9-17 (Table 3-3) and 18-25 (Table 3-4) for $D=10$. Then similar sets of tables for $D=30$ and $D=50$.)*

**Table 3-5: Number of FES to achieve the fixed accuracy level ($D=10$)**

| Prob | $1^{st}$ (Best) FES | $7^{th}$ FES | Median FES | $19^{th}$ FES | $25^{th}$ (Worst) FES | Mean FES    | Std FES     | Success Rate | Success Performance |
|------|-----------------|------------|------------|-------------|-------------------|-------------|-------------|--------------|---------------------|
| $F_1$  | 11607           | 12133      | 12372      | 12704       | 13022             | 1.2373e+4   | 3.6607e+2   | 100%         | 1.2373e+4           |
| $F_2$  | 17042           | 17608      | 18039      | 18753       | 19671             | 1.8163e+4   | 7.5123e+2   | 100%         | 1.8163e+4           |
| $F_3$  | N/A             | N/A        | N/A        | N/A         | N/A               | N/A         | N/A         | 0%           | N/A                 |
| ...  | ...             | ...        | ...        | ...         | ...               | ...         | ...         | ...          | ...                 |

*(This table would be presented for all 25 functions for $D=10$, then repeated for $D=30$ and $D=50$.)*

**Convergence Graphs (30D):**
*(The PDF shows sample graph structures. Actual graphs would be generated from experimental data.)*

*Figure 3-1: Convergence Graph for Functions 1-5 (Median error vs FES for D=30)*
*(Plot $\log_{10}(f(x)-f(x^*))$ vs FES)*

*(Similarly, Figure 3-2 for F6-10, Figure 3-3 for F11-14, Figure 3-4 for F15-20, Figure 3-5 for F21-25)*

**Algorithm Complexity:**

**Table 3-8: Computational Complexity**
*(Example data from PDF)*

| Dimension | $T_0$ (s) | $T_1$ (s) | $\overline{T2}$ (s) | $(\overline{T2}-T_1)/T_0$ |
|-----------|-----------|-----------|-----------------|-------------------------|
| $D=10$    | (value)   | 31.1250   | 82.3906         | 1.2963                  |
| $D=30$    | (value)   | 39.5470   | 90.8437         | 1.3331                  |
| $D=50$    | (value)   | 46.0780   | 108.9094        | 1.5888                  |
*(Note: $T_0$ is a single baseline value, not dimension-dependent in the table shown in PDF, but it should be measured on the specific test system.)*

**Parameters:**
*(This section would detail the parameters of the specific algorithm used, e.g., PSO's inertia weight, acceleration coefficients, population size, etc., as per guidelines in 3.1.6.)*
- a) All parameters to be adjusted: ...
- b) Corresponding dynamic ranges: ...
- c) Guidelines on how to adjust the parameters: ...
- d) Estimated cost of parameter tuning in terms of number of FES: ...
- e) Actual parameter values used: ...

---
## 4. Notes

**Note 1: Linear Transformation Matrix**
A linear transformation matrix $M$ is constructed as $M = P \cdot N \cdot Q$.
-   P, Q are two orthogonal matrices, generated using the Classical Gram-Schmidt method.
-   N is a diagonal matrix where diagonal elements $d_{ii}$ control the condition number. For example, $d_{ii} = c^{\frac{i-1}{D-1}}$, where $u = \text{rand}(1,D)$ and $c = \text{Cond}(M)$ is the desired condition number. (The PDF mentions $d_{ii} = c \frac{u_i - \min(u)}{\max(u) - \min(u)}$ which seems to scale random values, but then states $M$'s condition number Cond(M)=c. The power form $c^{\frac{i-1}{D-1}}$ is more standard for achieving a specific condition $c$ across scales).

**Note 2: Weight Adjustment in Composition Functions**
On page 18 (of PDF), the weight values $w_i$ are adjusted (e.g., $w_i^* = w_i \cdot (1 - \max(w_j)^{10})$ if $w_i$ is not the maximum weight). The objective is to ensure that each optimum (local or global) is primarily influenced by only one basic function in its immediate vicinity, while allowing a higher degree of mixing of different functions further away from the optima.

**Note 3: Objective Function Values**
We assign different positive and negative objective function values (biases $f_{\text{bias}_i}$) instead of always having $f(x^*)=0$. This may influence some algorithms that make use of the absolute objective values.

**Note 4: Comparison Pairs Objective Values**
We assign the same $f_{\text{bias}}$ values to the functions within comparison pairs (e.g., $F_{15}$ and $F_{16}$ both have $f_{\text{bias}} = 120$) to make their comparison easier in terms of target values.

**Note 5: High Condition Number Rotation**
High condition number rotation can sometimes transform a multimodal problem into a unimodal one (or one with a much more pronounced global basin). Hence, moderate condition numbers were generally used for the rotation matrices applied to inherently multimodal basic functions.

**Note 6: Additional Data Files for Verification**
Additional data files are provided with some coordinate positions and their corresponding fitness values. These are intended to help in the verification process during the translation and implementation of the benchmark functions.

**Note 7: Statistical Significance of Pairs**
It is insufficient to make any statistically meaningful conclusions solely based on the comparison pairs presented, as each case has at most 2 pairs. A more rigorous study would likely require 5, 10, or more pairs for each specific comparative aspect. This extension might be considered for future work or an edited volume.

**Note 8: Pseudo-Real World Problems**
Pseudo-real world problems are available from the web link: [http://www.cs.colostate.edu/~genitor/functions.html](http://www.cs.colostate.edu/~genitor/functions.html).
If you have any queries on these problems, please contact Professor Darrell Whitley directly. Email: whitley@CS.ColoState.EDU

**Note 9: Data Recording for Statistical Tests**
We are recording detailed performance data, such as 'the number of FES to reach the given fixed accuracy' and 'the objective function value at different FES counts' for each run, each problem, and each dimension. This is done to facilitate performing statistical significance tests on the results. The details of a suitable statistical significance test would be made available at a later date.

---
## References:

1.  N. Hansen, S. D. Muller and P. Koumoutsakos, "Reducing the Time Complexity of the Derandomized Evolution Strategy with Covariance Matrix Adaptation (CMA-ES)." *Evolutionary Computation*, 11(1), pp. 1-18, 2003.
2.  A. Klimke, "Weierstrass function's matlab code", <http://matlabdb.mathematik.uni-stuttgart.de/download.jsp?MC_ID=9&MP_ID=56>
3.  H-P. Schwefel, "Evolution and Optimum Seeking", <http://ls11-www.cs.uni-dortmund.de/lehre/wiley/> (Link might be outdated or specific to a course).
4.  D. Whitley, K. Mathias, S. Rana and J. Dzubera, "Evaluating Evolutionary Algorithms." *Artificial Intelligence*, 85 (1-2): 245-276, AUG 1996.