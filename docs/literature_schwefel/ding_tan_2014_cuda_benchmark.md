## A CUDA-Based Real Parameter Optimization Benchmark

Ke Ding and Ying Tan

School of Electronics Engineering and Computer Science, Peking University

Abstract. Benchmarking is key for developing and comparing optimization algorithms. In this paper, a CUDA-based real parameter optimization benchmark (cuROB) is introduced. Test functions of diverse properties are included within cuROB and implemented efficiently with CUDA. Speedup of one order of magnitude can be achieved in comparison with CPU-based benchmark of CEC'14.

Key words: Optimization Methods, Optimization Benchmark, GPU, CUDA.

## 1 Introduction

Proposed algorithms are usually tested on benchmark for comparing both performance and efficiency. However, as it can be a very tedious task to select and implement test functions rigorously. Thanks to GPUs' massive parallelism, a GPU-based optimization function suit will be beneficial to test and compare optimization algorithms.

Based on the well known CPU-based benchmarks presented in [1-3], we proposed a CUDA-based real parameter optimization test suit, called cuROB, targeting on GPUs. We think cuROB can be helpful for assessing GPU-based optimization algorithms, and hopefully, conventional CPU-based algorithms can benefit from cuROB's fast execution.

Considering the fact that research on the single objective optimization algorithms is the basis of the research on the more complex optimization algorithms such as constrained optimization algorithms, multi-objective optimizations algorithms and so forth, in this first release of cuROB a suit of single objective real-parameter optimization function are defined and implemented.

The test functions are selected according to the following criteria: 1) the functions should be scalable in dimension so that algorithms can be tested under various complexity; 2) the expressions of the functions should be with good parallelism, thus efficient implementation is possible on GPUs; 3) the functions should be comprehensible such that algorithm behaviours can be analysed in the topological context; 4) last but most important, the test suit should cover functions of various properties in order to get a systematic evaluation of the optimization algorithms.

The source code and a sample can be download from code.google.com/p/curob/.

## 1.1 Symbol Conventions and Definitions

Symbols and definitions used in the report are described in the following. By default, all vectors refer to column vectors, and are depicted by lowercase letter and typeset in bold.

- -[ · ] indicates the nearest integer value
- -glyph[floorleft]·glyph[floorright] indicates the largest integer less than or equal to
- x i denotes i -th element of vector x
- -f ( · ), g ( · ) and G ( · ) multi-variable functions
- -f opt optimal (minimal) value of function f
- x opt optimal solution vector, such that f ( x opt ) = f opt
- R normalized orthogonal matrix for rotation
- -D dimension
- 1 = (1 , . . . , 1) T all one vector

## 1.2 General Setup

The general setup of the test suit is presented as follows.

- Dimensions The test suit is scalable in terms of dimension. Within the hardware limit, any dimension D ≥ 2 works. However, to construct a real hybrid function, D should be at least 10.
- Search Space All functions are defined and can be evaluated over R D , while the actual search domain is given as [ -100 , 100] D .
- -f opt All functions, by definition, have a minimal value of 0, a bias ( f opt ) can be added to each function. The selection can be arbitrary, f opt for each function in the test suit is listed in Tab. 1.
- x opt The optimum point of each function is located at original. x opt which is randomly distributed in [ -70 , 70] D , is selected as the new optimum.
- Rotation Matrix To derive non-separable functions from separable ones, the search space is rotated by a normalized orthogonal matrix R . For a given function in one dimension, a different R is used. Variables are divided into three (almost) equal-sized subcomponents randomly. The rotation matrix for each subcomponent is generated from standard normally distributed entries by Gram-Schmidt orthonormalization. Then, these matrices consist of the R actually used.

## 1.3 CUDA Interface and Implementation

Asimple description of the interface and implementation is given in the following. For detail, see the source code and the accompanied readme file.

Interface Only benchmark.h need to be included to access the test functions, and the CUDA file benchmark.cu need be compiled and linked. Before the compiling start, two macro, DIM and MAX CONCURRENCY should be modified accordingly. DIM defines the dimension of the test suit to used while MAX CONCURRENCY controls the most function evaluations be invoked concurrently. As memory needed to be pre-allocated, limited by the hardware, don't set MAX CONCURRENCY greater than actually used.

Host interface function initialize () accomplish all initialization tasks, so must be called before any test function can be evaluated. Allocated resource is released by host interface function dispose ().

Both double precision and single precision are supported through func evaluate () and func evaluatef () respectively. Take note that device pointers should be passed to these two functions. For the convenience of CPU code, C interfaces are provided, with h func evaluate for double precision and h func evaluatef for single precision. (In fact, they are just wrappers of the GPU interfaces.)

Efficiency Concerns When configuration of the suit, some should be taken care for the sake of efficiency. It is better to evaluation a batch of vectors than many smaller. Dimension is a fold of 32 (the warp size) can more efficient. For example, dimension of 96 is much more efficient than 100, even though 100 is little greater than 96.

## 1.4 Test Suite Summary

The test functions fall into four categories: unimodal functions, basic multimodal functions, hybrid functions and composition functions. The summary of the suit is listed in Tab. 1. Detailed information of each function will given in the following sections.

## 2 Speedup

Under different hardware, various speedups can be achieved. 30 functions are the same as CEC'14 benchmark. We test the cuROB's speedup with these 30 functions under the following settings: Windows 7 SP1 x64 running on Intel i5-2310 CPU with NVIDIA 560 Ti, the CUDA version is 5.5. 50 evaluations were performed concurrently and repeated 1000 runs. The evaluation data were generated randomly from uniform distribution.

The speedups with respect to different dimension are listed by Tab. 2 (single precision) and Tab. 3 (double precision). Notice that the corresponding dimensions of cuROB are 10, 32, 64 and 96 respectively and the numbers are as in Tab. 1

Fig. 1 demonstrates the overall speedup for each dimension. On average, cuROB is never slower than its CPU-base CEC'14 benchmark, and speedup of one order of magnitude can be achieved when dimension is high. Single precision is more efficient than double precision as far as execution time is concerned.

Table 1. Summary of cuROB's Test Functions

|   No. | Functions                                 | ID           | Description                                                           |
|-------|-------------------------------------------|--------------|-----------------------------------------------------------------------|
|     0 | Rotated Sphere                            | SPHERE       | Optimum easy to track                                                 |
|     1 | Rotated Ellipsoid                         | ELLIPSOID    | Optimum easy to track                                                 |
|     2 | Rotated Elliptic                          | ELLIPTIC     | Optimum hard to track                                                 |
|     3 | Rotated Discus                            | DISCUS       | Optimum hard to track                                                 |
|     4 | Rotated Bent Cigar                        | CIGAR        | Optimum hard to track                                                 |
|     5 | Rotated Different Powers                  | POWERS       | Optimum hard to track                                                 |
|     6 | Rotated Sharp Valley                      | SHARPV       | Optimum hard to track                                                 |
|     7 | Rotated Step                              | STEP         | With adepuate global structure                                        |
|     8 | Rotated Weierstrass                       | WEIERSTRASS  | With adepuate global structure                                        |
|     9 | Rotated Griewank                          | GRIEWANK     | With adepuate global structure                                        |
|    10 | Rastrigin                                 | RARSTRIGIN U | With adepuate global structure                                        |
|    11 | Rotated Rastrigin                         | RARSTRIGIN   | With adepuate global structure                                        |
|    12 | Rotated Schaffer's F7                     | SCHAFFERSF7  | With adepuate global structure                                        |
|    13 | Rotated Expanded Griewank plus Rosenbrock | GRIE ROSEN   | With adepuate global structure                                        |
|    14 | Rotated Rosenbrock                        | ROSENBROCK   | With weak global                                                      |
|    15 | Modified Schwefel                         | SCHWEFEL U   | With weak global                                                      |
|    16 | Rotated Modified Schwefel                 | SCHWEFEL     | With weak global                                                      |
|    17 | Rotated Katsuura                          | KATSUURA     | With weak global                                                      |
|    18 | Rotated Lunacek bi-Rastrigin              | LUNACEK      | With weak global                                                      |
|    19 | Rotated Ackley                            | ACKLEY       | structure                                                             |
|    20 | Rotated HappyCat                          | HAPPYCAT     | With weak global                                                      |
|    21 | Rotated HGBat                             | HGBAT        | With weak global                                                      |
|    22 | Rotated Expanded Schaffer's F6            | SCHAFFERSF6  | With weak global                                                      |
|    23 | Hybrid Function 1                         | HYBRID1      | With different properties for different variables subcomponents       |
|    24 | Hybrid Function 2                         | HYBRID2      | With different properties for different variables subcomponents       |
|    25 | Hybrid Function 3                         | HYBRID3      | With different properties for different variables subcomponents       |
|    26 | Hybrid Function 4                         | HYBRID4      | With different properties for different variables subcomponents       |
|    27 | Hybrid Function 5                         | HYBRID5      | With different properties for different variables subcomponents       |
|    28 | Hybrid Function 6                         | HYBRID6      | With different properties for different variables subcomponents       |
|    29 | Composition Function 1                    | COMPOSITION1 | to particular sub-function when approaching the corresponding optimum |
|    30 | Composition Function 2                    | COMPOSITION2 | to particular sub-function when approaching the corresponding optimum |
|    31 | Composition Function 3                    | COMPOSITION3 | to particular sub-function when approaching the corresponding optimum |
|    32 | Composition Function 4                    | COMPOSITION4 | to particular sub-function when approaching the corresponding optimum |
|    33 | Composition Function 5                    | COMPOSITION5 | to particular sub-function when approaching the corresponding optimum |
|    34 | Composition Function 6                    | COMPOSITION6 | to particular sub-function when approaching the corresponding optimum |
|    35 | Composition Function 7                    | COMPOSITION7 | to particular sub-function when approaching the corresponding optimum |
|    36 | Composition Function 8                    | COMPOSITION8 | to particular sub-function when approaching the corresponding optimum |

Table 2. Speedup (single Precision)

| D                                                             | NO.3                                                          | NO.4                                                          | NO.5                                                          | NO.8                                                          | NO.9                                                          | NO.10                                                         | NO.11                                                         | NO.13                                                         | NO.14                                                         | NO.15                                                         |
|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|
| 10                                                            | 0.59                                                          | 0.20                                                          | 0.18                                                          | 12.23                                                         | 0.49                                                          | 0.28                                                          | 0.31                                                          | 0.32                                                          | 0.14                                                          | 0.77                                                          |
| 32                                                            | 3.82                                                          | 2.42                                                          | 2.00                                                          | 47.19                                                         | 3.54                                                          | 1.67                                                          | 3.83                                                          | 5.09                                                          | 2.06                                                          | 3.54                                                          |
| 64                                                            | 4.67                                                          | 2.72                                                          | 2.29                                                          | 50.17                                                         | 3.56                                                          | 0.93                                                          | 3.06                                                          | 2.88                                                          | 2.20                                                          | 3.39                                                          |
| 94                                                            | 13.40                                                         | 10.10                                                         | 8.50                                                          | 84.31                                                         | 11.13                                                         | 1.82                                                          | 9.98                                                          | 9.66                                                          | 8.75                                                          | 6.73                                                          |
| D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 |
| 10                                                            | 0.80                                                          | 3.25                                                          | 0.36                                                          | 0.20                                                          | 0.26                                                          | 0.45                                                          | 0.63                                                          | 0.44                                                          | 2.80                                                          | 0.52                                                          |
| 32                                                            | 5.57                                                          | 10.04                                                         | 3.46                                                          | 1.22                                                          | 1.42                                                          | 6.44                                                          | 3.95                                                          | 3.43                                                          | 11.47                                                         | 3.36                                                          |
| 64                                                            | 5.45                                                          | 13.19                                                         | 3.27                                                          | 2.10                                                          | 2.27                                                          | 3.81                                                          | 4.62                                                          | 3.07                                                          | 14.17                                                         | 3.34                                                          |
| 96                                                            | 14.38                                                         | 23.68                                                         | 11.32                                                         | 8.26                                                          | 8.49                                                          | 11.60                                                         | 13.67                                                         | 10.64                                                         | 30.11                                                         | 10.71                                                         |
| D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 |
| 10                                                            | 0.65                                                          | 0.72                                                          | 0.70                                                          | 0.55                                                          | 0.71                                                          | 3.49                                                          | 3.50                                                          | 0.84                                                          | 1.28                                                          | 0.70                                                          |
| 32                                                            | 2.73                                                          | 3.09                                                          | 3.63                                                          | 3.10                                                          | 4.10                                                          | 12.39                                                         | 12.51                                                         | 5.25                                                          | 5.19                                                          | 3.33                                                          |
| 64                                                            | 3.86                                                          | 4.01                                                          | 3.21                                                          | 2.67                                                          | 3.38                                                          | 12.68                                                         | 12.63                                                         | 3.80                                                          | 5.27                                                          | 3.13                                                          |
| 96                                                            | 12.04                                                         | 11.32                                                         | 8.15                                                          | 6.27                                                          | 8.49                                                          | 23.67                                                         | 23.64                                                         | 9.50                                                          | 11.79                                                         | 7.93                                                          |

Table 3. Speedup (Double Precision)

| D                                                             | NO.3                                                          | NO.4                                                          | NO.5                                                          | NO.8                                                          | NO.9                                                          | NO.10                                                         | NO.11                                                         | NO.13                                                         | NO.14                                                         | NO.15                                                         |
|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|
| 10                                                            | 0.56                                                          | 0.19                                                          | 0.17                                                          | 9.04                                                          | 0.43                                                          | 0.26                                                          | 0.29                                                          | 0.30                                                          | 0.14                                                          | 0.75                                                          |
| 32                                                            | 3.78                                                          | 2.43                                                          | 1.80                                                          | 33.37                                                         | 3.09                                                          | 1.59                                                          | 3.52                                                          | 4.81                                                          | 1.97                                                          | 3.53                                                          |
| 64                                                            | 4.34                                                          | 2.49                                                          | 1.93                                                          | 30.82                                                         | 3.15                                                          | 0.92                                                          | 2.87                                                          | 2.74                                                          | 2.11                                                          | 3.29                                                          |
| 96                                                            | 12.27                                                         | 9.24                                                          | 6.95                                                          | 46.01                                                         | 9.72                                                          | 1.78                                                          | 9.62                                                          | 8.74                                                          | 7.87                                                          | 5.92                                                          |
| D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 | D NO.16 NO.17 NO.19 NO.20 NO.21 NO.22 NO.23 NO.24 NO.25 NO.26 |
| 10                                                            | 0.79                                                          | 2.32                                                          | 0.34                                                          | 0.18                                                          | 0.26                                                          | 0.45                                                          | 0.59                                                          | 0.43                                                          | 1.97                                                          | 0.52                                                          |
| 32                                                            | 5.10                                                          | 6.79                                                          | 3.28                                                          | 1.13                                                          | 1.29                                                          | 6.10                                                          | 3.63                                                          | 3.14                                                          | 8.15                                                          | 3.23                                                          |
| 64                                                            | 4.75                                                          | 8.29                                                          | 3.06                                                          | 1.99                                                          | 2.18                                                          | 3.32                                                          | 4.02                                                          | 2.77                                                          | 9.80                                                          | 2.92                                                          |
| 96                                                            | 11.91                                                         | 13.81                                                         | 9.75                                                          | 7.37                                                          | 7.78                                                          | 10.24                                                         | 11.55                                                         | 9.57                                                          | 20.81                                                         | 9.40                                                          |
| D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 | D NO.27 NO.28 NO.29 NO.30 NO.31 NO.32 NO.33 NO.34 NO.35 NO.36 |
| 10                                                            | 0.79                                                          | 2.32                                                          | 0.34                                                          | 0.18                                                          | 0.26                                                          | 0.45                                                          | 0.59                                                          | 0.43                                                          | 1.97                                                          | 0.52                                                          |
| 32                                                            | 5.10                                                          | 6.79                                                          | 3.28                                                          | 1.13                                                          | 1.29                                                          | 6.10                                                          | 3.63                                                          | 3.14                                                          | 8.15                                                          | 3.23                                                          |
| 64                                                            | 4.75                                                          | 8.29                                                          | 3.06                                                          | 1.99                                                          | 2.18                                                          | 3.32                                                          | 4.02                                                          | 2.77                                                          | 9.80                                                          | 2.92                                                          |
| 96                                                            | 11.91                                                         | 13.81                                                         | 9.75                                                          | 7.37                                                          | 7.78                                                          | 10.24                                                         | 11.55                                                         | 9.57                                                          | 20.81                                                         | 9.40                                                          |

Fig. 1. Overall Speedup

<!-- image -->

## 3 Unimodal Functions

## 3.1 Shifted and Rotated Sphere Function

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Properties

- -Unimodal
- -Non-separable
- -Highly symmetric, in particular rotationally invariant

## 3.2 Shifted and Rotated Ellipsoid Function

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Properties

- -Unimodal
- -Non-separable

## 3.3 Shifted and Rotated High Conditioned Elliptic Function

<!-- formula-not-decoded -->

where z = R ( x -x opt ).

## Properties

- -Unimodal
- -Non-separable
- -Quadratic ill-conditioned
- -Smooth local irregularities

## 3.4 Shifted and Rotated Discus Function

<!-- formula-not-decoded -->

where z = R ( x -x opt ).

## Properties

- -Unimodal
- -Non-separable
- -Smooth local irregularities
- -With One sensitive direction

## 3.5 Shifted and Rotated Bent Cigar Function

<!-- formula-not-decoded -->

where z = R ( x -x opt ).

## Properties

- -Unimodal
- -Non-separable
- -Optimum located in a smooth but very narrow valley

## 3.6 Shifted and Rotated Different Powers Function

<!-- formula-not-decoded -->

where z = R (0 . 01( x -x opt )).

## Properties

- -Unimodal
- -Non-separable
- -Sensitivities of the z i -variables are different

## 3.7 Shifted and Rotated Sharp Valley Function

<!-- formula-not-decoded -->

where z = R ( x -x opt ).

## Properties

- -Unimodal
- -Non-separable
- -Global optimum located in a sharp (non-differentiable) ridge

## 4 Basic Multi-modal Functions

## 4.1 Shifted and Rotated Step Function

<!-- formula-not-decoded -->

where z = R ( x -x opt )

## Properties

- -Many Plateaus of different sizes
- -Non-separable

## 4.2 Shifted and Rotated Weierstrass Function

<!-- formula-not-decoded -->

where a = 0 . 5, b = 3, k max = 20, z = R (0 . 005 · ( x -x opt )).

## Properties

- -Multi-modal
- -Non-separable
- -Continuous everywhere but only differentiable on a set of points

## 4.3 Shifted and Rotated Griewank Function

<!-- formula-not-decoded -->

where z = R (6 · ( x -x opt )).

## Properties

- -Multi-modal
- -Non-separable
- -With many regularly distributed local optima

## 4.4 Shifted Rastrigin Function

<!-- formula-not-decoded -->

where z = 0 . 0512 · ( x -x opt ).

## Properties

- -Multi-modal
- -Separable
- -With many regularly distributed local optima

## 4.5 Shifted and Rotated Rastrigin Function

<!-- formula-not-decoded -->

where z = R (0 . 0512 · ( x -x opt )).

## Properties

- -Multi-modal
- -Non-separable
- -With many regularly distributed local optima

## 4.6 Shifted Rotated Schaffer's F7 Function

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Properties

- -Multi-modal
- -Non-separable

## 4.7 Expanded Griewank plus Rosenbrock Function

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where z = R (0 . 05 · ( x -x opt )) + 1 .

## Properties

- -Multi-modal
- -Non-separable

-

## 4.8 Shifted and Rotated Rosenbrock Function

<!-- formula-not-decoded -->

where z = R (0 . 02048 · ( x -x opt )) + 1 .

## Properties

- -Multi-modal
- -Non-separable
- -With a long, narrow, parabolic shaped flat valley from local optima to global optima

## 4.9 Shifted Modified Schwefel Function

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where z = 10 · ( x -x opt ).

## Properties

- -Multi-modal
- -Separable
- -Having many local optima with the second better local optima far from the global optima

## 4.10 Shifted Rotated Modified Schwefel Function

<!-- formula-not-decoded -->

where z = R (10 · ( x -x opt )) and g 1 ( · ) is defined as Eq. 17.

## Properties

- -Multi-modal
- -Non-separable
- -Having many local optima with the second better local optima far from the global optima

## 4.11 Shifted Rotated Katsuura Function

<!-- formula-not-decoded -->

where z = R (0 . 05 · ( x -x opt )).

## Properties

- -Multi-modal
- -Non-separable
- -Continuous everywhere but differentiable nowhere

## 4.12 Shifted and Rotated Lunacek bi-Rastrigin Function

<!-- formula-not-decoded -->

where z = R (0 . 1 · ( x -x opt ) + 2 . 5 ∗ 1 ), µ 1 = 2 . 5, µ 2 = -2 . 5, d = 1, s = 0 . 9.

## Properties

- -Multi-modal
- -Non-separable
- -With two funnel around µ 1 1 and µ 2 1

## 4.13 Shifted and Rotated Ackley Function

<!-- formula-not-decoded -->

where z = R ( x -x opt ).

## Properties

- -Multi-modal
- -Non-separable
- -Having many local optima with the global optima located in a very small basin

## 4.14 Shifted Rotated HappyCat Function

<!-- formula-not-decoded -->

where z = R (0 . 05 · ( x -x opt )) -1 .

## Properties

- -Multi-modal
- -Non-separable
- -Global optima located in curved narrow valley

## 4.15 Shifted Rotated HGBat Function

<!-- formula-not-decoded -->

where z = R (0 . 05 · ( x -x opt )) -1 .

## Properties

- -Multi-modal
- -Non-separable
- -Global optima located in curved narrow valley

## 4.16 Expanded Schaffer's F6 Function

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where z = R ( x -x opt ).

## Properties

- -Multi-modal
- -Non-separable

## 5 Hybrid Functions

Hybrid functions are constructed according to [3]. For each hybrid function, the variables are randomly divided into subcomponents and different basic functions (unimodal and multi-modal) are used for different subcomponents, as depicted by Eq. 25.

<!-- formula-not-decoded -->

where F ( · ) is the constructed hybrid function and G i ( · ) is the i -th basic function used, N is the number of basic functions. z i is constructed as follows.

<!-- formula-not-decoded -->

where S is a permutation of (1 : D ), such that z = [ z 1 , z 2 , . . . , z N ] forms the transformed vector and n i , i = 1 , . . . , N are the dimensions of the basic functions, which is derived as Eq. 26.

<!-- formula-not-decoded -->

p i is used to control the percentage of each basic functions.

## 5.1 Hybrid Function 1

- -N = 3
- -p = [0 . 3 , 0 . 3 , 0 . 4]
- -G 1 : Modified Schwefel's Function
- -G 2 : Rastrigin Function
- -G 3 : High Conditioned Elliptic Function

## 5.2 Hybrid Function 2

- -N = 5
- -p = [0 . 3 , 0 . 3 , 0 . 4]
- -G 1 : Bent Cigar Function
- -G 2 : HGBat Function
- -G 3 : Rastrigin Function

## 5.3 Hybrid Function 3

- -N = 4
- -p = [0 . 2 , 0 . 2 , 0 . 3 , 0 . 3]
- -G 1 : Griewank Function
- -G 2 : Weierstrass Function
- -G 3 : Rosenbrock Function
- -G 4 : Expanded Scaffer's F6 Function

## 5.4 Hybrid Function 4

- -N = 4
- -p = [0 . 2 , 0 . 2 , 0 . 3 , 0 . 3]
- -G 1 : HGBat Function
- -G 2 : Discus Function
- -G 3 : Expanded Griewank plus Rosenbrock Function
- -G 4 : Rastrigin Function

## 5.5 Hybrid Function 5

- -N = 5
- -p = [0 . 1 , 0 . 2 , 0 . 2 , 0 . 2 , 0 . 3]
- -G 1 : Expanded Scaffer's F6 Function
- -G 2 : HGBat Function
- -G 3 : Rosenbrock Function
- -G 4 : Modified Schwefel's Function
- -G 5 : High Conditioned Elliptic Function

## 5.6 Hybrid Function 6

- -N = 5
- -p = [0 . 1 , 0 . 2 , 0 . 2 , 0 . 2 , 0 . 3]
- -G 1 : Katsuura Function
- -G 2 : HappyCat Function
- -G 3 : Expanded Griewank plus Rosenbrock Function
- -G 4 : Modified Schwefel's Function
- -G 5 : Ackley Function

## 6 Composition Functions

Composition functions are constructed in the same manner as in [2, 3].

<!-- formula-not-decoded -->

- -F ( · ): the constructed composition function
- -G i ( · ): i -th basic function
- -N : number of basic functions used
- -bias i : define which optimum is the global optimum
- -σ i : control G i ( · )'s coverage range, a small σ i gives a narrow range for G i ( · )
- -λ i : control G i ( · )'s height
- -ω i : weighted value for G i ( · ), calculated as follows:

<!-- formula-not-decoded -->

where x opt,i represents the optimum position for G i ( · ). Then normalized w i to get ω i : ω i = w i / ∑ N i =1 w i .

glyph[negationslash]

<!-- formula-not-decoded -->

The constructed functions are multi-modal and non-separable and merge the properties of the sub-functions better and maintains continuity around the global/local optima. The local optimum which has the smallest bias value is the global optimum. The optimum of the third basic function is set to the origin as a trip in order to test the algorithms' tendency to converge to the search center.

Note that, the landscape is not only changes along with the selection of basic function, but the optima and σ and λ can effect it greatly.

## 6.1 Composition Function 1

- -N = 5
- -σ = [10 , 20 , 30 , 40 , 50]
- -λ = [1 e -10 , 1 e -6 , 1 e -26 , 1 e -6 , 1 e -6]
- -bias = [0 , 100 , 200 , 300 , 400]
- -G 1 : Rotated Rosenbrock Function
- -G 2 : High Conditioned Elliptic Function
- -G 3 : Rotated Bent Cigar Function
- -G 4 : Rotated Discus Function
- -G 5 : High Conditioned Elliptic Function

## 6.2 Composition Function 2

- -N = 3
- -σ = [15 , 15 , 15]
- -λ = [1 , 1 , 1]
- -bias = [0 , 100 , 200]
- -G 1 : Expanded Schwefel Function
- -G 2 : Rotated Rstrigin Function
- -G 3 : Rotated HGBat Function

## 6.3 Composition Function 3

- -N = 3
- -σ = [20 , 50 , 40]
- -λ = [0 . 25 , 1 , 1 e -7]
- -bias = [0 , 100 , 200]
- -G 1 : Rotated Schwefel Function
- -G 2 : Rotated Rastrigin Function
- -G 3 : Rotated High Conditioned Elliptic Function

## 6.4 Composition Function 4

- -N = 5
- -σ = [20 , 15 , 10 , 10 , 40]
- -λ = [2 . 5 e -2 , 0 . 1 , 1 e -8 , 0 . 25 , 1]
- -bias = [0 , 100 , 200 , 300 , 400]
- -G 1 : Rotated Schwefel Function
- -G 2 : Rotated HappyCat Function
- -G 3 : Rotated High Conditioned Elliptic Function
- -G 4 : Rotated Weierstrass Function
- -G 5 : Rotated Griewank Function

## 6.5 Composition Function 5

- -N = 5
- -σ = [15 , 15 , 15 , 15 , 15]
- -λ = [10 , 10 , 2 . 5 , 2 . 5 , 1 e -6]
- -bias = [0 , 100 , 200 , 300 , 400]
- -G 1 : Rotated HGBat Function
- -G 2 : Rotated Rastrigin Function
- -G 5 : Rotated Schwefel Function
- -G 4 : Rotated Weierstrass Function
- -G 3 : Rotated High Conditioned Elliptic Function

## 6.6 Composition Function 6

- -N = 5
- -σ = [10 , 20 , 30 , 40 , 50]
- -λ = [2 . 5 , 10 , 2 . 5 , 5 e -4 , 1 e -6]
- -bias = [0 , 100 , 200 , 300 , 400]
- -G 1 : Rotated Expanded Griewank plus Rosenbrock Function
- -G 2 : Rotated HappyCat Function
- -G 3 : Rotated Schwefel Function
- -G 4 : Rotated Expanded Scaffer's F6 Function
- -G 5 : High Conditioned Elliptic Function

## 6.7 Composition Function 7

- -N = 3
- -σ = [10 , 30 , 50]
- -λ = [1 , 1 , 1]
- -bias = [0 , 100 , 200]
- -G 1 : Hybrid Function 1
- -G 2 : Hybrid Function 2
- -G 3 : Hybrid Function 3

## 6.8 Composition Function 8

- -N = 3
- -σ = [10 , 30 , 50]
- -λ = [1 , 1 , 1]
- -bias = [0 , 100 , 200]
- -G 1 : Hybrid Function 4
- -G 2 : Hybrid Function 5
- -G 3 : Hybrid Function 6

## References

1. Finck, S., Hansen, N., Ros, R., Auger, A.: Real-parameter black-box optimization benchmarking 2010: Noiseless functions definitions. Technical Report 2009/20, Research Center PPE (2010)
2. Liang, J.J., Qu, B.Y., Suganthan, P.N., Hern´ andez-D´ ıaz, A.G.: Problem definitions and evaluation criteria for the cec 2013 special session and competition on real-parameter optimization. Technical Report 201212, Computational Intelligence Laboratory, Zhengzhou University and Nanyang Technological University, Singapore (2013)
3. Liang, J.J., Qu, B.Y., Suganthan, P.N.: Problem definitions and evaluation criteria for the cec 2014 special session and competition on single objective real-parameter numerical optimization. Technical Report 201311, Computational Intelligence Laboratory, Zhengzhou University and Nanyang Technological University, Singapore (2013)

## Appendices

## A Figures for 2-D Functions

<!-- image -->

Fig. 2. Sphere Function

<!-- image -->

<!-- image -->

Fig. 3. Ellipsoid Function

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 4. Elliptic Function

Fig. 5. Discus Function

<!-- image -->

<!-- image -->

Fig. 6. Bent Cigar Function

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 7. Different Powers Function

<!-- image -->

Fig. 8. Sharp Valley Function

<!-- image -->

<!-- image -->

Fig. 9. Step Function

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 10. Weierstrass Function

<!-- image -->

Fig. 11. Weierstrass Function

<!-- image -->

<!-- image -->

Fig. 12. Rastrigin Function

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 13. Rotated Rastrigin Function

<!-- image -->

Fig. 14. Schaffer's F7 Function

<!-- image -->

<!-- image -->

Fig. 15. Expanded Griewank Rosenbrock Function

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 16. Rosenbrock Function

Fig. 17. Schwefel Function

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 18. Rotated Schwefel Function

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 19. Katsuura Function

<!-- image -->

Fig. 20. Lunacek Function

<!-- image -->

Fig. 21. Ackley Function

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 22. HappyCat Function

Fig. 23. HGBat Function

<!-- image -->

<!-- image -->

Fig. 24. Expanded Scaffers' F6 Function

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 25. Composition Function 1

Fig. 26. Composition Function 2

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 27. Composition Function 3

<!-- image -->

<!-- image -->

<!-- image -->

Fig. 28. Composition Function 4

Fig. 29. Composition Function 5

<!-- image -->

<!-- image -->

Fig. 30. Composition Function 6

<!-- image -->

<!-- image -->