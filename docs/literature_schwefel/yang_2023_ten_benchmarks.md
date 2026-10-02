## Ten New Benchmarks for Optimization

## Xin-She Yang

School of Science and Technology, Middlesex University London, The Burroughs, London NW4 4BT, United Kingdom.

## Abstract

Benchmarks are used for testing new optimization algorithms and their variants to evaluate their performance. Most existing benchmarks are smooth functions. This chapter introduces ten new benchmarks with different properties, including noise, discontinuity, parameter estimation and unknown paths.

Keywords: Benchmark, Hybrid Algorithms, Nature-Inspired Algorithms, Optimization.

Citation Details: Xin-She Yang, Ten New Benchmarks for Optimization, in: Benchmarks and Hybrid Algorithms in Optimization and Applications (Edited by Xin-She Yang), Springer Tracts in Nature-Inspired Computing, pp. 19 - 32 (2023).

https://doi.org/10.1007/978-981-99-3970-1 2

## 1 Introduction

The literature of nature-inspired algorithms and swarm intelligence is expanding rapidly, and most nature-inspired optimization algorithms are swarm intelligence based algorithms [1, 2, 3, 4]. New algorithms for optimization appear regularly in the current literature and hundreds of papers are being published every year, about new algorithms and their variants as well as their applications. Obviously, new algorithms have to be tested and validated using various known problems with known optimality locations. Simple benchmark functions are often the first set of problems that new algorithms or variants attempt to solve.

However, most existing benchmarks are function benchmarks that are usually smooth with known optimality at a single location for a given function benchmark. In addition, the search domains of these functions in terms of their independent variables are usually regular, often expressed in terms of simple bounds and limits. Even though some function benchmarks have constraints, these constraints tend to be sufficiently simple, which do not change the shape of the search domains significantly. The tests using such smooth benchmarks with regular search domains may give some insights into the performance of the algorithms under consideration. However, the usefulness of such benchmarks may be quite limited because these functions have almost nothing to do with the realistic optimization problems from the real-world applications.

Ideally, we should have a diverse set of test benchmarks and case studies derived from real applications so that researchers can use them for testing algorithms, especially new algorithms. However, such test benchmarks are largely not available because case studies tend to be very specialized in a specific subject area. Even we may have such case studies as benchmarks, specialized knowledge in a subject area is needed to solve the problem properly and interpret the results correctly. If we can somehow extract the essential part of the optimization problems

and try to make them almost independent of special subjects, we can make the relevant problems as generic benchmarks.

In the rest of the chapter, we will first briefly discuss the role of benchmarking and the different types of benchmarks. Then, we will introduce ten new benchmarks with additional properties, such as noise, discontinuity, non-differentiability, multi-layered discrete values, and optimal paths from the calculus of variations.

## 2 Role of Benchmarks

There are many benchmark functions that are smooth functions with known optimal solutions and optimal objective values [5, 6, 7]. Such benchmarks enable researchers to test new algorithms so as to gain better understanding about the convergence behaviour, stability and performance of the algorithm under consideration.

Benchmarks can be very diverse. To validate any new optimization algorithm, a variety of test benchmarks should be used to see how the algorithm under consideration may perform for different types of problems. In general, benchmarks can be divided into five categories:

Figure 1: Different types of benchmarks.

<!-- image -->

- Smooth Functions : There are more than 200 smooth functions as benchmarks in the current literature [6, 8, 9, 7, 10]. Whether the problems are constrained or often unconstrained, their objective landscapes tend to be smooth. In addition, in some rare cases, such test functions can also be multi-objective optimization [11].
- Composite Functions : Though many simple test functions such as the Ackley function exist, composite functions have been designed to make it harder for algorithms to find the optimal solutions because the locations of the optimality of these functions are shifted and twisted.
- Implicit Solvers : In many applications, the exact form of the objective can be difficult to express explicitly. For example, many engineering design problems require to use finite element analysis (FEM) packages and computational fluid dynamics (CFD) packages to

evaluate the design performance. In this case, the actual evaluation of the design objectives are carried out by calling external solvers for given inputs (design variables and parameters) and then extracting the output (some objective or performance metrics).

- Special Benchmarks : Sometimes, specialized benchmarks are used to test certain types of optimization techniques. Benchmarks can be very specialized. For example, in protein folding and molecular biological applications, the test cases are often very specialized with given structure of data and objectives.
- Real-World Problems : For new optimization algorithms to be truly useful, they should be able to solve real-world problems. Therefore, extensive tests should be carried out using real-world problems as benchmarks. This category of benchmarks can be extremely diverse to cover almost all areas of applications.

Since there many different types of benchmarks, an important question is naturally: What benchmarks should be used for validating new algorithms?

For most of the studies in the current literature, the benchmarking process seems to use a finite set (typically about a dozen to two or three dozens of functions for testing algorithms, even though these functions are selected to have some diverse properties, such as mode shapes, optimality locations, separability of different dimensions, and even with some constraints. Though these functions themselves can be quite complicated with multiple modes, they are idealized functions, which may have nothing to do with problems arisen from real-world applications. Thus, whatever the conclusions may be drawn from testing functions, they may not be much use in practice. After all, most real-world problems can be even more complex with highly nonlinear constraints and irregular search domains. Therefore, it can be expected that algorithms work for test functions may not work well in practical applications.

## 3 New Benchmark Functions

Even the usefulness of simple benchmarking is quite limited, it is still an important part of the algorithm evaluation process. In addition to the existing test functions, we now introduce even more complex functions as new benchmarks. These new benchmarks highlight the challenges in algorithm testing and may inspire more research to test new algorithms from a wider perspective, including introducing noise, non-unique optimal solutions and even solutions in an infinitedimensional functional space.

## 3.1 Noisy Functions

As almost all existing benchmark problems are deterministic in the sense that the function forms and their solutions have no randomness. Now let us add some noise to a smooth function, which may make it more challenging for algorithms to find its optimality.

One way for adding noise without affecting the location of its optimality is to multiplying a random variable drawn from a uniform distribution. For example, we can have a noisy function

<!-- formula-not-decoded -->

as the extension to a standard sphere function. Here, all ϵ n are drawn from a uniform distribution in [0,1].

We can design an even more complicated function

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

where D ≥ 1 is the dimensionality of the function. Again, all ϵ n are drawn from a uniform distribution in [0, 1]. Its optimality f min = -1 occurs at x = (0 , 0 , ..., 0).

## 3.2 Non-differentiable Functions

The simplest function with a kink is probably f ( x ) = | x | , which has a global minimum f min = 0 at x ∗ = 0. However, this function does not have a well-defined derivative at x = 0. In the D -dimensional space, we can extend it to

<!-- formula-not-decoded -->

and its global minimum f min = 0 is located at x = (0 , 0 , ..., 0).

Obviously, we can design more functions with multiple kinks, such as

<!-- formula-not-decoded -->

with

<!-- formula-not-decoded -->

Its optimality f min = 0 occurs at x = ( π, 2 π, ..., nπ ).

## 3.3 Functions with Isolated Domains

In most benchmark problems, their search domains are typically regular in the form x n ∈ [ a, b ], ranging in an interval from x n = a to x n = b . This is usually true for unconstrained optimization problems with simple bounds or limits. For constrained optimization problems, depending on the actual constraints, feasible search domains can have very irregular shapes or even with isolated, fragmental regions.

For example, we can design a function with two isolated domains

<!-- formula-not-decoded -->

subject to and

<!-- formula-not-decoded -->

The minimum f min = a 2 occurs at x = ( a, 0 , 0 , ..., 0).

More generally, we can have a function with multiple isolated domains with four peaks as its optimal solutions [4]

<!-- formula-not-decoded -->

in the domains of

<!-- formula-not-decoded -->

where i, j are integers with N = 100 and a = 10. This function has 4( N +1) 2 local peaks, but it has four highest peaks at four corners. However, its domain is formed by many isolated regions, or 4( N +1) 2 = 40401 regions when N = 100.

<!-- formula-not-decoded -->

## 4 Benchmarks with Multiple Optimal Solutions

Almost all benchmark functions have their optimal objective values at a finite number of isolated points. For example, f ( x, y ) = x 2 + y 2 has only a single optimal solution at the point (0 , 0) with f min = 0. To make things more complicated, we can design functions with infinitely many solutions with equal objective values. For example, the function g ( x, y ) = x 2 + y 2 -2 xy = ( x -y ) 2 has the global minimum f min = 0 on the line y = x .

Many researchers use the closeness to the optimal solution such as the point (0,0) to measure the success of the algorithms used in the simulation, this may become a main issue if the optimal solutions are no long isolated points. We can expect that such types of functions can also make it more challenging for some algorithms to find their optimality.

## 4.1 Function on a Hyperboloid

We can easily extend a standard sphere function to a sphere function with a hyperboloid (of revolution) constraint

<!-- formula-not-decoded -->

subject to

<!-- formula-not-decoded -->

It has a minimum f min = a 2 on a ( D -1)-dimensional hyper-sphere

<!-- formula-not-decoded -->

which corresponds to infinitely many solutions.

In the case of 3D, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with f min = a 2 on x 2 + y 2 = 1.

## 4.2 Non-Smooth Multi-Layered Functions

Though the above functions are more complicated, their objective values are still continuous in most cases. There is no jump or discontinuity in their landscapes.

In addition to make the search domains for independent decision variables irregular and/or to make the objectives with kinks, we can also design functions with discontinuous objective landscapes. For example, we can make the objective values of a sphere function take only integer values, which can lead to a non-smooth multi-layered function

<!-- formula-not-decoded -->

where ⌊ x ⌋ is a floor function, which rounds x to the nearest integer smaller than x . That is, k = ⌊ x ⌋ &lt; = x . For example, ⌊ 2 . 3 ⌋ = 2 and ⌊ 0 . 17 ⌋ = 0.

The optimal solution of this function is f min = 0 inside a hyper-sphere

<!-- formula-not-decoded -->

subject to

Figure 2: A non-smooth function f ( x ) = ⌊ x 2 ⌋ with an optimal region -1 &lt; x &lt; +1.

<!-- image -->

Figure 3: Function ⌊| x | +cos( x 2 ) ⌋ with two lowest flat regions that are both optimal.

<!-- image -->

Any point inside this hyper-sphere is an optimal solution. Thus, this function can have infinitely many optimal solutions within a hyper-volume, and all solutions inside this region have the same optimal objective value f min = 0.

On the one hand, it seems that this may be easier for algorithms to find the solution; however, many statistical measures such as mean solutions and standard deviations used for comparison would not make much sense for this function. Care should be taken when analyzing results. On the other hand, this integer-value objective function is no longer smooth, thus gradient-based methods would not work well. For example, in the simplest one-dimensional case, the function is shown in Fig. 2.

To follow this same line of thinking, test benchmarks can have multiple isolated regions as optimal regions with the same objective value. For example, we can design a function in the following form:

<!-- formula-not-decoded -->

For D = 1, we have

<!-- formula-not-decoded -->

which has the global minimum value f min = 0 with infinitely many solutions in two disconnected, flat regions: ( -1 . 89714, -1 . 41299) and (1 . 41299, 1 . 89714), which are shown in Fig. 3.

Figure 4: Optimality occurs in multiple flat regions.

<!-- image -->

Table 1: Measured data for a vibration problem.

| Time t i   |   0 |      1 |      2 |      3 |      4 |      5 |      6 |      7 |      8 |      9 |     10 |
|------------|-----|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| y ( t i )  |   0 | 1.0706 | 1.3372 | 0.8277 | 0.9507 | 1.0848 | 0.9814 | 0.9769 | 1.0169 | 1.0012 | 0.9933 |

In the case of D = 2, we have

<!-- formula-not-decoded -->

which has 8 isolated flat regions with the same minimum f min = 0, as shown in Fig. 4.

## 5 Parameter Estimation as Benchmarks

For a vibration problem with a unit step input [12], we have its mathematical equation as an ordinary differential equation

<!-- formula-not-decoded -->

where ω and ζ are the two parameters to be estimated. Here, the unit step function is given

<!-- formula-not-decoded -->

Suppose for a given system, we have observed its actual response. The relevant measurements are given Table 1.

In order to estimate the two unknown parameter values ω and ζ , we can define the objective function as

<!-- formula-not-decoded -->

where y ( t i ) for i = 0 , 1 , ..., 10 are the observed values and y s ( t i ) are the values obtained by solving the differential equation (21), given a guessed set of values ζ and ω . Here, we have used x = ( ζ, ω ).

The true values are ζ = 1 4 and ω = 2. The aim of this benchmark is to solve the differential equation iteratively so as to find the best parameter values that minimize the objective or best-fit errors.

Figure 5: Variation of f ( x ) = sin( kx ) / ( x exp( βx )) with k = 3 and β = 0 . 25.

<!-- image -->

## 6 Integrals as Benchmarks

For almost all the benchmarks in the literature, integrals rarely appear in the problem formulations. Sometimes, the evaluation of an integral can be challenging, and thus the benchmarks involving integrals can make things more difficult.

From basic calculus, we know that it is very challenging to calculate the well-known integral

<!-- formula-not-decoded -->

Suppose we want to maximize the integral

<!-- formula-not-decoded -->

where β ≥ 0 is a real number and k &gt; 0 is an integer.

Though theoretically we know that

<!-- formula-not-decoded -->

which does not depend on k . The maximum value of this integral occurs at β = 0 for any positive integer k . In case of β = 0 . 5 and k = 3, the variation of the integrand is shown in Fig. 5.

Now let us use the maximization of this integral as a benchmark, which requires to evaluate the integral with an infinite limit. If without any prior knowledge of its true value of this integral, the effective evaluation of the objective function requires some sophisticated numerical integration. This requires a careful implementation.

## 7 Benchmarks of Infinite Dimensions

All the function benchmarks in the literature have reasonably a well-defined solution set as the possible optimal solutions. Even in the high-dimensional space, such optimal solutions are just points or a region in the D -dimensional search space because the solutions are represented as a solution vector x .

In some applications in science and engineering, the optimal solutions may not be represented by vectors. For example, in the calculus of variations [12], optimal solutions are curves and surfaces that cannot be represented by simple vectors. For this branch of mathematics, we have rigorous theory using the Euler-Lagrange equation to solve such type of problems.

Now let us reformulate such problems and try to solve them using optimization algorithms without using the Euler-Lagrange equation. Since almost all optimization algorithms have been designed to represent solutions in terms of vectors, this kind of problem from calculus of variations can be a major issue for standard optimization algorithms. In aerodynamics and shape optimization, shapes are represented by some parametric curves so as to simplify the representations. Even so, shape optimization can be quite challenging to implement.

To provide benchmarks for optimization algorithms from this perspective, let us use two examples to design benchmarks with solutions as paths or curves. Even the actual space ( x, y ) is two-dimensional, the representation of solutions may require many points (or infinitely many points) to form a smooth curve. In fact, we need a functional space to represent the potential curves properly. In this case, we can refer to this type of problem as benchmarks in infinite dimensions.

## 7.1 Shortest-Path Problem

In the two-dimensional space ( x, y ), there is a path or curve y ( x ) that minimize the integral

<!-- formula-not-decoded -->

From the plane geometry or calculus of variations [12], we know that the solution is a straight line y = x from the origin (0,0) to point (1,1).

The challenge for this benchmark as an optimization problem is that the solution is not a single point but a segment of a curve (or a straight line in this case). In order to find this solution (corresponding to infinitely many points), some parametrization of an unknown curve is needed. Therefore, any standard algorithms for solving single-objective optimization problems have to be modified so that solution paths can be represented in effectively.

Mathematically, the objective functional can be rewritten as

<!-- formula-not-decoded -->

subject to x ≥ 0 and y ( x ) ≥ 0.

## 7.2 Shape Optimization

The shape of a hanging rope under gravity takes the form that the potential energy is minimized, subject to a fixed length L . The shape of a loose rope hinged between two fixed points ( -a, 0) and ( a, 0) can be obtained by minimizing the potential energy

<!-- formula-not-decoded -->

where ρ is the density of the rope and g is the acceleration due to gravity. Here, we use the notation y ′ = dy/dx for a given smooth function y ( x ). Without loss of generality, we can use ρg = 1, and thus we have

<!-- formula-not-decoded -->

This minimization problem is subject to an equality

<!-- formula-not-decoded -->

with L &gt; 2 a.

Figure 6: Shape of a hanging rope with a fixed length.

<!-- image -->

The solution can be obtained by solving the Euler-Lagrange equation, which corresponds to the shape or curve (see Fig. 6)

<!-- formula-not-decoded -->

with a length

<!-- formula-not-decoded -->

In case of a = 1, we have L = e -e -1 ≈ 2 . 3504.

The challenge of this benchmark is not only to represent the solutions properly, but also to deal with the equality constraint correctly. In addition, the algorithm to be used must also be modified to accommodate such additional requirements.

## 8 Conclusions

There are many benchmarks for testing optimization algorithms; however, the benchmarks in the current literature tend to be smooth functions or problems with a finite number of optimal solutions. In addition, the search domains of existing benchmarks are usually regular with simple bounds or limits. To extend benchmarks to be more relevant to realistic problems, we have introduced ten new benchmarks by adding some noise, isolating the search domains and even making problems non-smooth with singularity and discontinuities.

These new benchmarks can be used to validate new algorithms and existing algorithms to see if they can cope with problems with non-smoothness and singularity well. We hope that this will inspire more research into benchmark problems and how to test new algorithms properly.

## References

- [1] Yang, X.S.: Nature-Inspired Optimization Algorithms. Elsevier Insight, London (2014)
- [2] Yang, X.S., He, X.S.: Mathematical Foundations of Nature-Inspired Algorithms. Springer Briefs in Optimization. Springer, Cham, Switzerland (2019)
- [3] Yang, X.S.: Optimization Techniques and Applications with Examples. John Wiley &amp; Sons, Hoboken, NJ, USA (2018)
- [4] Yang, X.S.: Nature-inspired optimization algorithms: challenges and open problems. Journal of Computational Science Article 101104 (2020)
- [5] Yang, X.S.: Firefly algorithm, stochastic test functions and design optimisation. Int. J. Bio-Inspired Computation 2 (2) (2010) 78-84

- [6] Jamil, M., Yang, X.S.: A literature survey of benchmark functions for global optimisation problems. International Journal of Mathematical Modelling and Numerical Optimisation 4 (2) (2013) 150-194
- [7] Suganthan, P., Hansen, N., Liang, J., Deb, K., Chen, Y., Auger, A., Tiwar, S.: Problem definitions and evaluation criteria for cec 2005, special session on real-parameter optimization, technical report. Technical report, Nanyang Technological University (NTU), Singapore (2005)
- [8] Mazhar, A.A.: Benchmark functions (web site). Technical report, Victoria University of Wellington, New Zealand (2020)
- [9] Hedar, A.: Global optimization test problems (web site). Technical report, University of Kyoto, Japan (2011)
- [10] Kumar, A., Wu, G., Ali, M.Z., Mallipeddi, R., Suganthan, P.N., Das, S.: A test-suite of non-convex constrained optimization problems for the real-world and some baseline results. Swarm and Evolutionary Computation 56 (Article 100693) (2020)
- [11] Zitzler, E., Deb, K., Thiele, L.: Comparison of multiobjective evolutionary algorithms: emperical results. Evolutionary Computation 8 (2) (2000) 173-195
- [12] Yang, X.S.: Engineering Mathematics with Examples and Applications. Academic Press, London (2017)