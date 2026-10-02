## A Literature Survey of Benchmark Functions For Global Optimization Problems

Momin Jamil ∗† , Xin-She Yang ‡ ∗ Blekinge Institute of Technology SE-37179, Karlskrona, Sweden † Harman International, Cooperate Division Becker-Goering Str. 16, D-76307 Karlsbad, Germany E-mail: momin.jamil@harman.com ‡ Middlesex University School of Science and Technology Hendon Campus, London NW4 4BT, UK E-mail: xin-she.yang@middlesex.ac.uk

## Citation details:

Momin Jamil and Xin-She Yang, A literature survey of benchmark functions for global optimization problems, Int. Journal of Mathematical Modelling and Numerical Optimisation , Vol. 4, No. 2, pp. 150-194 (2013). DOI: 10.1504/IJMMNO.2013.055204

Test functions are important to validate and compare the performance of optimization algorithms. There have been many test or benchmark functions reported in the literature; however, there is no standard list or set of benchmark functions. Ideally, test functions should have diverse properties so that can be truly useful to test new algorithms in an unbiased way. For this purpose, we have reviewed and compiled a rich set of 175 benchmark functions for unconstrained optimization problems with diverse properties in terms of modality, separability, and valley landscape. This is by far the most complete set of functions so far in the literature, and tt can be expected this complete set of functions can be used for validation of new optimization in the future.

## 1 Introduction

The test of reliability, efficiency and validation of optimization algorithms is frequently carried out by using a chosen set of common standard benchmarks or test functions from the literature. The number of test functions in most papers varied from a few to about two dozens. Ideally, the test functions used should be diverse and unbiased, however, there is no agreed set of test functions in the literature. Therefore, the major aim of this paper is to review and compile the most complete set of test functions that we can find from all the available literature so that they can be used for future validation and comparison of optimization algorithms.

For any new optimization, it is essential to validate its performance and compare with other existing algorithms over a good set of test functions. A common practice followed by many researches is to compare different algorithms on a large test set, especially when

the test involves function optimization (Gordon 1993, Whitley 1996). However, it must be noted that effectiveness of one algorithm against others simply cannot be measured by the problems that it solves if the the set of problems are too specialized and without diverse properties. Therefore, in order to evaluate an algorithm, one must identify the kind of problems where it performs better compared to others. This helps in characterizing the type of problems for which an algorithm is suitable. This is only possible if the test suite is large enough to include a wide variety of problems, such as unimodal, multimodal, regular, irregular, separable, non-separable and multi-dimensional problems.

Many test functions may be scattered in different textbooks, in individual research articles or at different web sites. Therefore, searching for a single source of test function with a wide variety of characteristics is a cumbersome and tedious task. The most notable attempts to assemble global optimization test problems can be found in [4, 7, 8, 15, 20, 29, 28, 32, 33, 52, 64, 65, 66, 74, 77, 78, 82, 83, 84, 86]. Online collections of test problems also exist, such as the GLOBAL library at the cross-entropy toolbox [18], GAMS World [36] CUTE [41], global optimization test problems collection by Hedar [43], collection of test functions [5, 37, 48, 53, 54, 55, 56, 57, 58, 59], a collection of continuous global optimization test problems COCONUT [61] and a subset of commonly used test functions [88]. This motivates us to carry out a thorough analysis and compile a comprehensive collection of unconstrained optimization test problems.

In general, unconstrained problems can be classified into two categories: test functions and real-world problems. Test functions are artificial problems, and can be used to evaluate the behavior of an algorithm in sometimes diverse and difficult situations. Artificial problems may include single global minimum, single or multiple global minima in the presence of many local minima, long narrow valleys, null-space effects and flat surfaces. These problems can be easily manipulated and modified to test the algorithms in diverse scenarios. On the other hand, real-world problems originate from different fields such as physics, chemistry, engineering, mathematics etc. These problems are hard to manipulate and may contain complicated algebraic or differential expressions and may require a significant amount of data to compile. A collection of real-world unstrained optimization problems can be found in [7, 8].

In this present work, we will focus on the test function benchmarks and their diverse properties such as modality and separability. A function with more than one local optimum is called multimodal. These functions are used to test the ability of an algorithm to escape from any local minimum. If the exploration process of an algorithm is poorly designed, then it cannot search the function landscape effectively. This, in turn, leads to an algorithm getting stuck at a local minimum. Multi-modal functions with many local minima are among the most difficult class of problems for many algorithms. Functions with flat surfaces pose a difficulty for the algorithms, since the flatness of the function does not give the algorithm any information to direct the search process towards the minima (Stepint, Matyas, PowerSum). Another group of test problems is formulated by separable and non-separable functions. According to [16], the dimensionality of the search space is an important issue with the problem. In some functions, the area that contains that global minima are very small, when compared to the whole search space, such as Easom, Michalewicz ( m =10) and Powell. For problems such as Perm, Kowalik and Schaffer, the global minimum is located very close to the local minima. If the algorithm cannot keep up the direction changes in the functions with a narrow curved valley, in case of functions like Beale, Colville, or cannot explore the search space effectively, in case of function like Pen Holder, Testtube-Holder having multiple global minima, the algoritm will fail for these kinds of problems. Another problem that

algorithms may suffer is the scaling problem with many orders of magnitude differences between the domain and the function hyper-surface [47], such as Goldstein-Price and Trid.

## 2 Characteristics of Test Functions

The goal of any global optimization (GO) is to find the best possible solutions x ∗ from a set X according to a set of criteria F = { f 1 , f 2 , · · · f n } . These criteria are called objective functions expressed in the form of mathematical functions. An objective function is a mathematical function f : D ⊂ /Rfractur n → /Rfractur subject to additional constraints. The set D is referred to as the set of feasible points in a search space. In the case of optimizing a single criterion f , an optimum is either its maximum or minimum. The global optimization problems are often defined as minimization problems, however, these problems can be easily converted to maximization problems by negating f . A general global optimum problem can be defined as follows:

<!-- formula-not-decoded -->

The true optimal solution of an optimization problem may be a set of x ∗ ∈ D of all optimal points in D , rather than a single minimum or maximum value in some cases. There could be multiple, even an infinite number of optimal solutions, depending on the domain of the search space. The tasks of any good global optimization algorithm is to find globally optimal or at least sub-optimal solutions. The objective functions could be characterized as continuous, discontinuous, linear, non-linear, convex, non-conxex, unimodal, multimodal, separable 1 and non-separable.

According to [20], it is important to ask the following two questions before start solving an optimization problem; (i) What aspects of the function landscape make the optimization process difficult? (ii) What type of a priori knowledge is most effective for searching particular types of function landscape? In order to answer these questions, benchmark functions can be classified in terms of features like modality, basins, valleys, separability and dimensionality [87].

## 2.1 Modality

The number of ambiguous peaks in the function landscape corresponds to the modality of a function. If algorithms encounters these peaks during a search process, there is a tendency that the algorithm may be trapped in one of such peaks. This will have a negative impact on the search process, as this can direct the search away from the true optimal solutions.

## 2.2 Basins

Arelatively steep decline surrounding a large area is called a basin. Optimization algorithms can be easily attracted to such regions. Once in these regions, the search process of an algorithm is severely hampered. This is due to lack of information to direct the search process towards the minimum. According to [20], a basin corresponds to the plateau for a maximization problem, and a problem can have multiple plateaus.

1 In this paper, partially separable functions are also considered as separable function

## 2.3 Valleys

A valley occurs when a narrow area of little change is surrounded by regions of steep descent [20]. As with the basins, minimizers are initially attracted to this region. The progress of a search process of an algorithm may be slowed down considerably on the floor of the valley.

## 2.4 Separability

The separability is a measure of difficulty of different benchmark functions. In general, separable functions are relatively easy to solve, when compared with their inseperable counterpart, because each variable of a function is independent of the other variables. If all the parameters or variables are independent, then a sequence of n independent optimization processes can be performed. As a result, each design variable or parameter can be optimized independently. According to [74], the general condition of separability to see if the function is easy to optimize or not is given as

<!-- formula-not-decoded -->

where g ( x i ) means any function of x i only and h ( x ) any function of any x . If this condition is satisfied, the function is called partially separable and easy to optimize, because solutions for each x i can be obtained independently of all the other parameters. This separability condition can be illustrated by the following two examples.

For example, function ( f 105 ) is not separable, because it does not satisfy the condition (2)

<!-- formula-not-decoded -->

On the other hand, the sphere function ( f 137 ) with two variables can indeed satisfy the above condition (2) as shown below.

<!-- formula-not-decoded -->

where h ( x ) is regarded as 1.

In [16], the formal definition of separability is given as

<!-- formula-not-decoded -->

In other words, a function of p variables is called separable, if it can written as a sum of p functions of just one variable [16]. On the other hand, a function is called nonseparable, if its variables show inter-relation among themselves or are not independent. If the objective function variables are independent of each other, then the objective functions can be decomposed into sub-objective functions. Then, each of these sub-objectives involves only one decision variable, while treating all the others as constant and can be expressed as

<!-- formula-not-decoded -->

## 2.5 Dimensionality

The difficulty of a problem generally increases with its dimensionality. According to [87, 90], as the number of parameters or dimension increases, the search space also increases exponentially. For highly nonlinear problems, this dimensionality may be a significant barrier for almost all optimization algorithms.

## 3 Benchmark Test Functions for Global Optimization

Now, we present a collection of 175 unconstrained optimization test problems which can be used to validate the performance of optimization algorithms. The dimensions, problem domain size and optimal solution are denoted by D , Lb ≤ x i ≤ Ub and f ( x ∗ ) = f ( x 1 , ...x n ), respectively. The symbols Lb and Ub represent lower, upper bound of the variables, respectively. It is worth noting that in several cases, the optimal solution vectors and their corresponding solutions are known only as numerical approximations.

1. Ackley 1 Function [9](Continuous, Differentiable, Non-separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -35 ≤ x i ≤ 35. The global minima is located at origin x ∗ = (0 , · · · , 0), f ( x ∗ ) = 0.

2. Ackley 2 Function [1] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -32 ≤ x i ≤ 32. The global minimum is located at origin x ∗ = (0 , 0), f ( x ∗ ) = -200.

3. Ackley 3 Function [1] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -32 ≤ x i ≤ 32. The global minimum is located at x ∗ = (0 , ≈ -0 . 4), f ( x ∗ ) ≈ -219 . 1418.

4. Ackley 4 or Modified Ackley Function (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -35 ≤ x i ≤ 35. It is highly multimodal function with two global minimum close to origin x = f ( {-1 . 479252 , -0 . 739807 } , { 1 . 479252 , -0 . 739807 } ), f ( x ∗ ) = -3 . 917275.

5. Adjiman Function [2](Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -1 ≤ x 1 ≤ 2, -1 ≤ x 2 ≤ 1. The global minimum is located at x ∗ = (2 , 0 . 10578), f ( x ∗ ) = -2 . 02181.

6. Alpine 1 Function [69](Continuous, Non-Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

∣ subject to -10 ≤ x i ≤ 10. The global minimum is located at origin x ∗ = (0 , · · · , 0), f ( x ∗ ) = 0.

7. Alpine 2 Function [21] (Continuous, Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x i ≤ 10. The global minimum is located at x ∗ = (7 . 917 · · · 7 . 917), f ( x ∗ ) = 2 . 808 D .

8. Brad Function [17] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

where u i = i , v i = 16 -i , w i = min( u i , v i ) and y = y i = [0 . 14, 0 . 18, 0 . 22, 0 . 25 , 0 . 29, 0 . 32 , 0 . 35, 0 . 39 , 0 . 37, 0 . 58 , 0 . 73 , 0 . 96, 1 . 34 , 2 . 10 , 4 . 39] T . It is subject to -0 . 25 ≤ x 1 ≤ 0 . 25, 0 . 01 ≤ x 2 , x 3 ≤ 2 . 5. The global minimum is located at x ∗ = (0 . 0824 , 1 . 133 , 2 . 3437), f ( x ∗ ) = 0 . 00821487.

9. Bartels Conn Function (Continuous, Non-differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

∣ ∣ ∣ ∣ ∣ ∣ subject to -500 ≤ x i ≤ 500. The global minimum is located at x ∗ = (0 , 0), f ( x ∗ ) = 1.

10. Beale Function (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -4 . 5 ≤ x i ≤ 4 . 5. The global minimum is located at x ∗ = (3 , 0 . 5), f ( x ∗ ) = 0.

11. Biggs EXP2 Function [13] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

where t i = 0 . 1 i , y i = e -t i -5 e 10 t i . It is subject to 0 ≤ x i ≤ 20. The global minimum is located at x ∗ = (1 , 10), f ( x ∗ ) = 0.

12. Biggs EXP3 Function [13] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

where t i = 0 . 1 i , y i = e -t i -5 e 10 t i . It is subject to 0 ≤ x i ≤ 20. The global minimum is located at x ∗ = (1 , 10 , 5), f ( x ∗ ) = 0.

13. Biggs EXP4 Function [13] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

where t i = 0 . 1 i , y i = e -t i -5 e 10 t i . It is subject to 0 ≤ x i ≤ 20. The global minimum is located at x ∗ = (1 , 10 , 1 , 5), f ( x ∗ ) = 0.

14. Biggs EXP5 Function [13] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

where t i = 0 . 1 i , y i = e -t i -5 e 10 t i +3 e -4 t i . It is subject to 0 ≤ x i ≤ 20. The global minimum is located at x ∗ = (1 , 10 , 1 , 5 , 4), f ( x ∗ ) = 0.

15. Biggs EXP5 Function [13] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

where t i = 0 . 1 i , y i = e -t i -5 e 10 t i +3 e -4 t i . It is subject to -20 ≤ x i ≤ 20. The global minimum is located at x ∗ = (1 , 10 , 1 , 5 , 4 , 3), f ( x ∗ ) = 0.

16. Bird Function [58] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -2 π ≤ x i ≤ 2 π . The global minimum is located at x ∗ = (4 . 70104, 3 . 15294),( -1 . 58214, -3 . 13024), f ( x ∗ ) = -106 . 764537.

17. Bohachevsky 1 Function [14] (Continuous, Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0.

18. Bohachevsky 2 Function [14] (Continuous, Differentiable, Non-separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0.

19. Bohachevsky 3 Function [14] (Continuous, Differentiable, Non-Separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0.

20. Booth Function (Continuous, Differentiable, Non-separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is located at x ∗ = f (1 , 3), f ( x ∗ ) = 0.

21. Box-Betts Quadratic Sum Function [4] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

subject to 0 . 9 ≤ x 1 ≤ 1 . 2, 9 ≤ x 2 ≤ 11 . 2, 0 . 9 ≤ x 2 ≤ 1 . 2. The global minimum is located at x ∗ = f (1 , 10 , 1) f ( x ∗ ) = 0.

22. Branin RCOS Function [15] (Continuous, Differentiable, Non-Separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

with domain -5 ≤ x 1 ≤ 10, 0 ≤ x 1 ≤ 15. It has three global minima at x ∗ = f ( {-π, 12 . 275 } , { π, 2 . 275 } , { 3 π, 2 . 425 } ), f ( x ∗ ) = 0 . 3978873.

23. Branin RCOS 2 Function [60] (Continuous, Differentiable, Non-Separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

with domain -5 ≤ x i ≤ 15. The global minimum is located at x ∗ = f ( -3 . 2 , 12 . 53), f ( x ∗ ) = 5 . 559037.

24. Brent Function [15] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

with domain -10 ≤ x i ≤ 10. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0.

25. Brown Function [10] (Continuous, Differentiable, Non-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -1 ≤ x i ≤ 4. The global minimum is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

Bukin functions [80] are almost fractal (with fine seesaw edges) in the surroundings of their minimal points. Due to this property, they are extremely difficult to optimize by any global or local optimization methods.

26. Bukin 2 Function (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -15 ≤ x 1 ≤ -5 and -3 ≤ x 2 ≤ -3. The global minimum is located at x ∗ = f ( -10 , 0), f ( x ∗ ) = 0.

27. Bukin 4 Function (Continuous, Non-Differentiable, Separable, Non-scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -15 ≤ x 1 ≤ -5 and -3 ≤ x 2 ≤ -3. The global minimum is located at x ∗ = f ( -10 , 0), f ( x ∗ ) = 0.

28. Bukin 6 Function (Continuous, Non-Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -15 ≤ x 1 ≤ -5 and -3 ≤ x 2 ≤ -3. The global minimum is located at x ∗ = f ( -10 , 1), f ( x ∗ ) = 0.

29. Camel Function - Three Hump [15] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -5 ≤ x i ≤ 5. The global minima is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0.

30. Camel Function - Six Hump [15] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -5 ≤ x i ≤ 5. The two global minima are located at x ∗ = f ( {-0 . 0898 , 0 . 7126 } , { 0 . 0898 , -0 . 7126 , 0 } ), f ( x ∗ ) = -1 . 0316.

31. Chen Bird Function [19] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500 The global minimum is located at x ∗ = f ( -7 18 , -13 18 ), f ( x ∗ ) = -2000.

32. Chen V Function [19] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500 The global minimum is located at x ∗ = f ( -0 . 3888889, 0 . 7222222), f ( x ∗ ) = -2000.

33. Chichinadze Function (Continuous, Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -30 ≤ x i ≤ 30. The global minimum is located at x ∗ = f (5 . 90133 , 0 . 5), f ( x ∗ ) = -43 . 3159.

34. Chung Reynolds Function [20] (Continuous, Differentiable, Partially-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

35. Cola Function [3] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

The 17-dimensional function computes indirectly the formula ( D,u ) by setting x 0 = y 0 , x 1 = u 0 , x i = u 2( i -2) , y i = u 2( i -2)+1

<!-- formula-not-decoded -->

where r i,j is given by

<!-- formula-not-decoded -->

and d is a symmetric matrix given by

<!-- formula-not-decoded -->

This function has bounds 0 ≤ x 0 ≤ 4 and -4 ≤ x i ≤ 4 for i = 1 . . . D -1. It has a global minimum of f ( x ∗ ) = 11 . 7464.

36. Colville Function (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minima is located at x ∗ = f (1 , · · · , 1), f ( x ∗ ) = 0.

37. Corana Function [22] (Discontinuous, Non-Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500. The global minimum is located at x ∗ = f (0 , 0 , 0 , 0), f ( x ∗ ) = 0.

38. Cosine Mixture Function [4] (Discontinuous, Non-Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -1 ≤ x i ≤ 1. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = (0 . 2or 0 . 4) for n = 2 and 4 respectively.

39. Cross-in-Tray Function [58] (Continuous, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10.

The four global minima are located at x ∗ = f ( ± 1 . 349406685353340, ± 1 . 349406608602084), f ( x ∗ ) = -2 . 06261218.

40. Csendes Function [25] (Continuous, Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -1 ≤ x i ≤ 1. The global minimum is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

41. Cube Function [49] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is located at x ∗ = f ( -1 , 1), f ( x ∗ ) = 0.

42. Damavandi Function [26] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x i ≤ 14. The global minimum is located at x ∗ = f (2 , 2), f ( x ∗ ) = 0.

43. Deb 1 Function (Continuous, Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -1 ≤ x i ≤ 1. The number of global minima is 5 D that are evenly spaced in the function landscape, where D represents the dimension of the problem.

44. Deb 3 Function (Continuous, Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -1 ≤ x i ≤ 1. The number of global minima is 5 D that are unevenly spaced in the function landscape, where D represents the dimension of the problem.

45. Deckkers-Aarts Function [4] (Continuous, Differentiable, Non-Separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

subject to -20 ≤ x i ≤ 20. The two global minima are located at x ∗ = f (0 , ± 15) f ( x ∗ ) = -24777.

46. deVilliers Glasser 1 Function [27](Continuous, Differentiable, Non-Separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

where t i = 0 . 1( i -1), y i = 60 . 137 × 1 . 371 t i sin(3 . 112 t i + 1 . 761). It is subject to -500 ≤ x i ≤ 500. The global minimum is f ( x ∗ ) = 0.

47. deVilliers Glasser 2 Function [27] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

where t i = 0 . 1( i -1), y i = 53 . 81 × 1 . 27 t i tanh(3 . 012 t i +sin(2 . 13 t i )) cos( e 0 . 507 t i ). It is subject to -500 ≤ x i ≤ 500. The global minimum is f ( x ∗ ) = 0.

48. Dixon &amp; Price Function [28] (Continuous, Differentiable, Non-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is located at x ∗ = f (2 ( 2 i -2 2 i )), f ( x ∗ ) = 0.

49. Dolan Function (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is f ( x ∗ ) = 0.

50. Easom Function [20](Continuous, Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is located at x ∗ = f ( π, π ), f ( x ∗ ) = -1.

51. El-Attar-Vidyasagar-Dutta Function [30] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500. The global minimum is located at x ∗ = f (2 . 842503 , 1 . 920175), f ( x ∗ ) = 0 . 470427.

52. Egg Crate Function (Continuous, Separable, Non-Scalable)

<!-- formula-not-decoded -->

subject to -5 ≤ x i ≤ 5. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0.

53. Egg Holder Function (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -512 ≤ x i ≤ 512. The global minimum is located at x ∗ = f (512 , 404 . 2319), f ( x ∗ ) ≈ 959 . 64.

54. Exponential Function [70] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -1 ≤ x i ≤ 1. The global minima is located at x = f (0 , · · · , 0), f ( x ∗ ) = 1.

## 55. Exp 2 Function [3] (Separable)

<!-- formula-not-decoded -->

with domain 0 ≤ x i ≤ 20. The global minimum is located at x ∗ = f (1 , 10), f ( x ∗ ) = 0.

56. Freudenstein Roth Function [71] (Continuous, Differentiable, Non-Separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is located at x ∗ = f (5 , 4), f ( x ∗ ) = 0.

57. Giunta Function [58] (Continuous, Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -1 ≤ x i ≤ 1. The global minimum is located at x ∗ = f (0 . 45834282 , 0 . 45834282), f ( x ∗ ) = 0 . 060447.

58. Goldstein Price Function [38] (Continuous, Differentiable, Non-separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

subject to -2 ≤ x i ≤ 2. The global minimum is located at x ∗ = f (0 , -1), f ( x ∗ ) = 3.

59. Griewank Function [40] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

60. Gulf Research Problem [79] (Continuous, Differentiable, Non-Separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

where u i = 25 + [ -50ln(0 . 01 i )] 1 / 1 . 5 subject to 0 . 1 ≤ x 1 ≤ 100, 0 ≤ x 2 ≤ 25 . 6 and 0 ≤ x 1 ≤ 5. The global minimum is located at x ∗ = f (50 , 25 , 1 . 5), f ( x ∗ ) = 0.

61. Hansen Function [34] (Continuous, Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The multiple global minima are located at

<!-- formula-not-decoded -->

62. Hartman 3 Function [42] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x j ≤ 1, j ∈ { 1 , 2 , 3 } with constants a ij , p ij and c i are given as

<!-- formula-not-decoded -->

The global minimum is located at x ∗ = f (0 . 1140 , 0 . 556 , 0 . 852), f ( x ∗ ) ≈ -3 . 862782.

63. Hartman 6 Function [42] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x j ≤ 1, j ∈ { 1 , · · · , 6 } with constants a ij , p ij and c i are given as

<!-- formula-not-decoded -->

The global minima is located at x = f (0 . 201690 , 0 . 150011 , 0 . 476874 , 0 . 275332 , ... 0 . 311652 , 0 . 657301), f ( x ∗ ) ≈ -3 . 32236.

64. Helical Valley [32] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

 subject to -10 ≤ x i ≤ 10. The global minima is located at x ∗ = f (1 , 0 , 0), f ( x ∗ ) = 0.

65. Himmelblau Function [45] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -5 ≤ x i ≤ 5. The global minimum is located at x ∗ = f (3 , 2), f ( x ∗ ) = 0.

66. Hosaki Function [11] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x 1 ≤ 5 and 0 ≤ x 2 ≤ 6. The global minimum is located at x ∗ = f (4 , 2), f ( x ∗ ) ≈ -2 . 3458.

67. Jennrich-Sampson Function [46] (Continuous, Differentiable, Non-Separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

subject to -1 ≤ x i ≤ 1. The global minimum is located at x ∗ = f (0 . 257825 , 0 . 257825), f ( x ∗ ) = 124 . 3612.

68. Langerman-5 Function [12] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x j ≤ 10, where j ∈ [0 , D -1] and m = 5. It has a global minimum value of f ( x ∗ ) = -1 . 4. The matrix A and column vector c are given as

The matrix A is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

69. Keane Function (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

subject to 0 ≤ x i ≤ 10.

<!-- formula-not-decoded -->

The multiple global minima are located at x ∗ = f ( { 0 , 1 . 39325 } , { 1 . 39325 , 0 } ), f ( x ∗ ) = -0 . 673668.

70. Leon Function [49](Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -1 . 2 ≤ x i ≤ 1 . 2. A global minimum is located at f ( x ∗ ) = f (1 , 1), f ( x ∗ ) = 0.

71. Matyas Function [43] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0.

72. McCormick Function [50] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -1 . 5 ≤ x 1 ≤ 4 and -3 ≤ x 2 ≤ 3. The global minimum is located at x ∗ = f ( -0 . 547 , -1 . 547), f ( x ∗ ) ≈ -1 . 9133.

73. Miele Cantrell Function [24] (Continuous, Differentiable, Non-Separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

subject to -1 ≤ x i ≤ 1. The global minimum is located at x ∗ = f (0 , 1 , 1 , 1), f ( x ∗ ) = 0.

74. Mishra 1 Function [53] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x i ≤ 1. The global minimum is f ( x ∗ ) = 2.

75. Mishra 2 Function [53] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x i ≤ 1. The global minimum is f ( x ∗ ) = 2.

76. Mishra 3 Function [58] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

∣ ∣ ∣ ∣ The global minimum is located at x ∗ = f ( -8 . 466 , -10), f ( x ∗ ) = -0 . 18467.

77. Mishra 4 Function [58] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

∣ ∣ ∣ ∣ The global minimum is located at x ∗ = f ( -9 . 94112 , -10), f ( x ∗ ) = -0 . 199409.

78. Mishra 5 Function [58] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

The global minimum is located at x ∗ = f ( -1 . 98682 , -10), f ( x ∗ ) = -1 . 01983.

79. Mishra 6 Function [58] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

The global minimum is located at x ∗ = f (2 . 88631 , 1 . 82326), f ( x ∗ ) = -2 . 28395.

80. Mishra 7 Function (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

The global minimum is f ( x ∗ ) = 0.

81. Mishra 8 Function [58] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

The global minimum is located at

<!-- formula-not-decoded -->

82. Mishra 9 Function [58] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

where a = 2 x 3 1 +5 x 1 x 2 +4 x 3 -2 x 2 1 x 3 -18, b = x 1 + x 3 2 + x 1 x 2 3 -22 c = 8 x 2 1 +2 x 2 x 3 +2 x 2 2 +3 x 3 2 -52. The global minimum is located at x ∗ = f (1 , 2 , 3), f ( x ∗ ) = 0.

83. Mishra 10 Function [58] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

The global minimum is located at x = f (0 , 0) , (2 , 2) , f ( x ) = 0.

<!-- formula-not-decoded -->

84. Mishra 11 Function [58] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

The global minimum is f ( x ∗ ) = 0.

85. Parsopoulos Function (Continuous, Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -5 ≤ x i ≤ 5, where ( x 1 , x 2 ) ∈ R 2 . This function has infinite number of global minima in R 2 , at points ( κ π 2 , λπ ), where κ = ± 1 , ± 3 , ... and λ = 0 , ± 1 , ± 2 , ... . In the given domain problem, function has 12 global minima all equal to zero.

86. Pen Holder Function [58] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -11 ≤ x i ≤ 11. The four global minima are located at x ∗ = f ( ± 9 . 646168, ± 9 . 646168), f ( x ∗ ) = -0 . 96354.

87. Pathological Function [69] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minima is located x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

88. Paviani Function [45] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 2 . 0001 ≤ x i ≤ 10, i ∈ 1 , 2 , ..., 10. The global minimum is located at x ∗ ≈ f (9 . 351 , ...., 9 . 351), f ( x ∗ ) ≈ -45 . 778.

89. Pint´ er Function [63] (Continuous, Differentiable, Non-separable, Scalable, Multimodal)

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where x 0 = x D and x D +1 = x 1 , subject to -10 ≤ x i ≤ 10. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 90. Periodic Function [4] (Separable)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0 . 9.

91. Powell Singular Function [64] (Continuous, Differentiable, Non-Separable Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -4 ≤ x i ≤ 5. The global minima is located at x ∗ = f (3 , -1 , 0 , 1 , · · · , 3 , -1 , 0 , 1), f ( x ∗ ) = 0.

92. Powell Singular 2 Function [35] (Continuous, Differentiable, Non-Separable Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -4 ≤ x i ≤ 5. The global minimum is f ( x ∗ ) = 0.

93. Powell Sum Function [69] (Continuous, Differentiable, Separable Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -1 ≤ x i ≤ 1. The global minimum is f ( x ∗ ) = 0.

94. Price 1 Function [67] (Continuous, Non-Differentiable, Separable Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500. The global minimum are located at x ∗ = f ( {-5 , -5 } , {-5 , 5 } , { 5 , -5 } , { 5 , 5 } ), f ( x ∗ ) = 0.

95. Price 2 Function [67] (Continuous, Differentiable, Non-Separable Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is located at x ∗ = f (0 · · · 0), f ( x ∗ ) = 0 . 9.

96. Price 3 Function [67] (Continuous, Differentiable, Non-Separable Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500. The global minimum are located at x ∗ = f ( {-5 , -5 } , {-5 , 5 } , { 5 , -5 } , { 5 , 5 } ), f ( x ∗ ) = 0.

97. Price 4 Function [67] (Continuous, Differentiable, Non-Separable Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500. The three global minima are located at x ∗ = f ( { 0 , 0 } , { 2 , 4 } , { 1 . 464 , -2 . 506 } ), f ( x ∗ ) = 0.

98. Qing Function [68] (Continuous, Differentiable, Separable Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500. The global minima are located at x ∗ = f ( ± √ i ), f ( x ∗ ) = 0.

99. Quadratic Function (Continuous, Differentiable, Non-Separable, Non-Scalable)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is located at x ∗ = f (0 . 19388 , 0 . 48513), f ( x ∗ ) = -3873 . 7243.

100. Quartic Function [81] (Continuous, Differentiable, Separable, Scalable)

<!-- formula-not-decoded -->

subject to -1 . 28 ≤ x i ≤ 1 . 28. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

101. Quintic Function [58](Continuous, Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is located at x ∗ = f (-1 or 2), f ( x ∗ ) = 0.

102. Rana Function [66] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500, where t 1 = √ ‖ x i +1 + x i +1 ‖ and t 2 = √ ‖ x i +1 -x i +1 ‖ .

## 103. Ripple 1 Function (Non-separable)

<!-- formula-not-decoded -->

subject to 0 ≤ x i ≤ 1. It has one global minimum and 252004 local minima. The global form of the function consists of 25 holes, which forms a 5 × 5 regular grid. Additionally, the whole function landscape is full of small ripples caused by high frequency cosine function which creates a large number of local minima.

## 104. Ripple 25 Function (Non-separable)

<!-- formula-not-decoded -->

subject to 0 ≤ x i ≤ 1. It has one global form of the Ripple-1 function without any ripples due to absence of cosine term.

## 105. Rosenbrock Function [73] (Continuous, Differentiable, Non-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -30 ≤ x i ≤ 30. The global minima is located at x ∗ = f (1 , · · · , 1), f ( x ∗ ) = 0.

## 106. Rosenbrock Modified Function (Continuous, Differentiable, Non-Separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

subject to -2 ≤ x i ≤ 2. In this function, a Gaussian bump at ( -1 , 1) is added, which causes a local minimum at (1 , 1) and global minimum is located at x ∗ = f ( -1 , -1), f ( x ∗ ) = 0. This modification makes it a difficult to optimize because local minimum basin is larger than the global minimum basin.

## 107. Rotated Ellipse Function (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0.

## 108. Rotated Ellipse 2 Function [66] (Continuous, Differentiable, Non-Separable, NonScalable, Unimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0. s

109. Rump Function [51] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0.

## 110. Salomon Function [74] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0.

## 111. Sargan Function [29] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

/negationslash

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 112. Scahffer 1 Function [59] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0.

## 113. Scahffer 2 Function [59] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0.

## 114. Scahffer 3 Function [59] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is located at x ∗ = f (0 , 1 . 253115), f ( x ∗ ) = 0 . 00156685.

## 115. Scahffer 4 Function [59] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is located at x ∗ = f (0 , 1 . 253115), f ( x ∗ ) = 0 . 292579.

116. Schmidt Vetters Function [50] (Continuous, Differentiable, Non-Separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

The global minimum is located at x ∗ = f (0 . 78547 , 0 . 78547 , 0 . 78547), f ( x ∗ ) = 3.

117. Schumer Steiglitz Function [75] (Continuous, Differentiable, Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

The global minimum is located at x ∗ = f (0 , . . . , 0), f ( x ∗ ) = 0.

## 118. Schwefel Function [77] (Continuous, Differentiable, Partially-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

where α ≥ 0, subject to -100 ≤ x i ≤ 100. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 119. Schwefel 1.2 Function [77] (Continuous, Differentiable, Non-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 120. Schwefel 2.4 Function [77] (Continuous, Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x i ≤ 10. The global minima is located at x ∗ = f (1 , · · · , 1), f ( x ∗ ) = 0.

121. Schwefel 2.6 Function [77] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minima is located at x ∗ = f (1 , 3), f ( x ∗ ) = 0.

122. Schwefel 2.20 Function [77] (Continuous, Non-Differentiable, Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

123. Schwefel 2.21 Function [77] (Continuous, Non-Differentiable, Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 124. Schwefel 2.22 Function [77] (Continuous, Differentiable, Non-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 125. Schwefel 2.23 Function [77] (Continuous, Differentiable, Non-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 126. Schwefel 2.23 Function [77] (Continuous, Differentiable, Non-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 127. Schwefel 2.25 Function [77] (Continuous, Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x i ≤ 10. The global minima is located at x ∗ = f (1 , · · · , 1), f ( x ∗ ) = 0.

## 128. Schwefel 2.26 Function [77] (Continuous, Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500. The global minimum is located at x ∗ = ± [ π (0 . 5 + k )] 2 , f ( x ∗ ) = -418 . 983.

## 129. Schwefel 2.36 Function [77] (Continuous, Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x i ≤ 500. The global minimum is located at x ∗ = f (12 , · · · , 12), f ( x ∗ ) = -3456.

130. Shekel 5 [62] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

subject to 0 ≤ x j ≤ 10. The global minima is located at x ∗ = f (4 , 4 , 4 , 4), f ( x ∗ ) ≈ -10 . 1499.

## 131. Shekel 7 [62] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

subject to 0 ≤ x j ≤ 10. The global minima is located at x ∗ = f (4 , 4 , 4 , 4), f ( x ∗ ) ≈ -10 . 3999.

## 132. Shekel 10 [62] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

subject to 0 ≤ x j ≤ 10. The global minima is located at x ∗ = f (4 , 4 , 4 , 4), f ( x ∗ ) ≈ -10 . 5319.

## 133. Shubert Function [44] (Continuous, Differentiable, Separable?, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

{-

.

.

7

1

.

5

,

0835

.

4

,

4828

,

,

4251

8580

subject to -10 ≤ x i ≤ 10, i ∈ 1 , 2 , · · · , n . The 18 global minima are located at x ∗ = f ( {-7 . 0835 , 4 . 8580 } , {-7 . 0835 , -7 . 7083 } ,

{-

-

{

{-

0

1

.

4251

,

{-

{-

4

.

}

{

}

-

.

7

7083

7

,

.

7083

0

8580

7

{-

{-

,

.

0835

7

,

.

7083

,

5

.

4828

-

1

.

5

,

4251

}

.

4828

}

,

,

,

.

0

.

8003

,

1

.

4251

}

,

8003

,

8003

.

7

,

.

-

-

0835

0

}

,

.

8003

}

,

-

7

.

7083

}

,

0

,

,

4

.

8580

4251

5

1

.

.

,

4828

.

8003

,

{-

-

}

}

{

{-

4

.

8580

,

7

.

0835

}

,

{-

}

{ f ( x ∗ ) /similarequal 186 . 7309.

<!-- formula-not-decoded -->

## 134. Shubert 3 Function [3] (Continuous, Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is f ( x ∗ ) /similarequal 29 . 6733337 with multiple solutions.

## 135. Shubert 4 Function [3] (Continuous, Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is f ( x ∗ ) /similarequal 25 . 740858 with multiple solutions.

## 136. Schaffer F6 Function [76] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minimum is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 137. Sphere Function [75] (Continuous, Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x i ≤ 10. The global minima is located x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 138. Step Function (Discontinuous, Non-Differentiable, Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minima is located x ∗ = f (0 , · · · , 0) = 0, f ( x ∗ ) = 0.

## 139. Step 2 Function [9] (Discontinuous, Non-Differentiable, Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minima is located x ∗ = f (0 . 5 , · · · , 0 . 5) = 0, f ( x ∗ ) = 0.

## 140. Step 3 Function (Discontinuous, Non-Differentiable, Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minima is located x ∗ = f (0 , · · · , 0) = 0, f ( x ∗ ) = 0.

141. Stepint Function (Discontinuous, Non-Differentiable, Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -5 . 12 ≤ x i ≤ 5 . 12. The global minima is located x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

142. Streched V Sine Wave Function [76] (Continuous, Differentiable, Non-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is located x ∗ = f (0 , 0), f ( x ∗ ) = 0.

143. Sum Squares Function [43] (Continuous, Differentiable, Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minima is located x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

144. Styblinski-Tang Function [80] (Continuous, Differentiable, Non-Separable, NonScalable, Multimodal)

<!-- formula-not-decoded -->

subject to -5 ≤ x i ≤ 5. The global minimum is located x ∗ = f ( -2 . 903534 , -2 . 903534), f ( x ∗ ) = -78 . 332.

145. Table 1 / Holder Table 1 Function [58] (Continuous, Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10.

The four global minima are located at x ∗ = f ( ± 9 . 646168, ± 9 . 646168), f ( x ∗ ) = -26 . 920336.

146. Table 2 / Holder Table 2 Function [58] (Continuous, Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10.

The four global minima are located at x ∗ = f ( ± 8 . 055023472141116, ± 9 . 664590028909654), f ( x ∗ ) = -19 . 20850.

147. Table 3 / Carrom Table Function [58] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10.

The four global minima are located at x ∗ = f ( ± 9 . 646157266348881, ± 9 . 646134286497169), f ( x ∗ ) = -24 . 1568155.

148. Testtube Holder Function [58] (Continuous, Differentiable, Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The two global minima are located at x ∗ = f ( ± π/ 2 , 0), f ( x ∗ ) = -10 . 872300.

149. Trecanni Function [29] (Continuous, Differentiable, Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -5 ≤ x i ≤ 5. The two global minima are located at x ∗ = f ( { 0 , 0 } , {-2 , 0 } ), f ( x ∗ ) = 0.

150. Trid 6 Function [43] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -6 2 ≤ x i ≤ 6 2 . The global minima is located at f ( x ∗ ) = -50.

151. Trid 10 Function [43] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100. The global minima is located at f ( x ∗ ) = -200.

152. Trefethen Function [3] (Continuous, Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is located at x ∗ = f ( -0 . 024403 , 0 . 210612), f ( x ∗ ) = -3 . 30686865.

153. Trigonometric 1 Function [29] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x i ≤ pi . The global minimum is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0

154. Trigonometric 2 Function [35] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ x i ≤ 500. The global minimum is located at x ∗ = f (0 . 9 , · · · , 0 . 9), f ( x ∗ ) = 1

155. Tripod Function [69] (Discontinuous, Non-Differentiable, Non-Separable, Non-Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -100 ≤ x i ≤ 100, where p ( x ) = 1 for x ≥ 0. The global minimum is located at x ∗ = f (0 , -50), f ( x ∗ ) = 0.

156. Ursem 1 Function [72] (Separable)

<!-- formula-not-decoded -->

subject to -2 . 5 ≤ x 1 ≤ 3 and -2 ≤ x 2 ≤ 2, and has single global and local minima.

## 157. Ursem 3 Function [72](Non-separable)

<!-- formula-not-decoded -->

subject to -2 ≤ x 1 ≤ 2 and -1 . 5 ≤ x 2 ≤ 1 . 5, and has single global minimum and four regularly spaced local minima positioned in a direct line, such that global minimum is in the middle.

## 158. Ursem 4 Function [72] (Non-separable)

<!-- formula-not-decoded -->

subject to -2 ≤ x i ≤ 2, and has single global minimum positioned at the middle and four local minima at the corners of the search space.

## 159. Ursem Waves Function [72](Non-separable)

<!-- formula-not-decoded -->

subject to -0 . 9 ≤ x 1 ≤ 1 . 2 and -1 . 2 ≤ x 2 ≤ 1 . 2, and has single global minimum and nine irregularly spaced local minima in the search space.

## 160. Venter Sobiezcczanski-Sobieski Function [10] (Continuous, Differentiable, Separable, Non-Scalable)

<!-- formula-not-decoded -->

subject to -50 ≤ x i ≤ 50. The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = -400.

## 161. Watson Function [77] (Continuous, Differentiable, Non-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to | x i | ≤ 10, where the coefficient a i = i/ 29 . 0. The global minimum is located at x ∗ = f ( -0 . 0158 , 1 . 012 , -0 . 2329 , 1 . 260 , -1 . 513 , 0 . 9928), f ( x ∗ ) = 0 . 002288.

162. Wayburn Seader 1 Function [85] (Continuous, Differentiable, Non-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

The global minimum is located at x ∗ = f { (1 , 2) , (1 . 597 , 0 . 806) } , f ( x ∗ ) = 0.

163. Wayburn Seader 2 Function [85] (Continuous, Differentiable, Non-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ 500. The global minimum is located at x ∗ = f { (0 . 2 , 1) , (0 . 425 , 1) } , f ( x ∗ ) = 0.

164. Wayburn Seader 3 Function [85] (Continuous, Differentiable, Non-Separable, Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -500 ≤ 500. The global minimum is located at x ∗ = f (5 . 611 , 6 . 187), f ( x ∗ ) = 21 . 35.

165. W / Wavy Function [23] (Continuous, Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -π ≤ x i ≤ π . The global minimum is located at x ∗ = f (0 , 0), f ( x ∗ ) = 0. The number of local minima is kn and ( k +1) n for odd and even k respectively. For D = 2 and k = 10, there are 121 local minima.

166. Weierstrass Function [82](Continuous, Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -0 . 5 ≤ x i ≤ 0 . 5. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 167. Whitley Function [86] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

combines a very steep overall slope with a highly multimodal area around the global minimum located at x i = 1, where i = 1 , ..., D .

168. Wolfe Function [77] (Continuous, Differentiable, Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to 0 ≤ x i ≤ 2. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 169. Xin-She Yang (Function 1) (Separable)

This is a generic stochastic and non-smooth function proposed in [88, ? ].

<!-- formula-not-decoded -->

subject to -5 ≤ x i ≤ 5. The variable /epsilon1 i , ( i = 1 , 2 , · · · , D ) is a random variable uniformly distributed in [0 , 1]. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 170. Xin-She Yang (Function 2) (Non-separable)

<!-- formula-not-decoded -->

subject to -2 π ≤ x i ≤ 2 π . The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

## 171. Xin-She Yang (Function 3) (Non-separable)

<!-- formula-not-decoded -->

subject to -20 ≤ x i ≤ 20. The global minima for m = 5 and β = 15 is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = -1.

## 172. Xin-She Yang (Function 4) (Non-separable)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = -1.

173. Zakharov Function [69] (Continuous, Differentiable, Non-Separable, Scalable, Multimodal)

<!-- formula-not-decoded -->

subject to -5 ≤ x i ≤ 10. The global minima is located at x ∗ = f (0 , · · · , 0), f ( x ∗ ) = 0.

174. Zettl Function [78] (Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -5 ≤ x i ≤ 10. The global minima is located at x ∗ = f ( -0 . 0299 , 0), f ( x ∗ ) = -0 . 003791.

175. Zirilli or Aluffi-Pentini's Function [4] (Continuous, Differentiable, Separable, Non-Scalable, Unimodal)

<!-- formula-not-decoded -->

subject to -10 ≤ x i ≤ 10. The global minimum is located at x ∗ = ( -1 . 0465 , 0), f ( x ∗ ) ≈ -0 . 3523.

## 4 Conclusions

Test functions are important to validate and compare optimization algorithms, especially newly developed algorithms. Here, we have attempted to provide the most comprehensive list of known benchmarks or test functions. However, it is may be possibly that we have missed some functions, but this is not intentional. This list is based on all the literature known to us by the time of writing. It can be expected that all these functions should be used for testing new optimization algorithms so as to provide a more complete view about the performance of any algorithms of interest.

## References

- [1] D. H. Ackley, 'A Connectionist Machine for Genetic Hill-Climbing,' Kluwer, 1987.
- [2] C. S. Adjiman, S. Sallwig, C. A. Flouda, A. Neumaier, 'A Global Optimization Method, aBB for General Twice-Differentiable NLPs-1, Theoretical Advances,' Computers Chemical Engineering, vol. 22, no. 9, pp. 1137-1158, 1998.
- [3] E. P. Adorio, U. P. Dilman, 'MVF -Multivariate Test Function Library in C for Unconstrained Global Optimization Methods,' [Available Online]: http://www.geocities.ws/eadorio/mvf.pdf
- [4] M. M. Ali, C. Khompatraporn, Z. B. Zabinsky, 'A Numerical Evaluation of Several Stochastic Algorithms on Selected Continuous Global Optimization Test Problems,' Journal of Global Optimization, vol. 31, pp. 635-672, 2005.
- [5] N. Andrei, 'An Unconstrained Optimization Test Functions Collection,' Advanced Modeling and Optimization, vol. 10, no. 1, pp.147-161, 2008.
- [6] A. Auger, N. Hansen, N. Mauny, R. Ros, M. Schoenauer, 'Bio-Inspired Continuous Optimization: The Coming of Age,' Invited Lecture, IEEE Congress on Evolutionary Computation, NJ, USA, 2007.
- [7] B. M. Averick, R. G. Carter, J. J. Mor´ e, 'The MINIPACK-2 Test Problem Collection,' Mathematics and Computer Science Division, Agronne National Laboratory, Technical Memorandum No. 150, 1991.
- [8] B. M. Averick, R. G. Carter, J. J. Mor´ e, G. L. Xue, 'The MINIPACK-2 Test Problem Collection,' Mathematics and Computer Science Division, Agronne National Laboratory, Preprint MCS-P153-0692, 1992.
- [9] T. B¨ ack, H. P. Schwefel, 'An Overview of Evolutionary Algorithm for Parameter Optimization,' Evolutionary Computation, vol. 1, no. 1, pp. 1-23, 1993.
- [10] O. Begambre, J. E. Laier, 'A hybrid Particle Swarm Optimization - Simplex Algorithm (PSOS) for Structural Damage Identification,' Journal of Advances in Engineering Software, vol. 40, no. 9, pp. 883-891, 2009.
- [11] G. A. Bekey, M. T. Ung, 'A Comparative Evaluation of Two Global Search Algorithms,' IEEE Transaction on Systems, Man and Cybernetics, vol. 4, no. 1, pp. 112116, 1974.
- [12] H. Bersini, M. Dorigo, S. Langerman, 'Results of the First International Contest on Evolutionary Optimization,' IEEE International Conf. on Evolutionary Computation, Nagoya, Japan, pp. 611-615, 1996.
- [13] M. C. Biggs, 'A New Variable Metric Technique Taking Account of Non-Quadratic Behaviour of the Objective function,' IMA Journal of Applied Mathematics, vol. 8, no. 3, pp. 315-327, 1971.
- [14] I. O. Bohachevsky, M. E. Johnson, M. L. Stein, 'General Simulated Annealing for Function Optimization,' Technometrics, vol. 28, no. 3, pp. 209-217, 1986.

| [15]   | F. H. Branin Jr., 'Widely Convergent Method of Finding Multiple Solutions of Simul- taneous Nonlinear Equations,' IBM Journal of Research and Development, vol. 16, no. 5, pp. 504-522, 1972.                                                                     |
|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [16]   | D. O. Boyer, C. H. Martfnez, N. G. Pedrajas, 'Crossover Operator for Evolutionary Algorithms Based on Population Features', Journal of Artificial Intelligence Research, vol. 24, pp. 1-48, 2005.                                                                 |
| [17]   | Y. Brad, 'Comparison of Gradient Methods for the Solution of Nonlinear Parametric Estimation Problem,' SIAM Journal on Numerical Analysis, vol. 7, no. 1, pp. 157-186, 1970.                                                                                      |
| [18]   | The Cross-Entropy Toolbox http://www.maths.uq.edu.au/CEToolBox/                                                                                                                                                                                                   |
| [19]   | Y. Chen 'Computer Simulation of Electron Positron Annihila- tion Processes,' Technical Report SLAC-Report-646, Stanford Lin- ear Accelerator Center, Stanford University, 2003. [Available Online]: http://www.slac.stanford.edu/pubs/slacreports/slac-r-646.html |
| [20]   | C. J. Chung, R. G. Reynolds, 'CAEP: An Evolution-Based Tool for Real-Valued Func- tion Optimization Using Cultural Algorithms,' International Journal on Artificial In- telligence Tool, vol. 7, no. 3, pp. 239-291, 1998.                                        |
| [21]   | M. Clerc, 'The Swarm and the Queen, Towards a Deterministic and Adaptive Particle Swarm Optimization, ' IEEE Congress on Evolutionary Computation, Washington DC, USA, pp. 1951-1957, 1999.                                                                       |
| [22]   | A. Corana, M. Marchesi, C. Martini, S. Ridella, 'Minimizing Multimodal Functions of Continuous Variables with Simulated Annealing Algorithms,' ACM Transactions on Mathematical Software, vol. 13, no. 3, pp. 262-280, 1987.                                      |
| [23]   | P. courrieu, 'The Hyperbell Algorithm for Global Optimization: A Random Walk Using Cauchy Densities,' Journal of Global Optimization, vol. 10, no. 1, pp. 111-133, 1997.                                                                                          |
| [24]   | E. E. Cragg, A. V. Levy, 'Study on Supermemory Gradient Method for the Minimiza- tion of Functions,' Journal of Optimization Theory and Applications, vol. 4, no. 3, pp. 191-205, 1969.                                                                           |
| [25]   | T. Csendes, D. Ratz, 'Subdivision Direction Selection in Interval Methods for Global Optimization,' SIAM Journal on Numerical Analysis, vol. 34, no. 3, pp. 922-938.                                                                                              |
| [26]   | N. Damavandi, S. Safavi-Naeini, 'A Hybrid Evolutionary Programming Method for Circuit Optimization,' IEEE Transaction on Circuit and Systems I, vol. 52, no. 5, pp. 902-910, 2005.                                                                                |
| [27]   | N. deVillers, D. Glasser, 'A Continuation Method for Nonlinear Regression,' SIAM Journal on Numerical Analysis, vol. 18, no. 6, pp. 1139-1154, 1981.                                                                                                              |
| [28]   | L. C. W. Dixon, R. C. Price, 'The Truncated Newton Method for Sparse Unconstrained Optimisation Using Automatic Differentiation,' Journal of Optimization Theory and Applications, vol. 60, no. 2, pp. 261-275, 1989.                                             |

- [29] L. C. W. Dixon, G. P. Szeg¨ o (eds.), 'Towards Global Optimization 2,' Elsevier, 1978.
- [30] R. A. El-Attar, M. Vidyasagar, S. R. K. Dutta, 'An Algorithm for II-norm Minimization With Application to Nonlinear II-approximation,' SIAM Journal on Numverical Analysis, vol. 16, no. 1, pp. 70-86, 1979.
- [31] 'Two Algorithms for Global Optimization of General NLP Problems,' International Journal on Numerical Methods in Engineering, vol. 39, no. 19, pp. 3305-3325, 1996.
- [32] R. Fletcher, M. J. D. Powell, 'A Rapidly Convergent Descent Method for Minimzation,' Computer Journal, vol. 62, no. 2, pp. 163-168, 1963. [Available Online]: http://galton.uchicago.edu/ ~ lekheng/courses/302/classics/fletcher-powell.pdf
- [33] C. A. Flouda, P. M. Pardalos, C. S. Adjiman, W. R. Esposito, Z. H. G¨ um¨ us, S. T. Harding, J. L. Klepeis, C. A. Meyer, and C. A. Schweiger, 'Handbook of Test Problems in Local and Global Optimization,' Kluwer, Boston, 1999.
- [34] C. Fraley, 'Software Performances on Nonlinear Least-Squares Problems,' Technical Report no. STAN-CS-89-1244, Department of Computer Science, Stanford University, 1989. [Available Online]: http://www.dtic.mil/dtic/tr/fulltext/u2/a204526.pdf
- [35] M. C. Fu, J. Hu, S. I. Marcus, 'Model-Based Randomized Methods for Global Optimization,' Proc. 17th International Symp. Mathematical Theory Networks Systems, Kyoto, Japan, pp. 355-365, 2006.
- [36] GAMS World, GLOBAL Library, [Available Online]: http://www.gamsworld.org/global/globallib.html.
- [37] GEATbx - The Genetic and Evolutionary Algorithm Toolbox for Matlab, [Available Online]: http://www.geatbx.com/
- [38] A. A. Goldstein, J. F. Price, 'On Descent from Local Minima,' Mathematics and Comptutaion, vol. 25, no. 115, pp. 569-574, 1971.
- [39] V. S. Gordon, D. Whitley, Serial and Parallel Genetic Algoritms as Function Optimizers,' In S. Forrest (Eds), 5th Intl. Conf. on Genetic Algorithms, pp. 177-183, Morgan Kaufmann.
- [40] A. O. Griewank, 'Generalized Descent for Global Optimization,' Journal of Optimization Theory and Applications, vol. 34, no. 1, pp. 11-39, 1981.
- [41] N. I. M. Gould, D. Orban, and P. L. Toint, 'CUTEr, A Constrained and Un-constrained Testing Environment, Revisited,' [Available Online]: http://cuter.rl.ac.uk/cuter-www/problems.html.
- [42] J. K. Hartman, 'Some Experiments in Global Optimization,' [Available Online]: http://ia701505.us.archive.org/9/items/someexperimentsi00hart/someexperimentsi00hart.pdf
- [43] A.-R. Hedar, 'Global Optimization Test Problems,' [Available Online]: http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar\_files/TestGO.htm.
- [44] J. P. Hennart (ed.), 'Numerical Analysis,' Proc. 3rd AS Workshop, Lecture Notes in Mathematics, vol. 90, Springer, 1982.

- [45] D. M. Himmelblau, 'Applied Nonlinear Programming,' McGraw-Hill, 1972.
- [46] R. I. Jennrich, P. F. Sampson, 'Application of Stepwise Regression to Non-Linear estimation,' Techometrics, vol. 10, no. 1, pp. 63-72, 1968. http://www.jstor.org/discover/10.2307/1266224?uid=3737864&amp;uid=2129&amp;uid=2&amp;uid=70&amp;uid=4&amp;sid=
- [47] A. D. Junior, R. S. Silva, K. C. Mundim, and L. E. Dardenne, 'Performance and Parameterization of the Algorithm Simplified Generalized Simulated Annealing,' Genet. Mol. Biol., vol. 27, no. 4, pp. 616-622, 2004. [Available Online]: http://www.scielo.br/scielo.php?script=sci\_arttext&amp;pid=S1415-47572004000400024&amp;lng=en&amp;nrm= ISSN 1415-4757. http://dx.doi.org/10.1590/S1415-47572004000400024.
- [48] Test Problems for Global Optimization, http://www2.imm.dtu.dk/ ~ kajm/Test\_ex\_forms/test\_ex.html
- [49] A. Lavi, T. P. Vogel (eds), 'Recent Advances in Optimization Techniques,' John Wliley &amp; Sons, 1966.
- [50] F. A. Lootsma (ed.), ' Numerical Methods for Non-Linear Optimization,' Academic Press, 1972.
- [51] R. E. Moore, 'Reliability in Computing,' Academic Press, 1998.
- [52] J. J. Mor´ e, B. S. Garbow, K. E. Hillstrom, 'Testing Unconstrained Optimization Software,', ACM Trans. on Mathematical Software, vol. 7, pp. 17-41, 1981.
- [53] S. K. Mishra, 'Performance of Differential Evolution and Particle Swarm Methods on Some Relatively Harder Multi-modal Benchmark Functions,' [Available Online]: http://mpra.ub.uni-muenchen.de/449/
- [54] S. K. Mishra, 'Performance of the Barter, the Differential Evolution and the Simulated Annealing Methods of Global Pptimization On Some New and Some Old Test Functions,' [Available Online]: http://www.ssrn.com/abstract=941630
- [55] S. K. Mishra, 'Repulsive Particle Swarm Method On Some Difficult Test Problems of Global Optimization,' [Available Online]: http://mpra.ub.uni-muenchen.de/1742/
- [56] S. K. Mishra, 'Performance of Repulsive Particle Swarm Method in Global Optimization of Some Important Test Functions: A Fortran Program,' [Available Online]: http://www.ssrn.com/abstract=924339
- [57] S. K. Mishra, 'Global Optimization by Particle Swarm Method: A Fortran Program,' Munich Research Papers in Economics, [Available Online]: http://mpra.ub.uni-muenchen.de/874/
- [58] S. K. Mishra, 'Global Optimization By Differential Evolution and Particle Swarm Methods: Evaluation On Some Benchmark Functions,' Munich Research Papers in Economics, [Available Online]: http://mpra.ub.uni-muenchen.de/1005/
- [59] S. K. Mishra, 'Some New Test Functions For Global Optimization And Performance of Repulsive Particle Swarm Method,' [Available Online]: http://mpra.ub.uni-muenchen.de/2718/

| [60]   | C. Muntenau, lutionayr                                                                                                                                                                                                                                                                                                                 | V. Framework:                                                                                                                                                                                                                                                                                                                          | Lazarescu, The                                                                                                                                                                                                                                                                                                                         | 'Global Adaptive                                                                                                                                                                                                                                                                                                                       | Using a Reservoir Genetic                                                                                                                                                                                                                                                                                                              | New                                                                                                                                                                                                                                                                                                                                    | Search Evo- Algo-                                                                                                                                                                                                                                                                                                                      |                                                                                                                                                                                                                                                                                                                                        |
|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [61]   | A. Neumaier, 'COCONUT Benchmark,' [Available                                                                                                                                                                                                                                                                                           | A. Neumaier, 'COCONUT Benchmark,' [Available                                                                                                                                                                                                                                                                                           | A. Neumaier, 'COCONUT Benchmark,' [Available                                                                                                                                                                                                                                                                                           | A. Neumaier, 'COCONUT Benchmark,' [Available                                                                                                                                                                                                                                                                                           | A. Neumaier, 'COCONUT Benchmark,' [Available                                                                                                                                                                                                                                                                                           | A. Neumaier, 'COCONUT Benchmark,' [Available                                                                                                                                                                                                                                                                                           | Online]:                                                                                                                                                                                                                                                                                                                               | A. Neumaier, 'COCONUT Benchmark,' [Available                                                                                                                                                                                                                                                                                           |
| [62]   | J. Opaˇ ci´ c, 'A Heuristic Method for Finding Most extrema of a Nonlinear Functional,' IEEE Transactions on Systems, Man and Cybernetics,                                                                                                                                                                                             | J. Opaˇ ci´ c, 'A Heuristic Method for Finding Most extrema of a Nonlinear Functional,' IEEE Transactions on Systems, Man and Cybernetics,                                                                                                                                                                                             | J. Opaˇ ci´ c, 'A Heuristic Method for Finding Most extrema of a Nonlinear Functional,' IEEE Transactions on Systems, Man and Cybernetics,                                                                                                                                                                                             | J. Opaˇ ci´ c, 'A Heuristic Method for Finding Most extrema of a Nonlinear Functional,' IEEE Transactions on Systems, Man and Cybernetics,                                                                                                                                                                                             | J. Opaˇ ci´ c, 'A Heuristic Method for Finding Most extrema of a Nonlinear Functional,' IEEE Transactions on Systems, Man and Cybernetics,                                                                                                                                                                                             | J. Opaˇ ci´ c, 'A Heuristic Method for Finding Most extrema of a Nonlinear Functional,' IEEE Transactions on Systems, Man and Cybernetics,                                                                                                                                                                                             | vol. 3, no. 1, pp. 102-107, 1973.                                                                                                                                                                                                                                                                                                      | J. Opaˇ ci´ c, 'A Heuristic Method for Finding Most extrema of a Nonlinear Functional,' IEEE Transactions on Systems, Man and Cybernetics,                                                                                                                                                                                             |
| [63]   | J. D. Pint´ er, 'Global Optimization in Action: Continuous and Lipschitz Optimization Algorithms, Implementations and Applications,' Kluwer, 1996.                                                                                                                                                                                     | J. D. Pint´ er, 'Global Optimization in Action: Continuous and Lipschitz Optimization Algorithms, Implementations and Applications,' Kluwer, 1996.                                                                                                                                                                                     | J. D. Pint´ er, 'Global Optimization in Action: Continuous and Lipschitz Optimization Algorithms, Implementations and Applications,' Kluwer, 1996.                                                                                                                                                                                     | J. D. Pint´ er, 'Global Optimization in Action: Continuous and Lipschitz Optimization Algorithms, Implementations and Applications,' Kluwer, 1996.                                                                                                                                                                                     | J. D. Pint´ er, 'Global Optimization in Action: Continuous and Lipschitz Optimization Algorithms, Implementations and Applications,' Kluwer, 1996.                                                                                                                                                                                     | J. D. Pint´ er, 'Global Optimization in Action: Continuous and Lipschitz Optimization Algorithms, Implementations and Applications,' Kluwer, 1996.                                                                                                                                                                                     | J. D. Pint´ er, 'Global Optimization in Action: Continuous and Lipschitz Optimization Algorithms, Implementations and Applications,' Kluwer, 1996.                                                                                                                                                                                     | J. D. Pint´ er, 'Global Optimization in Action: Continuous and Lipschitz Optimization Algorithms, Implementations and Applications,' Kluwer, 1996.                                                                                                                                                                                     |
| [64]   | M. J. D. Powell, 'An Iterative Method for Finding Stationary Values of of Several Variables,' Computer Journal, vol. 5, no. 2, pp. 147-151, Online]: http://comjnl.oxfordjournals.org/content/5/2/147.full.pdf                                                                                                                         | M. J. D. Powell, 'An Iterative Method for Finding Stationary Values of of Several Variables,' Computer Journal, vol. 5, no. 2, pp. 147-151, Online]: http://comjnl.oxfordjournals.org/content/5/2/147.full.pdf                                                                                                                         | M. J. D. Powell, 'An Iterative Method for Finding Stationary Values of of Several Variables,' Computer Journal, vol. 5, no. 2, pp. 147-151, Online]: http://comjnl.oxfordjournals.org/content/5/2/147.full.pdf                                                                                                                         | M. J. D. Powell, 'An Iterative Method for Finding Stationary Values of of Several Variables,' Computer Journal, vol. 5, no. 2, pp. 147-151, Online]: http://comjnl.oxfordjournals.org/content/5/2/147.full.pdf                                                                                                                         | M. J. D. Powell, 'An Iterative Method for Finding Stationary Values of of Several Variables,' Computer Journal, vol. 5, no. 2, pp. 147-151, Online]: http://comjnl.oxfordjournals.org/content/5/2/147.full.pdf                                                                                                                         | M. J. D. Powell, 'An Iterative Method for Finding Stationary Values of of Several Variables,' Computer Journal, vol. 5, no. 2, pp. 147-151, Online]: http://comjnl.oxfordjournals.org/content/5/2/147.full.pdf                                                                                                                         | a Function 1962. [Available                                                                                                                                                                                                                                                                                                            | M. J. D. Powell, 'An Iterative Method for Finding Stationary Values of of Several Variables,' Computer Journal, vol. 5, no. 2, pp. 147-151, Online]: http://comjnl.oxfordjournals.org/content/5/2/147.full.pdf                                                                                                                         |
| [65]   | M. J. D. Powell, 'An Efficient Method for Finding the Minimum of a Several Variables Without Calculating Derivatives,' Computer Journal, vol. pp. 155-162, 1964.                                                                                                                                                                       | M. J. D. Powell, 'An Efficient Method for Finding the Minimum of a Several Variables Without Calculating Derivatives,' Computer Journal, vol. pp. 155-162, 1964.                                                                                                                                                                       | M. J. D. Powell, 'An Efficient Method for Finding the Minimum of a Several Variables Without Calculating Derivatives,' Computer Journal, vol. pp. 155-162, 1964.                                                                                                                                                                       | M. J. D. Powell, 'An Efficient Method for Finding the Minimum of a Several Variables Without Calculating Derivatives,' Computer Journal, vol. pp. 155-162, 1964.                                                                                                                                                                       | M. J. D. Powell, 'An Efficient Method for Finding the Minimum of a Several Variables Without Calculating Derivatives,' Computer Journal, vol. pp. 155-162, 1964.                                                                                                                                                                       | M. J. D. Powell, 'An Efficient Method for Finding the Minimum of a Several Variables Without Calculating Derivatives,' Computer Journal, vol. pp. 155-162, 1964.                                                                                                                                                                       | Function for 7, no. 2,                                                                                                                                                                                                                                                                                                                 | M. J. D. Powell, 'An Efficient Method for Finding the Minimum of a Several Variables Without Calculating Derivatives,' Computer Journal, vol. pp. 155-162, 1964.                                                                                                                                                                       |
| [66]   | K. V. Price, R. M. Storn, J. A. Lampinen, 'Differential Evolution: proach to Global Optimization,' Springer, 2005.                                                                                                                                                                                                                     | K. V. Price, R. M. Storn, J. A. Lampinen, 'Differential Evolution: proach to Global Optimization,' Springer, 2005.                                                                                                                                                                                                                     | K. V. Price, R. M. Storn, J. A. Lampinen, 'Differential Evolution: proach to Global Optimization,' Springer, 2005.                                                                                                                                                                                                                     | K. V. Price, R. M. Storn, J. A. Lampinen, 'Differential Evolution: proach to Global Optimization,' Springer, 2005.                                                                                                                                                                                                                     | K. V. Price, R. M. Storn, J. A. Lampinen, 'Differential Evolution: proach to Global Optimization,' Springer, 2005.                                                                                                                                                                                                                     | K. V. Price, R. M. Storn, J. A. Lampinen, 'Differential Evolution: proach to Global Optimization,' Springer, 2005.                                                                                                                                                                                                                     | A Practical Ap-                                                                                                                                                                                                                                                                                                                        | K. V. Price, R. M. Storn, J. A. Lampinen, 'Differential Evolution: proach to Global Optimization,' Springer, 2005.                                                                                                                                                                                                                     |
| [68]   | W. L. Price, 'A Controlled Random Search Procedure for Global Optimisa- tion,' Computer journal, vol. 20, no. 4, pp. 367-370, 1977. [Available Online]: http://comjnl.oxfordjournals.org/content/20/4/367.full.pdf A. Qing, 'Dynamic Differential Evolution Strategy and Applications in                                               | W. L. Price, 'A Controlled Random Search Procedure for Global Optimisa- tion,' Computer journal, vol. 20, no. 4, pp. 367-370, 1977. [Available Online]: http://comjnl.oxfordjournals.org/content/20/4/367.full.pdf A. Qing, 'Dynamic Differential Evolution Strategy and Applications in                                               | W. L. Price, 'A Controlled Random Search Procedure for Global Optimisa- tion,' Computer journal, vol. 20, no. 4, pp. 367-370, 1977. [Available Online]: http://comjnl.oxfordjournals.org/content/20/4/367.full.pdf A. Qing, 'Dynamic Differential Evolution Strategy and Applications in                                               | W. L. Price, 'A Controlled Random Search Procedure for Global Optimisa- tion,' Computer journal, vol. 20, no. 4, pp. 367-370, 1977. [Available Online]: http://comjnl.oxfordjournals.org/content/20/4/367.full.pdf A. Qing, 'Dynamic Differential Evolution Strategy and Applications in                                               | W. L. Price, 'A Controlled Random Search Procedure for Global Optimisa- tion,' Computer journal, vol. 20, no. 4, pp. 367-370, 1977. [Available Online]: http://comjnl.oxfordjournals.org/content/20/4/367.full.pdf A. Qing, 'Dynamic Differential Evolution Strategy and Applications in                                               | W. L. Price, 'A Controlled Random Search Procedure for Global Optimisa- tion,' Computer journal, vol. 20, no. 4, pp. 367-370, 1977. [Available Online]: http://comjnl.oxfordjournals.org/content/20/4/367.full.pdf A. Qing, 'Dynamic Differential Evolution Strategy and Applications in                                               | Electromag- remote Sens-                                                                                                                                                                                                                                                                                                               | W. L. Price, 'A Controlled Random Search Procedure for Global Optimisa- tion,' Computer journal, vol. 20, no. 4, pp. 367-370, 1977. [Available Online]: http://comjnl.oxfordjournals.org/content/20/4/367.full.pdf A. Qing, 'Dynamic Differential Evolution Strategy and Applications in                                               |
|        | netic Inverse Scattering Problems,' IEEE Transactions on Geoscience and ing, vol. 44, no. 1, pp. 116-125, 2006. S. Rahnamyan, H. R. Tizhoosh, N. M. M. Salama, 'A Novel Population Method for Accelerating Evolutionary Algorithms,' Computers and Applications,                                                                       | netic Inverse Scattering Problems,' IEEE Transactions on Geoscience and ing, vol. 44, no. 1, pp. 116-125, 2006. S. Rahnamyan, H. R. Tizhoosh, N. M. M. Salama, 'A Novel Population Method for Accelerating Evolutionary Algorithms,' Computers and Applications,                                                                       | netic Inverse Scattering Problems,' IEEE Transactions on Geoscience and ing, vol. 44, no. 1, pp. 116-125, 2006. S. Rahnamyan, H. R. Tizhoosh, N. M. M. Salama, 'A Novel Population Method for Accelerating Evolutionary Algorithms,' Computers and Applications,                                                                       | netic Inverse Scattering Problems,' IEEE Transactions on Geoscience and ing, vol. 44, no. 1, pp. 116-125, 2006. S. Rahnamyan, H. R. Tizhoosh, N. M. M. Salama, 'A Novel Population Method for Accelerating Evolutionary Algorithms,' Computers and Applications,                                                                       | netic Inverse Scattering Problems,' IEEE Transactions on Geoscience and ing, vol. 44, no. 1, pp. 116-125, 2006. S. Rahnamyan, H. R. Tizhoosh, N. M. M. Salama, 'A Novel Population Method for Accelerating Evolutionary Algorithms,' Computers and Applications,                                                                       | netic Inverse Scattering Problems,' IEEE Transactions on Geoscience and ing, vol. 44, no. 1, pp. 116-125, 2006. S. Rahnamyan, H. R. Tizhoosh, N. M. M. Salama, 'A Novel Population Method for Accelerating Evolutionary Algorithms,' Computers and Applications,                                                                       | Initialization Mathematics with                                                                                                                                                                                                                                                                                                        | netic Inverse Scattering Problems,' IEEE Transactions on Geoscience and ing, vol. 44, no. 1, pp. 116-125, 2006. S. Rahnamyan, H. R. Tizhoosh, N. M. M. Salama, 'A Novel Population Method for Accelerating Evolutionary Algorithms,' Computers and Applications,                                                                       |
| [69]   |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        | vol. 53, no. 10, pp. 1605-1614, 2007.                                                                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                                                                                                                        |
| [70]   |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        | S. Rahnamyan, H. R. Tizhoosh, N. M. M. Salama, 'Opposition-Based Differential Evolution (ODE) with Variable Jumping Rate,' IEEE Sympousim Foundations Com- putation Intelligence, Honolulu, HI, pp. 81-88, 2007.                                                                                                                       |                                                                                                                                                                                                                                                                                                                                        |
| [71]   | 2009. J. R¨ onkk¨ onen, 'Continuous Multimodal Global Optimization With Evolution-Based Methods,' PhD Thesis, Lappeenranta University of H. H. Rosenbrock, 'An Automatic Method for Finding the Greatest or a Function,' Computer Journal, vol. 3, no. 3, pp. 175-184, 1960. http://comjnl.oxfordjournals.org/content/3/3/175.full.pdf | 2009. J. R¨ onkk¨ onen, 'Continuous Multimodal Global Optimization With Evolution-Based Methods,' PhD Thesis, Lappeenranta University of H. H. Rosenbrock, 'An Automatic Method for Finding the Greatest or a Function,' Computer Journal, vol. 3, no. 3, pp. 175-184, 1960. http://comjnl.oxfordjournals.org/content/3/3/175.full.pdf | 2009. J. R¨ onkk¨ onen, 'Continuous Multimodal Global Optimization With Evolution-Based Methods,' PhD Thesis, Lappeenranta University of H. H. Rosenbrock, 'An Automatic Method for Finding the Greatest or a Function,' Computer Journal, vol. 3, no. 3, pp. 175-184, 1960. http://comjnl.oxfordjournals.org/content/3/3/175.full.pdf | 2009. J. R¨ onkk¨ onen, 'Continuous Multimodal Global Optimization With Evolution-Based Methods,' PhD Thesis, Lappeenranta University of H. H. Rosenbrock, 'An Automatic Method for Finding the Greatest or a Function,' Computer Journal, vol. 3, no. 3, pp. 175-184, 1960. http://comjnl.oxfordjournals.org/content/3/3/175.full.pdf | 2009. J. R¨ onkk¨ onen, 'Continuous Multimodal Global Optimization With Evolution-Based Methods,' PhD Thesis, Lappeenranta University of H. H. Rosenbrock, 'An Automatic Method for Finding the Greatest or a Function,' Computer Journal, vol. 3, no. 3, pp. 175-184, 1960. http://comjnl.oxfordjournals.org/content/3/3/175.full.pdf | 2009. J. R¨ onkk¨ onen, 'Continuous Multimodal Global Optimization With Evolution-Based Methods,' PhD Thesis, Lappeenranta University of H. H. Rosenbrock, 'An Automatic Method for Finding the Greatest or a Function,' Computer Journal, vol. 3, no. 3, pp. 175-184, 1960. http://comjnl.oxfordjournals.org/content/3/3/175.full.pdf | 2009. J. R¨ onkk¨ onen, 'Continuous Multimodal Global Optimization With Evolution-Based Methods,' PhD Thesis, Lappeenranta University of H. H. Rosenbrock, 'An Automatic Method for Finding the Greatest or a Function,' Computer Journal, vol. 3, no. 3, pp. 175-184, 1960. http://comjnl.oxfordjournals.org/content/3/3/175.full.pdf | 2009. J. R¨ onkk¨ onen, 'Continuous Multimodal Global Optimization With Evolution-Based Methods,' PhD Thesis, Lappeenranta University of H. H. Rosenbrock, 'An Automatic Method for Finding the Greatest or a Function,' Computer Journal, vol. 3, no. 3, pp. 175-184, 1960. http://comjnl.oxfordjournals.org/content/3/3/175.full.pdf |
| [72]   |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        | Differential Technology, 2009.                                                                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                                                                                        |
| [73]   |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        | least Value of [Available Online]:                                                                                                                                                                                                                                                                                                     |                                                                                                                                                                                                                                                                                                                                        |
| [74]   |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                                                                                                        | R. Salomon, 'Re-evaluating Genetic Algorithm Performance Under Corodinate Rota- tion of Benchmark Functions: A Survey of Some Theoretical and Practical Aspects of Genetic Algorithms,' BioSystems, vol. 39, no. 3, pp. 263-278, 1996.                                                                                                 |                                                                                                                                                                                                                                                                                                                                        |

| [75]   | M. A. Schumer, K. Steiglitz, 'Adaptive Step Size Random Search,' IEEE Transactions on Automatic Control. vol. 13, no. 3, pp. 270-276, 1968.                                                                                                                                                                                                    |
|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [76]   | J. D. Schaffer, R. A. Caruana, L. J. Eshelman, R. Das, 'A Study of Control Parameters Affecting Online Performance of Genetic Algorithms for Function Optimization,' Proc. 3rd International Conf. on Genetic Algorithms, George Mason Uni., pp. 51-60, 19889.                                                                                 |
| [77]   | H. P. Schwefel, 'Numerical Optimization for Computer Models,' John Wiley Sons, 1981.                                                                                                                                                                                                                                                           |
| [78]   | H. P. Schwefel, 'Evolution and Optimum Seeking,' John Wiley Sons, 1995.                                                                                                                                                                                                                                                                        |
| [79]   | D. F. Shanno, 'Conditioning of Quasi-Newton Methods for Function Minimization,' Mathematics of Computation, vol. 24, no. 111, pp. 647-656, 1970.                                                                                                                                                                                               |
| [80]   | Z. K. Silagadze, 'Finding Two-Dimesnional Peaks,' Physics of Particles and Nuclei Letters, vol. 4, no. 1, pp. 73-80, 2007.                                                                                                                                                                                                                     |
| [81]   | R. Storn, K. Price, 'Differntial Evolution - A Simple and Efficient Adaptive Scheme for Global Optimization over Continuous Spaces,' Technical Report no. TR-95-012, International Computer Science Institute, Berkeley, CA, 1996. [Available Online] : http://www1.icsi.berkeley.edu/ ~ storn/TR-95-012.pdf                                   |
| [82]   | P. N. Suganthan, N. Hansen, J. J. Liang, K. Deb, Y.-P. Chen, A. Auger, S. Tiwari, 'Problem Definitions and Evaluation Criteria for CEC 2005, Special Session on Real-Parameter Optimization,' Nanyang Techno- logical University (NTU), Singapore, Tech. Rep., 2005. [Available Online]: http://www.lri.fr/ ~ hansen/Tech-Report-May-30-05.pdf |
| [83]   | K. Tang, X. Yao, P. N. Suganthan, C. MacNish, Y.-P. Chen, C.-M. Chen, Z. Yang, 'Benchmark Functions for the CEC2008 Special Session and Competi- tion on Large Scale Global Optimization,', Tech. Rep., 2008. [Available Online]: http://nical.ustc.edu.cn/cec08ss.php                                                                         |
| [84]   | K. Tang, X. Li, P. N. Suganthan, Z. Yang, T. Weise, 'Bench- mark Functions for the CEC2010 Special Session and Competition on Large-Scale Global Optimization,' Tech. Rep., 2010. [Available Online]: http://sci2s.ugr.es/eamhco/cec2010_functions.pdf                                                                                         |
| [85]   | T. L. Wayburn, J. D. Seader, 'Homotopy Continuation Methods for Computer-Aided Process Design,' Computers and Chemical Engineering, vol. 11, no. 1, pp. 7-25, 1987.                                                                                                                                                                            |
| [86]   | , D. Whitley, K. Mathias, S. Rana, J. Dzubera, 'Evaluating Evolutionary Algorithms,' Artificial Intelligence, vol. 85, pp. 245-276, 1996.                                                                                                                                                                                                      |
| [87]   | P. H. Winston, 'Artificial Intelligence, 3 rd ed.' Addison-Wesley, 1992.                                                                                                                                                                                                                                                                       |
| [88]   | X. S. Yang, 'Test Problems in Optimization,' Engineering Optimization: An Intro- duction with Metaheuristic Applications John Wliey & Sons, 2010. [Available Online]: http://arxiv.org/abs/1008.0549                                                                                                                                           |
| [89]   | X. S. Yang, 'Firefly Algorithm, Stochastic Test Functions and Design Optimisa- tion,' Intl. J. Bio-Inspired Computation, vol. 2, no. 2, pp. 78-84. [Available Online]: http://arxiv.org/abs/1008.0549                                                                                                                                          |

- [90] X. Yao, Y. Liu, 'Fast Evolutionary Programming,' Proc. 5 th Conf. on Evolutionary Programming, 1996.