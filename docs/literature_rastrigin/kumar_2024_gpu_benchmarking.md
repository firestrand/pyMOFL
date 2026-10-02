## Benchmarking of GPU-optimized Quantum-Inspired Evolutionary Optimization Algorithm using Functional Analysis

Kandula Eswara Sai Kumar Principal Optimization Scientist

Supreeth B S

Quantum Algorithm Researcher

Rajas Dalvi

Quantum Optimization Researcher

Aman Mittal

HPC Researcher

Aakif Akhtar

Ferdin Don Bosco

Rut Lineswala

Abhishek Chopra

Quantum Algorithm Developer

Senior Computational Scientist

Chief Technology Officer

CEO &amp; Chief Scientific Officer

BosonQ Psi (BQP), New York, USA

eswara.sai@bosonqpsi.com

abhishek.chopra@bosonqpsi.com

Abstract -This article presents a comparative analysis of GPUparallelized implementations of the quantum-inspired evolutionary optimization (QIEO) approach and one of the well-known classical metaheuristic techniques - the genetic algorithm (GA). The study assesses the performance of both algorithms on highly non-linear, non-convex, and non-separable function optimization problems, viz., Ackley, Rosenbrock, and Rastrigin, that are representative of the complex real-world optimization problems. The performance of these algorithms is checked by varying the population sizes by keeping all other parameters constant and comparing the fitness value it reached along with the number of function evaluations they required for convergence. The results demonstrate that QIEO performs better for these functions than GA, by achieving the target fitness with fewer function evaluations and significantly reducing the total optimization timeapproximately three times for the Ackley function and four times for the Rosenbrock and Rastrigin functions. Furthermore, QIEO exhibits greater consistency across trials, with a steady convergence rate that leads to a more uniform number of function evaluations, highlighting its reliability in solving challenging optimization problems. The findings indicate that QIEO is a promising alternative to GA for these kind of functions.

Index Terms -Performance Analysis, Optimization, Accuracy, Quantum-inspired

## I. INTRODUCTION

Ant Colony Optimization [6], have become powerful tools for tackling such challenges. Inspired by natural processes, these algorithms efficiently find near-optimal solutions for a wide range of optimization problems in reasonable time frames [7], [8]. These algorithms have demonstrated considerable potential in solving optimization problems where exact algorithms are impractical. These methods offer efficiency and scalability for multiple objectives [9], and they can be readily adapted to various problem domains, including combinatorial, discrete, and continuous optimization [10]. They exhibit robustness to imperfect objective functions, which are often discontinuous, non-differentiable, and erratic [11], and can be implemented in distributed and parallel computing infrastructures [12]. However, challenges arise in these meta-heuristic algorithms from their dependence on extensive hyper-parameter tuning, which complicates the development of a general-purpose optimization method. Although these algorithms are intended to be general-purpose, their performance can be inconsistent, unreliable, and time-consuming [13], [14]. This inconsistency arises from the need to adjust numerous hyper-parameters and the inherent randomness in their processes, making it difficult to achieve predictable results in any given trial [15].

Traditional optimization methods, such as gradient descent, Newton-Raphson method, etc., are powerful but have notable drawbacks. Firstly, they are local in scope and often fail to find high-quality solutions in complex, non-convex landscapes. Secondly, these algorithms rely on gradient information throughout the search process, necessitating that the design space be continuous and smooth, which is often not the case in the real world [1]. Additionally, they can be computationally expensive, especially in high-dimensional spaces, due to the need for frequent gradient evaluations. When the design space is discrete, researchers have employed enumeration algorithms; however, as the number of design variables grows, the performance of these algorithms rapidly deteriorates as they suffer from the curse of dimensionality [2]

The utilization of meta-heuristics, such as Genetic Algorithm (GA) [3], [4], Particle Swarm Optimization [5], and

In engineering optimization, the problem's dimensionality can reach as high as on the order of 10 3 or 10 4 . This might necessitate very large population sizes, thereby increasing the total number of function evaluations, with each individual requiring a Finite Element Method (FEM) evaluation over multiple iterations that can span from several hours to days. This can prove to be very costly. For instance, topology optimization [16], [17], shape optimization [18], flight trajectory optimization [19], aircraft routing optimization [20], supply chain optimization [21], energy storage optimization [22], job shop scheduling optimization [23] and traveling salesman problem [24], all of these different optimization problems require a higher computational time for each function evaluation. Therefore, it is crucial for an algorithm to achieve the desired fitness with fewer function evaluations consistently since the evaluations are expensive.

All of these factors motivated us to develop and benchmark a quantum meta-heuristic algorithm, namely, QuantumInspired Evolutionary Optimization (QIEO). It is characterized by the independent evolutionary strategy [25], [26] and reliance on only one parameter-dependent operator, which offers the potential to mitigate the discussed limitations of conventional meta-heuristics. Inspired by the probabilistic principles of quantum physics, these algorithms have demonstrated significant potential in optimization.

In this article, we assess the performance of the GPUoptimized QIEO algorithm by comparing it with the GPUoptimized GA using benchmark functions. The evaluation focuses on two key metrics: the number of function evaluations and the consistency in achieving the desired accuracy and convergence rate across multiple trials. Benchmark functions play a crucial role in understanding the algorithm's effectiveness in solving real-world optimization problems. For this purpose, we selected complex, multi-modal, multidimensional, and nonseparable functions from the CEC2014 benchmark suite [27], which closely reflect the challenges encountered in practical optimization scenarios.

## II. METHODOLOGY

## A. Background on quantum-inspired evolutionary algorithm

QIEO algorithm aims to improve optimization processes by leveraging quantum mechanical principles. These algorithms are designed to execute on classical high-performance computers while being adaptable for execution on quantum hardware [28]. Their objective is to explore solution spaces more effectively, potentially outperforming traditional optimization strategies, particularly for specific optimization problems [29].

The principles of quantum mechanics, such as qubits, quantum superposition, quantum gates, and quantum measurement, are mainly used for developing QIEO algorithms. The classical computer uses bits, which can be either 0 or 1. On the other hand, a qubit (the fundamental unit of information in a quantum computer) can exist in a superposition of both states. However, a qubit collapses upon measurement into a classical state, either 0 or 1. In the algorithm, it constitutes a quantum gene, which can be represented as follows:

<!-- formula-not-decoded -->

The state of the quantum gene can be mathematically described as:

<!-- formula-not-decoded -->

Here α and β are complex numbers that specify the probability amplitudes associated with the classical states | 0 ⟩ and | 1 ⟩ , respectively. Their squared magnitudes, | α | 2 and | β | 2 , denote the probability of observing the qubit in the states | 0 ⟩ and | 1 ⟩ , respectively, upon measurement. When m qubits are considered, they can together exist in a superposition of 2 m classical states. These m-qubits form a quantum individual, which is depicted as follows:

<!-- formula-not-decoded -->

The state of the quantum individual is given by:

<!-- formula-not-decoded -->

Here, x represents a classical state with m bits, and p x represents its probability amplitude. For example, a system of three qubits can exist in a superposition of eight classical states whose quantum individual is as follows:

<!-- formula-not-decoded -->

The state of the quantum individual is given by:

<!-- formula-not-decoded -->

where p 000 = α 1 α 2 α 3 , p 001 = α 1 α 2 β 3 and so on. A group of n such quantum individuals form a quantum population Q = { q 1 , q 2 , . . . , q n } . Upon measurement, a classical population P = { p 1 , p 2 , . . . , p n } is created. Having understood this, we will describe the quantum-inspired evolutionary algorithm.

## Algorithm 1 Quantum-Inspired Evolutionary Optimization Algorithm

- 1: t ← 0
- 2: Initialize quantum parallelism to form Q(t) .
- 3: Obtain the population P(t) by observing Q(t) .
- 4: Perform classical information processing by evaluating P(t) , storing the best solution b , and obtaining the θ for R Y ( t ) gate.
- 5: while (not termination condition) do
- 6: t ← t +1
- 7: Perform quantum information processing to exploit the quantum parallelism: Evolve Q ( t -1) using a U -gate to obtain Q ( t ) : R Y ( t -1) · Q ( t -1) = Q ( t ) .
- 8: Obtain the population P ( t ) by observing Q ( t ) .
- 9: Evaluate P ( t ) , store the best solution b till the t th iteration, and obtain the θ s for R Y ( t ) gate.
- 10: end while

The details of the QIEO algorithm are as follows:

Step 1 : The iteration counter t is initialized to 0.

Step 2 - Initialization of quantum parallelism: A population of quantum individuals Q ( t ) = { q t 1 , q t 2 , . . . , q t n } is created such that α s and β s of each q t j (in Eq.(3)) are set to 1 and 0 , respectively. The state of each individual is given by:

<!-- formula-not-decoded -->

The Hadamard gate is then applied which evolves the α s and β s of each q t j to 1 √ 2 , creating the quantum state for the individual:

<!-- formula-not-decoded -->

Step 3 - Measurement: A classical population P ( t ) = { p t 1 , p t 2 , . . . , p t n } is obtained by making a measurement on Q ( t ) , where each p t j is a binary string of length m . The measurement operation on the i th qubit of the j th individual is as follows: it randomly generates a number r between 0 and 1. It then compares this random number to the qubit's | α | 2 . If the random number is greater than | α | 2 , the corresponding classical bit x is set to 1; otherwise, it's set to 0. This process is carried out on all the individuals' qubits, effectively simulating the collapse of the quantum state into a classical state based on the probabilistic measurement. Step 4 - Classical information processing: The obtained classical population is mapped to points within the search space. Their fitness is then evaluated through the objective function, and the fittest solution among P ( t ) is stored as b . The R Y ( t ) gate parameter θ corresponding to the i th qubit of q t j is obtained by comparing the i th bit of b and the i th bit of p t j in P ( t ) . The θ s for all the qubits are determined such that when q t j is evolved through the R Y gate, it has a slightly higher likelihood of generating individuals similar to that of b than before. This process is carried out to generate θ s for all the quantum individuals.

Algorithm 2 Classical information processing to obtain θ , where i b is the ith bit of the fittest individual.

<!-- formula-not-decoded -->

Step 5 and 6: We enter the while loop and update the iteration counter to t = t+1.

Step 7 -Quantum information processing: The θ s obtained for each qubit in each individual from classical information processing are used to construct the corresponding R Y ( t -1) gate. The gate evolves the state of the qubit in q t -1 j to a new state, and its action is shown as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and after the evolution, respectively, and the square matrix is the R Y gate. This process is repeated for all the quantum individuals' qubits to generate the quantum population Q ( t ) . Now that we have obtained the new quantum population Q(t) , obtain P(t) using Q(t) as described in step 3 and then follow the procedure in step 4 with a minor change that the fittest string among P(t=0,1,2...t) is stored as b . This is continued till the termination criterion is satisfied.

## B. Background on Genetic algorithm

For the comparative analysis, this article implemented a generation model genetic algorithm that is presented as follows [30]:

## Algorithm 3 Generational Model Genetic Algorithm

- 1: t ← 1 2: Initialize random population P ( t ) 3: Evaluate P ( t ) 4: while (not termination condition) do 5: Binary tournament selection on P ( t ) to create S ( t ) 6: Crossover on S ( t ) to create C ( t ) 7: Mutation on C ( t ) to create M ( t ) 8: Evaluate M ( t ) 9: Survival of the fittest among ( P ( t ) , M ( t )) to create P ( t +1) . 10: t ← t +1 11: end while

The iteration counter begins at 1, followed by the initialization of a random population of size n . These individuals are then mapped to points in the design variable space and evaluated.

Upon entering the while loop:

- Binary tournament selection is performed on the n individuals of P ( t ) to form a mating pool of n individuals denoted as S ( t ) .
- The individuals of S ( t ) are recombined, resulting in n offspring individuals, C ( t ) .
- Subsequently, mutation is executed on C ( t ) to generate n individuals in M ( t ) .
- M ( t ) is then mapped to points in the design variable space and evaluated.

Finally, the fittest individuals from both P ( t ) and M ( t ) are selected to proceed to the next iteration.

## C. Benchmarking on function optimization

Function optimization is crucial in assessing the effectiveness of newly developed optimization techniques. Test functions, also referred to as artificial landscapes, serve as standardized benchmarks to evaluate algorithm performance under controlled conditions. These functions are categorized based on various factors, such as the type of optimization problem they represent, their complexity, and their specific characteristics. This categorization aids in understanding how different algorithms perform across different issues and can provide insights into their strengths and weaknesses. Common categories of standard functions include:

- Unimodal Functions: These functions have a single optimal solution, making them relatively more straightforward to optimize than multimodal functions.
- Multimodal Functions: Unlike unimodal functions, multimodal functions have multiple local optima, posing a

more significant challenge to optimization algorithms to find the global optimum.

- Separable Functions: These functions can be decomposed into independent sub-functions, allowing optimization algorithms to potentially optimize each sub-function separately.
- Non-Separable Functions: Unlike separable functions, non-separable functions have interdependencies between variables, making them more challenging to optimize.

The following three multi-modal (or non-convex) functions are selected for QIEO benchmarking :

- 1) Ackley: The Ackley function is challenging for optimization algorithms due to its complex landscape, which includes many local minima and a narrow global minimum surrounded by a nearly flat region. This makes it difficult for algorithms to distinguish between local and global optima. The flat regions can cause slow convergence, while the sharp, isolated global minimum requires high precision for an algorithm to reach it. Additionally, the oscillatory nature of the function's surface complicates gradient-based methods. Together, these factors pose significant hurdles for optimization techniques and it is defined as:

<!-- formula-not-decoded -->

- 2) Rosenbrock: The Rosenbrock function is challenging due to its extremely narrow ridge, which makes optimization difficult. The ridge follows a parabolic curve with a sharp peak at its tip. Algorithms that struggle to identify promising search directions often perform poorly on this problem. This characteristic makes Rosenbrock a tough test for many optimization algorithms. The function is defined as follows:

<!-- formula-not-decoded -->

- 3) Rastrigin: Rastrigin's function is notoriously challenging for optimization algorithms due to its vast search space complexity and numerous local minima. The function's surface is shaped by external parameters A and c , which influence the amplitude and frequency of modulation, respectively. With A = 10 and c = 2 π , the modulation dominates the selected domain. This function is highly multimodal, with local minima arranged in a rectangular grid of size 1 . As the distance from the global minimum increases, the fitness values of the local minima grow larger, and the function definition is as follows:

<!-- formula-not-decoded -->

The above three functions are highly nonlinear (increase the function complexity) and non-convex (increase the design space complexity). The global minimum of three functions is zero, and converging to the global minimum is a tedious task with gradient-based approaches as they converge to local minima with the help of search direction. The search direction for the non-linear and non-convex functions is highly dependent on the point where it is and involves more computation. Most engineering optimization problems in the real world have non-linear, multimodal functions as objective functions. By testing optimization algorithms on these standard functions of different categories, researchers can gain valuable insights into the performance characteristics of different algorithms and make informed conclusions regarding their applicability to real-world problems [27].

## III. RESULTS AND DISCUSSIONS

This section presents the results obtained by testing the GPU-optimized QIEO and GA on function optimization, with the functions as described in Section II-C. GA and QIEO algorithms employed in these simulations are developed using CUDA C++, leveraging the computational power of an NVIDIA A100 GPU card. The Center for Computational Research at the University at Buffalo supports these computational facilities [31]. Both algorithms use the same encoding of the design space and apply identical convergence criteria. The convergence criteria used for GA and QIEO are as follows:

- Maximum number of generations: The algorithm terminates when it reaches 3000 generations, regardless of other factors.
- Fitness value tolerance: The algorithm stops when the change in fitness value reaches a tolerance, 1 e -8 , indicating that the solution is sufficiently optimized.

The performance of both algorithms, QIEO and GA, is evaluated through a comprehensive comparison of the results obtained by varying their population sizes. The following two key metrics are used for an effective comparison:

- Accuracy: The ability of the algorithm to reach the fitness value within the given tolerance. The three selected benchmark functions have the global minimum as zero. A solution that is close to zero is generally considered tolerance. The tolerance value for the Ackley and the Rosenbrock is 1 e -3 , while for the Rastrigin, it is 1 e -6 .
- Convergence rate: The number of generations required by the algorithm to reach the specified accuracy.

To analyze the influence of population size on performance, the following populations were tested: 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 5000, 8000, and 10,000. Each optimization problem is evaluated across 30 trials to check the repeatability and reliability of the algorithm. The results are presented using box-and-whisker plots to provide a comprehensive view of performance variation. Figure 1 shows the fitness variation of the Ackley function for different population sizes using the GA and QIEO. Similar plots for the Rosenbrock function are provided in Fig. 4, and for the

Fig. 1: Ackley: Distribution of achieved fitness values across 30 trials for different population sizes.

<!-- image -->

<!-- image -->

Fig. 2: Ackley: Distribution of convergence rates across 30 trials for different population sizes.

<!-- image -->

<!-- image -->

Rastrigin function in Fig. 7, illustrating the accuracy of both algorithms. From these accuracy plots (Figs. 1, 4, and 7), it is evident that GA consistently requires larger population sizes than QIEO for satisfying the convergence criteria. For example, in the case of the Ackley function (ten-dimensional), GA requires a population size of 2000 to achieve the desired fitness accuracy in all trials. In contrast, QIEO accomplishes the same result with a population size of only 100. This disparity is smaller for the Rosenbrock and Rastrigin functions (two-dimensional); for Rosenbrock, GA needs approximately 1000 individuals, while QIEO requires 200. In the Rastrigin function, GA requires 200 individuals compared to QIEO's 100. Despite the smaller gap, QIEO consistently converges with smaller population sizes.

vergence rate plots, even when using the same population size for both QIEO and GA, in most cases, QIEO requires fewer generations to reach the desired accuracy. This is evident from the lower maximum points in the box-and-whisker plots for QIEO compared to GA across most population sizes. The time efficiency naturally follows from the average total number of function evaluations, with related plots shown in Figs. 3b, 6b, and 9b. These plots visually represent the convergence process over time for GA and QIEO, averaged over 30 trials. The bars showcase the variance of the fitness value over 30 trials for that particular time. The population sizes used for demonstrating time efficiency are listed in Table I, and it is clear from these plots that QIEO consistently converges faster than GA to the desired accuracy.

Figure 2 presents the convergence rate variation of the Ackley function for different population sizes from the GA and the QIEO. The similar plots for Rosenbrock and Rastrigin are presented in Figs. 5 and 8 respectively. From these con-

In order to analyze the impact of dimensionality on the optimizer, we vary the dimension of the problem by keeping the population size constant for both GA and QIEO. An increase in dimension inherently increases the number of

<!-- image -->

<!-- image -->

(a) Function Evaluations for GA vs. QIEO

(b) Convergence curves of GA and QIEO

Fig. 3: Ackley: Comparison of GA and QIEO, showing the distribution of the total number of function evaluations across 30 trials to achieve the same fitness and the evolution of the average fitness over time across 30 trials.

200

Fig. 4: Rosenbrock: Distribution of achieved fitness values across 30 trials for different population sizes.

<!-- image -->

<!-- image -->

design variables/unknowns of the optimization problems and provides insights into the scalability of the algorithm. The Ackley function is chosen for this exploration due to its complex nonlinear design space, and the following dimensions are chosen: 2, 5, 10, 20, 25, 30, 40, 50, 100. The variation of fitness across 30 trials is presented in Fig. 10. From the dimension plots, we observe that QIEO finds the desired fitness in all the trials across most dimensions, with the exception of the higher ones - 50 and 100. In stark contrast, GA fails to approach the desired fitness in almost all dimensions and exhibits a wide variance, performing poorly overall and only achieving the desired fitness with low variance only in the lower dimensions - 2 and 5.

If an algorithm achieves the desired fitness with a smaller population size, it might require more generations to do so, which diminishes any practical advantage due to the increased number of function evaluations. Therefore, to accurately assess an algorithm's efficiency, we must consider the average total number of function evaluations required to achieve the target fitness. This number can be derived by multiplying the population size by the average convergence rate across 30 trials derived from the convergence rate plots. For instance, in the Ackley function, QIEO requires a population size of 100 to reach the desired accuracy. By multiplying this population size with the average convergence rate of 387, we calculate a total of 38,700 function evaluations. In contrast, GA requires a population size of 2000 to achieve the same fitness, with an average convergence rate of 243 generations. This results

<!-- image -->

<!-- image -->

Fig. 5: Rosenbrock: Distribution of convergence rates across 30 trials for different population sizes.

<!-- image -->

<!-- image -->

(a) Rosenbrock: Function Evaluations

(b) Rosenbrock: Convergence Plot

Fig. 6: Rosenbrock: Comparison of GA and QIEO, showing the distribution of total function evaluations across 30 trials to achieve the same fitness, and the evolution of average fitness with time

in 486,000 evaluations, indicating that GA demands 12 times more evaluations than QIEO to reach the same fitness level. Similar calculations for the Rastrigin and Rosenbrock functions reveal that GA requires 5.1 and 2.2 times more function evaluations than QIEO. The average convergence rate for the desired accuracy, the population size required, and the total number of function evaluations required for all three functions are tabulated in Table I. The distribution of the total number of function evaluations to achieve the desired fitness across 30 trials is illustrated in Figs. 3a, 6a, 9a. It is evident that QIEO exhibits less variance compared to GA, making it a more reliable choice.

provides insights on the GA takes 430.79 milliseconds for the Ackley function, while QIEO takes 148 milliseconds, making QIEO 2.9 times faster. Similarly, for the Rosenbrock and Rastrigin functions, QIEO is faster by a factor of 3 . 9 and 3 . 84 , respectively, as detailed in Table II.

The speedup achieved by QIEO is calculated as the ratio of the average time taken by GA to the average time taken by QIEO across 30 trials to reach the desired accuracy. It

We have established that QIEO performs well with smaller population sizes and achieves the desired accuracy in fewer generations. This is due to the enhanced exploration of the overall search space provided by QIEO, which helps limit premature convergence [32]. Now, consider a scenario in GA where we start with a very small population, say 10 individuals. If these individuals begin in a poor region of the search space, for them to explore different regions, the crossover point-chosen randomly-must occur in the significant bits of the individual. Suppose each design variable is represented

<!-- image -->

<!-- image -->

Fig. 7: Rastrigin: Distribution of achieved fitness values across 30 trials for different population sizes.

<!-- image -->

<!-- image -->

200

Fig. 8: Rastrigin: Distribution of convergence rates across 30 trials for different population sizes.

by 16 bits; the likelihood of making substantial changes to the variable depends on the probability of the crossover point being selected in these significant bits, as well as the number of crossover events per generation. With fewer individuals, there are fewer crossover opportunities, which reduces diversity in the offspring created [33]. While the mutation operator could change the significant bits, mutation rates are typically kept low. This means the probability of altering the most significant bits is also low, limiting the algorithm's ability to escape unpromising regions of the search space. So, the significant bits may not undergo many changes, thereby forcing the algorithm to stay in the unpromising regions of the search space. This could be one way in which the exploration process is hindered, leading to premature convergence in GA. There's another common reason that, irrespective of the encoding of why GA struggles with smaller population sizes, random events like which individuals are selected for reproduction can lead to certain genes becoming more common by chance, not because they are the best solutions. Over time, this can cause the algorithm to 'drift' towards certain solutions, even if they are not optimal, a phenomenon known as genetic drift [34] [35].

In contrast, in QIEO, even with a small population starting in a poor region from the initial measurement of the uniform superposition, the application of the Ry gate with a small angle ensures that the most significant qubits have an equal likelihood of changing as any other qubits. This enables QIEO to have the most significant qubits with similar probabilities that can be collapsed to 1 and 0 . The rotation gate only partially aligns the most significant qubit with the best individual's significant bit, thereby promoting exploration through the most significant bits. As a result, QIEO enhances exploration and reduces the likelihood of premature convergence even with lower population sizes [36]. The global exploration capabilities of QIEO, without the deterioration of local search capabilities as a result of the probabilistic measurement process, seem to

<!-- image -->

(a) GA vs. QIEO comparison of total number of function evaluations

<!-- image -->

(b) GA vs. QIEO: Convergence curve

Fig. 9: Rastrigin: Comparison of GA and QIEO, showing the distribution of total function evaluations across 30 trials to achieve the same fitness, and the evolution of average fitness over time across 30 trials.

<!-- image -->

Fig. 10: Ackley: Distribution of achieved fitness across 30 trials for different dimensions.

<!-- image -->

provide additional potential for convergence to better solutions.

The variance in convergence rate plots (Figs. 3b, 6b, 9b) for QIEO is consistently lower than that of GA across all cases. Both algorithms, GA and QIEO, initiate the search process randomly, meaning each trial starts from a different set of points. Ideally, an efficient algorithm should require roughly the same number of generations to achieve the desired accuracy, regardless of the starting points, minimizing its dependence on them. QIEO's lower variance suggests it excels in this regard. This can be attributed to the fact that, at the start, the probability amplitudes of the qubits in quantum individuals are close to 1 √ 2 . As a result, the quantum individuals have a high probability of collapsing into various regions of the search space, facilitating an effective global search. In contrast,

GA appears to be more reliant on the initial population. For example, if GA starts in a region of the search space with few optimal minima, the tournament selection operator will still choose the best individuals from this population, but those individuals may still be far from optimal because the region lacks high-quality solutions. This makes GA more dependent on the initial population distribution, contributing to its higher variance. When the crossover and mutation operators are applied to this population, they will recombine and slightly modify these already sub-optimal solutions. While crossover might introduce some new combinations, it is essentially working with poor-quality building blocks [37], making it less likely to create highly fit individuals. Mutation, though capable of introducing diversity, tends to introduce changes, which may sometimes not be sufficient

TABLE I: GA vs. QIEO comparison with respect to total number of function evaluations to reach same accuracy for the three different functions

| Functions   | GA              | GA          | GA                   | QIEO            | QIEO        | QIEO                 |
|-------------|-----------------|-------------|----------------------|-----------------|-------------|----------------------|
|             | Population size | Generations | Function evaluations | Population size | Generations | Function evaluations |
| Ackley      | 2000            | 243         | 486000               | 100             | 387         | 38700                |
| Rosenbrock  | 1000            | 85          | 85000                | 200             | 84          | 16800                |
| Rastrigin   | 200             | 82          | 16400                | 100             | 73          | 7300                 |

TABLE II: Speed-up factor comparison of GA and QIEO for the three different functions

| Functions   |   Time for GA (ms) |   Time for QIEO (ms) |   Speed up |
|-------------|--------------------|----------------------|------------|
| Ackley      |             430.79 |               148.05 |       2.9  |
| Rosenbrock  |              62.93 |                15.85 |       3.9  |
| Rastrigin   |              54.73 |                14.15 |       3.84 |

to break out of the low-quality region of the search space. As a result, GA can get stuck exploring unpromising areas of the search space for several generations before eventually finding better regions, or it might converge prematurely on local optima. This dependency on initial conditions often leads to higher variance in GA's convergence rate across different trials. This can also lead to premature convergence when compared to QIEO, which is better equipped to explore the entire search space early on due to the idea inspired by the uniform quantum superposition. In essence, GA's reliance on selection, crossover, and mutation can sometimes hinder its ability to efficiently navigate the search space if it starts in a poor region or operates with small population sizes. At the same time, QIEO's quantum characteristics allow it to explore more globally from the outset, reducing its dependence on initial conditions or population size. The quantum nature of the individuals allows for a thorough exploration of the entire search space early on. The algorithm gradually gains a good understanding of the landscape and transitions steadily into a more focused search, zeroing in on the region containing the high-quality solutions. This local search phase happens after a few generations in the QIEO [36].

approach reduces the challenge of parameter optimization, making QIEO an effective and scalable solution for complex optimization problems.

A key challenge in classical meta-heuristic algorithms is determining suitable parameter values. Navigating the design space effectively while iteratively improving solution optimality requires a balance between local and global search capabilities. However, achieving this balance is difficult, as prioritizing one often diminishes the other. In genetic algorithms (GA), this balance is managed by multiple operators-primarily crossover and mutation-both of which are parameter-dependent. Understanding how these parameters affect performance and tuning them to optimize results for a specific problem is a complex task [38]. On the other hand, QIEO simplifies this process with only one parameter -Ry gate, paired with the measurement operation. This combination naturally balances exploitation and exploration. Moreover, both components are well-suited for parallelization, enabling independent evolution of individuals without the need for complex parameter tuning [26]. This streamlined

Achieving a suitable balance between exploration, which entails global search, and exploitation, which involves local search, has remained a persistent challenge in meta-heuristic algorithms. Quantum computing concepts in QIEO have bolstered the global search capability without deteriorating the local search capabilities [39], [40]. A key feature of QIEO is the updating of quantum individuals based on the best solution found in previous iterations. By adjusting the quantum individuals' probability amplitudes using the R Y gate, they are guided to have a slightly higher likelihood of collapsing near the best solution. This adjustment helps preserve solution quality while maintaining diversity. The small probabilities associated with this process ensure that quantum individuals do not fully converge to the best solution, leaving a significant chance for them to collapse into other regions of the search space, thereby promoting exploration. The ability of quantum individuals to exist in a superposition of solutions-both near the best solution and in other parts of the search space-enables a balanced search behavior. When measurement occurs, some quantum individuals collapse near the best solution (exploitation), while others collapse into different parts of the search space (exploration). This mechanism replaces traditional operators like crossover and mutation, which are typically used in evolutionary algorithms to improve solutions and maintain diversity. By leveraging quantum principles, QIEO effectively balances guided behavior toward the best solutions and exploration of new areas in the search space.

## IV. CONCLUSIONS

This study evaluates the performance of the GPU-optimized QIEO algorithm in comparison to the GPU-optimized GA on three benchmark functions: Ackley, Rastrigin, and Rosenbrock. The results clearly show that QIEO significantly outperforms GA, requiring far fewer function evaluations to achieve the desired fitness -- 12 times fewer for Ackley, 5 times fewer for Rosenbrock, and 2.5 times fewer for Rastrigin. This reduction in function evaluations translates to a substantial decrease in time taken to convergence, with Ackley completing it in one-third of the time and Rosenbrock and Rastrigin in one-fourth of the time compared to GA. Additionally, QIEO demonstrates lower variance in both fitness outcomes and convergence rates across 30 trials, underscoring its superior reliability. This combination of enhanced reliability, reduced computational effort, and faster optimization makes QIEO particularly valuable for engineering optimization problems, where the function evaluation is the costliest process.

## ACKNOWLEDGMENT

The authors would like to thank the Ministry of Heavy Industries (Government of India) and the Indian School of Business, who supported K.E.S.K, S.B.S., and A.A. for the duration of this project. The authors would like to thank the team of the Center for Computational Research at the University of Buffalo, which gave us the resources to support our work, both in computing capabilities and knowledge from their staff.

## REFERENCES

- [1] S. S. Rao. Engineering Optimization: Theory and Practice . John Wiley &amp; Sons, Inc., Hoboken, New Jersey, 2009.
- [2] Richard Bellman. The theory of dynamic programming. Bulletin of the American Mathematical Society , 60(6):503-516, 1954.
- [3] John H. (John Henry) Holland. Adaptation in natural and artificial systems : an introductory analysis with applications to biology, control, and artificial intelligence. page 211, 1992.
- [4] David E. Goldberg and John H. Holland. Genetic algorithms and machine learning. Machine Learning , 3:95-99, 1988.
- [5] J. Kennedy and R. Eberhart. Particle swarm optimization. Proceedings of ICNN'95 - International Conference on Neural Networks , 4:19421948.
- [6] Marco Dorigo, Vittorio Maniezzo, and Alberto Colorni. Ant system: Optimization by a colony of cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B: Cybernetics , 26:29-41, 1996.
- [7] Miguel Leon Ortiz Ning Xiong, Daniel Molina and Francisco Herrera. A walk into metaheuristics for engineering optimization: Principles, methods and recent trends. International Journal of Computational Intelligence Systems , 8(4):606-636, 2015.
- [8] S. O. Degertekin and Zong Woo Geem. Metaheuristic Optimization in Structural Engineering , pages 75-93. Springer International Publishing, Cham, 2016.
- [9] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan. A fast and elitist multiobjective genetic algorithm: Nsga-ii. IEEE Transactions on Evolutionary Computation , 6:182-197, 4 2002.
- [10] Christian Blum and Andrea Roli. Metaheuristics in combinatorial optimization. ACM Computing Surveys (CSUR) , 35:268-308, 9 2003.
- [11] Zbigniew Michalewicz and David B. Fogel. How to solve it: Modern heuristics. How to Solve It: Modern Heuristics , 2004.
- [12] El-Gazali Talbi. Metaheuristics: From desing to implementation. john wiley &amp; sons, inc. page 624, 2009.
- [13] Jinwoo Kim, Minyoung Kim, Mark-Oliver Stehr, Hyunok Oh, and Soonhoi Ha. A parallel and distributed meta-heuristic framework based on partially ordered knowledge sharing. Journal of Parallel and Distributed Computing , 72(4):564-578, 2012.
- [14] Zhou Jincheng, Lilhore Umesh Kumar, M Poongodi, Hai Tao, Simaiya Sarita, Abang Jawawi Dayang Norhayati, Alsekait Deemamohammed, Ahuja Sachin, Biamba Cresantus, and Hamdi Mounir. Comparative analysis of metaheuristic load balancing algorithms for efficient load balancing in cloud computing. Journal of Cloud Computing , 12(85), 2023.
- [15] Francisco Luna, David L. Gonz´ alez- ´ Alvarez, Francisco Chicano, and Miguel A. Vega-Rodr´ ıguez. On the scalability of multi-objective metaheuristics for the software scheduling problem. In 2011 11th International Conference on Intelligent Systems Design and Applications , pages 1110-1115, 2011.
- [16] Martin Philip Bendsoe and Ole Sigmund. Topology optimization: theory, methods, and applications . Springer Science &amp; Business Media, 2013.
- [17] Kumar K. E. S. and Rakshit S. Topology optimization of the hip bone for gait cycle. Structural and Multidisciplinary Optimization , 62:20352049, 2020.
- [18] Rabii EL MAANI, Bouchaib RADI, and Abdelkhalak EL HAMI. Cfd analysis and shape optimization of naca0012 airfoil for different mach numbers. In 2019 5th International Conference on Optimization and Applications (ICOA) , pages 1-6, 2019.
- [19] Dai Ran and Cochran Jr John E. Three-dimensional trajectory optimization in constrained airspace. Journal of Aircraft , 46, 2009.
- [20] Florian Arnold, Michel Gendreau, and Kenneth S¨ orensen. Efficiently solving very large-scale routing problems. Computers &amp; Operations Research , 107:32-42, 2019.
- [21] E.P. Schulz, M.S. Diaz, and J.A. Bandoni. Supply chain optimization of large-scale continuous processes. Computers &amp; Chemical Engineering , 29(6):1305-1316, 2005. Selected Papers Presented at the 14th European Symposium on Computer Aided Process Engineering.
- [22] Jie Wu, Jia Wang, Kun Li, Hai Zhou, Qin Lv, Li Shang, and Yihe Sun. Large-scale energy storage system design and optimization for emerging electric-drive vehicles. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems , 32(3):325-338, 2013.
- [23] Rong-Hwa Huang and Tung-Han Yu. An effective ant colony optimization algorithm for multi-objective job-shop scheduling with equal-size lot-splitting. Applied Soft Computing , 57:642-656, 2017.
- [24] Chao Jiang, Zhongping Wan, and Zhenhua Peng. A new efficient hybrid algorithm for large scale multiple traveling salesman problems. Expert Systems with Applications , 139:112867, 2020.
- [25] Kuk-Hyun Han and Jong-Hwan Kim. Genetic quantum algorithm and its application to combinatorial optimization problem. In Proceedings of the 2000 Congress on Evolutionary Computation. CEC00 (Cat. No.00TH8512) , volume 2, pages 1354-1360 vol.2, 2000.
- [26] Shikha Gupta and Naveen Kumar. Gpu-based massively parallel quantum inspired genetic algorithm for detection of communities in complex networks. In Proceedings of the Companion Publication of the 2014 Annual Conference on Genetic and Evolutionary Computation , GECCO Comp '14, page 163-164, New York, NY, USA, 2014. Association for Computing Machinery.
- [27] Jing Liang, B. Qu, and Ponnuthurai Suganthan. Problem definitions and evaluation criteria for the cec 2014 special session and competition on single objective real-parameter numerical optimization. 2013.
- [28] Giovanni Acampora and Autilia Vitiello. Implementing evolutionary optimization on actual quantum processors. Information Sciences , 575:542-562, 2021.
- [29] Zhang Gexiang. Quantum-inspired evolutionary algorithms: a survey and empirical study. Journal of Heuristics , 17:303--351, 2011.
- [30] Larry J. Eshelman. The chc adaptive search algorithm: How to have safe search when engaging in nontraditional genetic recombination. In Foundations of Genetic Algorithms , 1990.
- [31] https://ubir.buffalo.edu/xmlui/handle/10477/79221.
- [32] Kuk-Hyun Han and Jong-Hwan Kim. On the analysis of the quantuminspired evolutionary algorithm with a single individual. In Proceedings of the 2006 IEEE Congress on Evolutionary Computation (CEC) , pages 2622-2629, 2006.
- [33] John J. Grefenstette. Incorporating problem specific knowledge into genetic algorithms. In Lawrence Davis, editor, Genetic Algorithms and Simulated Annealing , pages 122-128. Morgan Kaufmann, Los Altos, 1987.
- [34] Sami Ullah and Mohsin Masood. Genetic drift and its effects on the performance of genetic algorithm(ga). In 2023 International Conference on Robotics and Automation in Industry (ICRAI) , pages 1-5, 2023.
- [35] A. Rogers and A. Prugel-Bennett. Genetic drift in genetic algorithm selection schemes. IEEE Transactions on Evolutionary Computation , 3(4):298-303, 1999.
- [36] Kuk-Hyun Han and Jong-Hwan Kim. Quantum-inspired evolutionary algorithm for a class of combinatorial optimization. IEEE Transactions on Evolutionary Computation , 6(6):580-593, 2002.
- [37] Darrell Whitley. A genetic algorithm tutorial. Statistics and Computing , 4(2):65-85, 1994.
- [38] Mohsen Mosayebi and Manbir Sodhi. Tuning genetic algorithm parameters using design of experiments. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference Companion , GECCO '20, page 1937-1944, New York, NY, USA, 2020. Association for Computing Machinery.

[39] Kuk-Hyun Han and Jong-Hwan Kim. On the analysis of the quantuminspired evolutionary algorithm with a single individual. In 2006 IEEE International Conference on Evolutionary Computation , pages 26222629, 2006.

[40] Shahin Hakemi, Mahboobeh Houshmand, Esmaeil KheirKhah, and Seyyed Abed Hosseini. A review of recent advances in quantum-inspired metaheuristics. Evolutionary Intelligence , 17(2):627-642, 2024.