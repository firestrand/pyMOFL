# Rastrigin Function Literature

This directory contains literature and documentation for the Rastrigin optimization function.

## Available Documentation

### Modern GPU Performance Analysis
- **kumar_2024_gpu_benchmarking.md** - Comprehensive 2024 study including:
  - Detailed analysis of Rastrigin function characteristics and optimization challenges
  - Performance comparison between Quantum-Inspired Evolutionary Optimization (QIEO) and Genetic Algorithms (GA)  
  - Results showing QIEO achieves ~4x faster convergence than GA on Rastrigin function
  - Dimensional analysis and population size effects on convergence
  - Tolerance specifications: 1e-6 for 2-dimensional Rastrigin problems

## Function Implementation in pyMOFL

The pyMOFL library implements the standard **Rastrigin Function**:
- **Mathematical Definition**: f(x) = A·n + Σ[x_i² - A·cos(c·x_i)] where A=10, c=2π
- **Global Minimum**: f(0, 0, ..., 0) = 0
- **Default Bounds**: Typically [-5.12, 5.12] for each dimension
- **Properties**: Continuous, Differentiable, Non-Separable, Scalable, Highly Multimodal

## Key Information from Literature

### Function Characteristics (from Kumar et al. 2024)
> "Rastrigin's function is notoriously challenging for optimization algorithms due to its vast search space complexity and numerous local minima. The function's surface is shaped by external parameters A and c, which influence the amplitude and frequency of modulation, respectively. With A = 10 and c = 2π, the modulation dominates the selected domain. This function is highly multimodal, with local minima arranged in a rectangular grid of size 1. As the distance from the global minimum increases, the fitness values of the local minima grow larger."

### Original Definition (Rastrigin 1974)
The function was originally introduced by L.A. Rastrigin in 1974 as part of research on systems of extremal control. The original paper "Systems of extremal control" established the mathematical foundation for this now-ubiquitous benchmark function.

### Optimization Challenges
The Rastrigin function presents several key challenges:
- **High Multimodality**: Numerous local minima arranged in a grid pattern
- **Deceptive Landscape**: Local optima increase in value with distance from global minimum
- **Scale Sensitivity**: Performance varies significantly with problem dimension
- **Algorithm Dependency**: Different optimizers show vastly different convergence rates

## Benchmarking Information

### Performance Metrics (from Kumar et al. 2024)
- **Standard Tolerance**: 1e-6 for 2D problems, 1e-3 for higher dimensions
- **Population Requirements**: QIEO needs ~100 individuals vs GA requiring ~200
- **Convergence Speed**: QIEO converges in ~73 generations vs GA requiring ~82 generations
- **Function Evaluations**: QIEO requires 7,300 vs GA requiring 16,400 evaluations (2.2x improvement)

### Multi-Algorithm Comparisons
The literature provides comprehensive comparisons across:
- Quantum-Inspired Evolutionary Optimization (QIEO): 4x faster convergence  
- Genetic Algorithms (GA): Traditional baseline performance
- Various metaheuristic approaches from broader optimization literature

### Dimensional Analysis
Performance degradation patterns:
- **2D**: Manageable complexity, clear convergence patterns
- **Higher Dimensions**: Exponential increase in difficulty
- **Population Scaling**: Larger populations needed for higher dimensions

## Processing Information

- **Processed**: August 28, 2025
- **Total Pages**: 41 pages from Kumar et al. 2024 GPU benchmarking study
- **Processing Method**: IBM Docling PDF-to-markdown conversion
- **Coverage**: Modern GPU-accelerated optimization techniques (2024)
- **Cross-Reference**: Also covers Ackley and Rosenbrock functions in same study