# Schwefel Functions Literature

This directory contains literature and documentation for the Schwefel family of optimization functions.

## Available Documentation

### Primary Literature Survey
- **jamil_yang_2013_literature_survey.md** - Comprehensive survey of 175 benchmark functions including:
  - Schwefel Function (#118)
  - Schwefel 1.2 Function (#119)  
  - Schwefel 2.4 Function (#120)
  - Schwefel 2.6 Function (#121)
  - Schwefel 2.20 Function (#122)
  - Schwefel 2.21 Function (#123)
  - Schwefel 2.22 Function (#124)
  - Schwefel 2.23 Function (#125)
  - Schwefel 2.25 Function (#127)
  - Schwefel 2.26 Function (#128)
  - Schwefel 2.36 Function (#129)

### Implementation References
- **ding_tan_2014_cuda_benchmark.md** - CUDA implementation details for benchmark functions
- **yang_2023_ten_benchmarks.md** - Modern benchmark function development

## Function Implementations in pyMOFL

The pyMOFL library implements three primary Schwefel variants:

1. **Schwefel_1_2** - Unimodal, non-separable function with nested sum structure
2. **Schwefel_2_6** - Unimodal function using max of linear combinations  
3. **Schwefel_2_13** - Multimodal function with trigonometric components

## Key Information from Literature

### Mathematical Definitions
The literature provides detailed mathematical definitions, characteristics, and properties for each Schwefel variant.

### Properties Summary (from Jamil & Yang 2013)
- **Schwefel 1.2**: Continuous, Differentiable, Non-Separable, Scalable, Unimodal
- **Schwefel 2.6**: Continuous, Differentiable, Non-Separable, Non-Scalable, Unimodal  
- **Other variants**: Various combinations of separability, scalability, and modality

### Historical Context
Original work by H. P. Schwefel:
- "Numerical Optimization for Computer Models" (1981)
- "Evolution and Optimum Seeking" (1995)

These functions have been widely adopted in the optimization community for algorithm testing and validation.

## Processing Information

- **Processed**: August 26, 2025
- **Source**: ArXiv search for benchmark function literature
- **Processing Method**: IBM Docling PDF-to-markdown conversion
- **Total Pages Processed**: 85 pages across 3 documents