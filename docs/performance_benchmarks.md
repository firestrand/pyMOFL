### Competitive Benchmark: pyMOFL vs Competitors (Batch Size $N=1000$)

| Suite | Problem Function | Dim | Competitor | Competitor Latency | pyMOFL Batch | pyMOFL Throughput | Speedup vs Competitor | Speedup vs Python Loop |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CEC 2014** | F1: Rotated High Conditioned Elliptic (cec14_f01) | 10 | opfunu |  12.15 ms | **  0.08 ms** | 12.67 M/s | **153.9x** | 178.8x |
| **CEC 2014** | F1: Rotated High Conditioned Elliptic (cec14_f01) | 30 | opfunu |   9.69 ms | **  0.13 ms** |  7.44 M/s | ** 72.1x** |  56.7x |
| **CEC 2014** | F4: Shifted & Rotated Rosenbrock (cec14_f04) | 10 | opfunu |  11.76 ms | **  0.13 ms** |  7.80 M/s | ** 91.8x** | 150.4x |
| **CEC 2014** | F4: Shifted & Rotated Rosenbrock (cec14_f04) | 30 | opfunu |  11.99 ms | **  0.31 ms** |  3.26 M/s | ** 39.1x** |  78.6x |
| **CEC 2014** | F17: Hybrid Function 1 (cec14_f17) | 10 | opfunu |  52.40 ms | **  0.36 ms** |  2.80 M/s | **146.8x** | 175.1x |
| **CEC 2014** | F17: Hybrid Function 1 (cec14_f17) | 30 | opfunu |  52.30 ms | **  1.05 ms** |  0.95 M/s | ** 49.6x** |  70.5x |
| **CEC 2014** | F23: Composition Function 1 (cec14_f23) | 10 | opfunu |  73.23 ms | **  0.73 ms** |  1.38 M/s | **100.8x** | 131.2x |
| **CEC 2014** | F23: Composition Function 1 (cec14_f23) | 30 | opfunu |  75.02 ms | **  1.20 ms** |  0.83 M/s | ** 62.6x** |  75.7x |
| **CEC 2017** | F1: Shifted & Rotated Bent Cigar (cec17_f01) | 10 | opfunu |   7.20 ms | **  0.08 ms** | 11.81 M/s | ** 85.0x** | 143.2x |
| **CEC 2017** | F1: Shifted & Rotated Bent Cigar (cec17_f01) | 30 | opfunu |   7.23 ms | **  0.09 ms** | 10.53 M/s | ** 76.2x** | 112.3x |
| **CEC 2017** | F5: Shifted & Rotated Rastrigin (cec17_f05) | 10 | opfunu |  19.01 ms | **  0.30 ms** |  3.38 M/s | ** 64.2x** |  53.0x |
| **CEC 2017** | F5: Shifted & Rotated Rastrigin (cec17_f05) | 30 | opfunu |  35.24 ms | **  0.62 ms** |  1.61 M/s | ** 56.7x** |  26.2x |
| **CEC 2017** | F11: Hybrid Function 1 (cec17_f11) | 10 | opfunu |  49.43 ms | **  0.18 ms** |  5.47 M/s | **270.3x** | 286.1x |
| **CEC 2017** | F11: Hybrid Function 1 (cec17_f11) | 30 | opfunu |  45.23 ms | **  0.39 ms** |  2.55 M/s | **115.1x** | 105.3x |
| **CEC 2017** | F21: Composition Function 1 (cec17_f21) | 10 | opfunu |  74.81 ms | **  0.79 ms** |  1.27 M/s | ** 95.2x** |  80.2x |
| **CEC 2017** | F21: Composition Function 1 (cec17_f21) | 30 | opfunu |  90.07 ms | **  1.14 ms** |  0.88 M/s | ** 79.0x** |  59.0x |
| **BBOB** | F1: Sphere | 10 | cocoex (C) |   0.83 ms | **  0.04 ms** | 26.41 M/s | ** 21.9x** | 115.8x |
| **BBOB** | F1: Sphere | 20 | cocoex (C) |   0.89 ms | **  0.05 ms** | 20.87 M/s | ** 18.5x** |  91.4x |
| **BBOB** | F8: Rosenbrock | 10 | cocoex (C) |   0.95 ms | **  0.10 ms** | 10.13 M/s | **  9.6x** | 109.2x |
| **BBOB** | F8: Rosenbrock | 20 | cocoex (C) |   1.55 ms | **  0.15 ms** |  6.61 M/s | ** 10.2x** |  68.7x |
| **BBOB** | F15: Rastrigin | 10 | cocoex (C) |   2.56 ms | **  0.89 ms** |  1.12 M/s | **  2.9x** |  59.2x |
| **BBOB** | F15: Rastrigin | 20 | cocoex (C) |   3.56 ms | **  1.92 ms** |  0.52 M/s | **  1.9x** |  24.0x |
| **BBOB** | F21: Gallagher 101 Peaks | 10 | cocoex (C) |   2.46 ms | **  5.17 ms** |  0.19 M/s | **  0.5x** |  95.1x |
| **BBOB** | F21: Gallagher 101 Peaks | 20 | cocoex (C) |   3.68 ms | **  7.86 ms** |  0.13 M/s | **  0.5x** |  60.3x |