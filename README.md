# `pattingalloang`: A Python Library for Aizawa Attractor Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/pattingalloang.svg)](https://pypi.org/project/pattingalloang/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![JAX](https://img.shields.io/badge/JAX-%23FF6F00.svg?logo=google&logoColor=white)](https://github.com/google/jax)
[![NumPy](https://img.shields.io/badge/NumPy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![netCDF4](https://img.shields.io/badge/netCDF4-%23004B87.svg)](https://unidata.github.io/netcdf4-python/)

A high-performance Python library for simulating and analyzing the Aizawa strange attractor with JAX GPU/CPU acceleration and comprehensive chaos metrics.

## The Aizawa Attractor

The Aizawa attractor is a three-dimensional chaotic dynamical system discovered by Yoji Aizawa. It exhibits beautiful butterfly-like structures and complex chaotic dynamics.

### Governing Equations

$$\dot{x} = (z - b)x - dy$$

$$\dot{y} = dx + (z - b)y$$

$$\dot{z} = c + az - \frac{z^3}{3} - (x^2 + y^2)(1 + ez) + fzx^3$$

### Standard Parameters

| Parameter | Symbol | Default Value | Description |
|:---------:|:------:|:-------------:|:------------|
| a | $a$ | 0.95 | Linear growth rate in z |
| b | $b$ | 0.7 | Offset parameter |
| c | $c$ | 0.6 | Constant forcing |
| d | $d$ | 3.5 | Rotation coupling |
| e | $e$ | 0.25 | Nonlinear coupling |
| f | $f$ | 0.1 | Cubic coupling |

## Chaos Metrics

This library implements rigorous chaos quantification measures:

| Metric | Description | Interpretation |
|:------:|:------------|:---------------|
| **Lyapunov Exponents** | Rate of divergence of nearby trajectories | λ₁ > 0 indicates chaos |
| **Lyapunov Dimension** | Kaplan-Yorke dimension from Lyapunov spectrum | Fractal dimension estimate |
| **Correlation Dimension** | Grassberger-Procaccia algorithm | Attractor complexity |
| **Kolmogorov-Sinai Entropy** | Sum of positive Lyapunov exponents | Information production rate |
| **Recurrence Rate** | Fraction of recurrent points | System periodicity |
| **Determinism** | Diagonal structure in recurrence plot | Predictability measure |
| **Average Mutual Information** | Nonlinear correlation measure | Optimal embedding delay |
| **Largest Lyapunov Exponent** | Maximum expansion rate | Primary chaos indicator |

## Installation

**From PyPI:**
```bash
pip install pattingalloang
```

**From source:**
```bash
git clone https://github.com/sandyherho/pattingalloang.git
cd pattingalloang
pip install .
```

**Development installation with Poetry:**
```bash
git clone https://github.com/sandyherho/pattingalloang.git
cd pattingalloang
poetry install
```

**For GPU acceleration:**
```bash
# CUDA 12
pip install jax[cuda12]
# CUDA 11
pip install jax[cuda11]
```

## Quick Start

**CLI:**
```bash
pattingalloang case1              # Standard Aizawa attractor
pattingalloang case2              # High-resolution analysis
pattingalloang case3              # Long trajectory chaos analysis
pattingalloang case4              # Parameter variation study
pattingalloang case5              # Multi-trajectory visualization
pattingalloang --all              # Run all test cases
pattingalloang case1 --gpu        # Use GPU acceleration
pattingalloang case1 --no-gif     # Skip GIF generation
```

**Python API:**
```python
from pattingalloang import AizawaSystem, AizawaSolver
from pattingalloang import compute_all_metrics

# Create system with standard parameters
system = AizawaSystem(
    a=0.95, b=0.7, c=0.6,
    d=3.5, e=0.25, f=0.1
)

# Initialize JAX-accelerated solver
solver = AizawaSolver(dt=0.01, use_gpu=False)

# Integrate trajectory
result = solver.solve(
    system=system,
    n_steps=80000,
    initial_state=[0.1, 0.0, 0.0]
)

# Compute chaos metrics
metrics = compute_all_metrics(
    result['trajectory'],
    result['time'],
    system.params,
    dt=0.01
)

print(f"Largest Lyapunov exponent: {metrics['lyapunov_1']:.4f}")
print(f"Kaplan-Yorke dimension: {metrics['kaplan_yorke_dim']:.4f}")
print(f"Correlation dimension: {metrics['correlation_dim']:.4f}")
```

## Features

- **JAX Acceleration**: GPU/CPU JIT-compiled integration (10-100x speedup)
- **Comprehensive Chaos Metrics**: Lyapunov exponents, dimensions, entropy
- **Beautiful Visualizations**: Stunning dark-themed 3D plots and animations
- **Multiple Output Formats**: CSV, NetCDF (CF-compliant), PNG, GIF
- **Flexible Configuration**: Text-based config files or Python API
- **Batch Processing**: Multiple trajectories with parallel computation

## Output Files

The library generates:

- **CSV files**: 
  - `*_trajectory.csv` - Full trajectory data (x, y, z)
  - `*_chaos_metrics.csv` - All chaos measures
  - `*_lyapunov_timeseries.csv` - Lyapunov exponent evolution

- **NetCDF**: Full trajectory with all metrics and CF-compliant metadata
  - Variables: `x`, `y`, `z`, `time`, all chaos metrics
  - Global attributes: system parameters, integration settings

- **PNG**: High-resolution static visualizations
  - 3D attractor plot with multiple viewing angles
  - Phase portraits (xy, xz, yz projections)
  - Chaos metrics summary panel

- **GIF**: Animated 3D visualization with smooth camera rotation

## Test Cases

| Case | Description | Steps | Focus |
|:----:|:------------|:-----:|:------|
| 1 | Standard Aizawa | 80,000 | Classic butterfly structure |
| 2 | High Resolution | 150,000 | Fine attractor detail |
| 3 | Long Trajectory | 300,000 | Accurate chaos metrics |
| 4 | Parameter Variation | 80,000 | Effect of parameter changes |
| 5 | Multi-Trajectory | 80,000 | Sensitivity to initial conditions |

## Dependencies

- **jax** >= 0.4.0
- **jaxlib** >= 0.4.0
- **numpy** >= 1.20.0
- **scipy** >= 1.7.0
- **matplotlib** >= 3.5.0
- **pandas** >= 1.3.0
- **netCDF4** >= 1.5.0
- **Pillow** >= 9.0.0
- **tqdm** >= 4.60.0

## License

MIT © Sandy H. S. Herho

## Citation

```bibtex
@software{herho2025_pattingalloang,
  title   = {pattingalloang: A Python library for Aizawa attractor analysis},
  author  = {Herho, Sandy H. S.},
  year    = {2025},
  url     = {https://github.com/sandyherho/pattingalloang}
}
```

## References

- Aizawa, Y. (1982). Global aspects of the dissipative dynamical systems. 
- Sprott, J. C. (2010). Elegant Chaos: Algebraically Simple Chaotic Flows.
- Kantz, H., & Schreiber, T. (2004). Nonlinear Time Series Analysis.
