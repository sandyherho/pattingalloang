# `pattingalloang`: Aizawa Attractor Analysis

[![Tests](https://github.com/sandyherho/pattingalloang/actions/workflows/tests.yml/badge.svg)](https://github.com/sandyherho/pattingalloang/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/pattingalloang.svg)](https://pypi.org/project/pattingalloang/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

JAX-accelerated Python library for simulating and analyzing the Aizawa strange attractor with comprehensive chaos metrics.

## Governing Equations

$$\dot{x} = (z - b)x - dy$$

$$\dot{y} = dx + (z - b)y$$

$$\dot{z} = c + az - \frac{z^3}{3} - (x^2 + y^2)(1 + ez) + fzx^3$$

**Standard Parameters:** $a=0.95$, $b=0.7$, $c=0.6$, $d=3.5$, $e=0.25$, $f=0.1$

## Installation

```bash
pip install pattingalloang          # From PyPI
pip install jax[cuda12]             # Optional: GPU support
```

**From source:**
```bash
git clone https://github.com/sandyherho/pattingalloang.git
cd pattingalloang && pip install .
```

## Quick Start

**CLI:**
```bash
pattingalloang case1              # Standard Aizawa
pattingalloang --all              # Run all 8 cases
pattingalloang case1 --gpu        # GPU acceleration
```

**Python API:**
```python
from pattingalloang import AizawaSystem, AizawaSolver, compute_all_metrics

system = AizawaSystem(a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1)
solver = AizawaSolver(dt=0.01, use_gpu=False)
result = solver.solve(system, n_steps=80000, initial_state=[0.1, 0.0, 0.0])

lyap_result = solver.solve_with_lyapunov(system, n_steps=30000)
metrics = compute_all_metrics(result['trajectory'], result['time'], 
                              system.params.to_tuple(), dt=0.01,
                              log_diags=lyap_result['log_diags'])

print(f"λ₁={metrics['lyapunov_1']:.4f}, D_KY={metrics['kaplan_yorke_dim']:.4f}")
```

## Test Cases

| Case | Description | Steps | Focus |
|:----:|:------------|------:|:------|
| 1 | Standard Aizawa | 80K | Classic butterfly structure |
| 2 | High Resolution | 150K | Fine attractor detail |
| 3 | Long Trajectory | 300K | Accurate chaos metrics |
| 4 | Parameter Variation | 80K | Effect of parameter changes |
| 5 | Multi-Trajectory | 80K | Sensitivity to initial conditions |
| 6 | Butterfly Wings | 120K | Wing structure emphasis |
| 7 | Chaotic Spiral | 100K | Spiral-like dynamics |
| 8 | Double Loop | 100K | Double-loop structure |

## Chaos Metrics

| Metric | Description |
|:-------|:------------|
| **Lyapunov Exponents** | Rate of trajectory divergence ($\lambda_1 > 0$ → chaos) |
| **Kaplan-Yorke Dimension** | Fractal dimension from Lyapunov spectrum |
| **Correlation Dimension** | Grassberger-Procaccia attractor complexity |
| **KS Entropy** | Information production rate ($\sum \lambda_i$ for $\lambda_i > 0$) |
| **Recurrence Rate** | Fraction of recurrent points |
| **Determinism** | Predictability from recurrence plot |

## Output Files

- **CSV:** `*_trajectory.csv`, `*_chaos_metrics.csv`
- **NetCDF:** CF-compliant with all variables and metadata
- **PNG:** High-resolution static visualizations
- **GIF:** Animated 3D rotation

## Dependencies

`jax` `jaxlib` `numpy` `scipy` `matplotlib` `pandas` `netCDF4` `Pillow` `tqdm`

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
