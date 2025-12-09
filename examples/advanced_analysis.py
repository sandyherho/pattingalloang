#!/usr/bin/env python
"""
Example: Advanced analysis with parameter study.

This script demonstrates advanced features including:
- Parameter variation studies
- Multi-trajectory sensitivity analysis
- GPU acceleration
- Comprehensive chaos analysis

Run with:
    python examples/advanced_analysis.py
"""

import numpy as np
from pathlib import Path
import pandas as pd

from pattingalloang import AizawaSystem, AizawaSolver
from pattingalloang import (
    compute_lyapunov_exponents,
    compute_correlation_dimension,
    compute_kaplan_yorke_dimension,
    compute_kolmogorov_sinai_entropy,
    compute_recurrence_metrics,
)
from pattingalloang.visualization.animator import Animator


def parameter_study():
    """
    Study how Lyapunov exponents change with parameter 'a'.
    """
    print("\n" + "=" * 60)
    print("PARAMETER STUDY: Effect of 'a' on chaos")
    print("=" * 60)
    
    # Range of parameter 'a' to study
    a_values = np.linspace(0.7, 1.2, 11)
    
    results = []
    solver = AizawaSolver(dt=0.01, use_gpu=False)
    
    for a in a_values:
        print(f"\n  Testing a = {a:.2f}...")
        
        system = AizawaSystem(a=a)
        
        # Get Lyapunov exponents
        lyap_result = solver.solve_with_lyapunov(
            system=system,
            n_steps=30000,
            verbose=False
        )
        
        lyap = compute_lyapunov_exponents(lyap_result['log_diags'], dt=0.01)
        d_ky = compute_kaplan_yorke_dimension(lyap)
        h_ks = compute_kolmogorov_sinai_entropy(lyap)
        
        results.append({
            'a': a,
            'lambda_1': lyap[0],
            'lambda_2': lyap[1],
            'lambda_3': lyap[2],
            'D_KY': d_ky,
            'h_KS': h_ks,
            'is_chaotic': lyap[0] > 0
        })
        
        print(f"    λ₁={lyap[0]:.4f}, D_KY={d_ky:.3f}, Chaotic={lyap[0] > 0}")
    
    # Save results
    df = pd.DataFrame(results)
    output_path = Path("example_outputs/parameter_study_a.csv")
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n  Saved results to: {output_path}")
    
    return df


def sensitivity_analysis():
    """
    Analyze sensitivity to initial conditions.
    """
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: Initial condition dependence")
    print("=" * 60)
    
    system = AizawaSystem()
    solver = AizawaSolver(dt=0.01, use_gpu=False)
    
    # Create slightly perturbed initial conditions
    base_ic = np.array([0.1, 0.0, 0.0])
    epsilon = 1e-6
    
    initial_conditions = [
        base_ic,
        base_ic + np.array([epsilon, 0, 0]),
        base_ic + np.array([0, epsilon, 0]),
        base_ic + np.array([0, 0, epsilon]),
    ]
    
    print(f"  Base IC: {base_ic}")
    print(f"  Perturbation: ε = {epsilon}")
    
    # Integrate all trajectories
    result = solver.solve_batch(
        system=system,
        n_steps=50000,
        initial_conditions=initial_conditions,
        transient_steps=0,  # Keep all points for divergence analysis
        verbose=True
    )
    
    trajectories = result['trajectories']
    time = result['time']
    
    # Compute divergence from base trajectory
    base_traj = trajectories[0]
    divergences = []
    
    for i in range(1, len(trajectories)):
        diff = trajectories[i] - base_traj
        dist = np.sqrt(np.sum(diff**2, axis=1))
        divergences.append(dist)
    
    # Estimate Lyapunov exponent from divergence
    divergences = np.array(divergences)
    mean_div = np.mean(divergences, axis=0)
    
    # Linear fit to log(divergence) vs time for estimation
    valid = (mean_div > 0) & (mean_div < 10)  # Reasonable range
    if np.sum(valid) > 100:
        log_div = np.log(mean_div[valid])
        t_valid = time[valid]
        
        # Simple linear regression
        coeffs = np.polyfit(t_valid[:5000], log_div[:5000], 1)
        estimated_lambda = coeffs[0]
        
        print(f"\n  Estimated λ₁ from divergence: {estimated_lambda:.4f}")
    
    # Create visualization
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    
    animator = Animator()
    png_path = output_dir / "sensitivity_analysis.png"
    animator.create_static_plot(
        result, png_path,
        title="Sensitivity Analysis - 4 Trajectories",
        multi_trajectory=True
    )
    print(f"\n  Saved visualization to: {png_path}")


def recurrence_analysis():
    """
    Perform recurrence quantification analysis.
    """
    print("\n" + "=" * 60)
    print("RECURRENCE QUANTIFICATION ANALYSIS")
    print("=" * 60)
    
    system = AizawaSystem()
    solver = AizawaSolver(dt=0.01, use_gpu=False)
    
    result = solver.solve(
        system=system,
        n_steps=30000,
        transient_steps=5000,
        verbose=True
    )
    
    trajectory = result['trajectory']
    
    print("\n  Computing RQA metrics...")
    rqa = compute_recurrence_metrics(trajectory, n_samples=3000)
    
    print("\n  === RQA METRICS ===")
    print(f"  Recurrence Rate: {rqa['recurrence_rate']:.4f}")
    print(f"  Determinism: {rqa['determinism']:.4f}")
    print(f"  Avg Diagonal Line: {rqa['avg_diagonal_line']:.2f}")
    print(f"  Max Diagonal Line: {rqa['max_diagonal_line']}")
    print(f"  Diagonal Entropy: {rqa['diagonal_entropy']:.4f}")
    print(f"  Threshold: {rqa['threshold']:.4f}")
    
    # Interpretation
    print("\n  === INTERPRETATION ===")
    if rqa['determinism'] > 0.8:
        print("  High determinism suggests strong predictability structure.")
    elif rqa['determinism'] > 0.5:
        print("  Moderate determinism indicates mixed predictable/chaotic dynamics.")
    else:
        print("  Low determinism suggests highly chaotic, unpredictable dynamics.")


def main():
    print("=" * 60)
    print("pattingalloang: Advanced Aizawa Attractor Analysis")
    print("=" * 60)
    
    # Run studies
    parameter_study()
    sensitivity_analysis()
    recurrence_analysis()
    
    print("\n" + "=" * 60)
    print("Advanced analysis complete!")
    print("Check 'example_outputs' directory for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
