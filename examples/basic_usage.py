#!/usr/bin/env python
"""
Example: Basic usage of pattingalloang library.

This script demonstrates how to use the pattingalloang library
to simulate and analyze the Aizawa attractor.

Run with:
    python examples/basic_usage.py
"""

import numpy as np
from pathlib import Path

# Import pattingalloang components
from pattingalloang import AizawaSystem, AizawaSolver
from pattingalloang import compute_all_metrics
from pattingalloang.io.data_handler import DataHandler
from pattingalloang.visualization.animator import Animator


def main():
    print("=" * 60)
    print("pattingalloang: Aizawa Attractor Analysis")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("example_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Define the Aizawa system with standard parameters
    print("\n[1] Creating Aizawa system...")
    system = AizawaSystem(
        a=0.95,
        b=0.7,
        c=0.6,
        d=3.5,
        e=0.25,
        f=0.1
    )
    print(f"    {system}")
    print(system.describe())
    
    # 2. Initialize the JAX-accelerated solver
    print("[2] Initializing solver...")
    solver = AizawaSolver(
        dt=0.01,
        use_gpu=False  # Set to True if you have GPU
    )
    print(f"    dt={solver.dt}, Backend={solver.backend}")
    
    # 3. Integrate the trajectory
    print("\n[3] Integrating trajectory...")
    result = solver.solve(
        system=system,
        n_steps=50000,
        initial_state=[0.1, 0.0, 0.0],
        transient_steps=5000,
        verbose=True
    )
    
    trajectory = result['trajectory']
    print(f"\n    Trajectory shape: {trajectory.shape}")
    print(f"    X range: [{trajectory[:, 0].min():.3f}, {trajectory[:, 0].max():.3f}]")
    print(f"    Y range: [{trajectory[:, 1].min():.3f}, {trajectory[:, 1].max():.3f}]")
    print(f"    Z range: [{trajectory[:, 2].min():.3f}, {trajectory[:, 2].max():.3f}]")
    
    # 4. Compute Lyapunov exponents
    print("\n[4] Computing Lyapunov exponents...")
    lyap_result = solver.solve_with_lyapunov(
        system=system,
        n_steps=30000,
        verbose=True
    )
    
    # 5. Compute all chaos metrics
    print("\n[5] Computing chaos metrics...")
    metrics = compute_all_metrics(
        trajectory=trajectory,
        time_array=result['time'],
        params=system.params.to_tuple(),
        dt=solver.dt,
        log_diags=lyap_result['log_diags'],
        verbose=True
    )
    
    print("\n    === CHAOS METRICS ===")
    print(f"    Lyapunov exponents: [{metrics['lyapunov_1']:.4f}, "
          f"{metrics['lyapunov_2']:.4f}, {metrics['lyapunov_3']:.4f}]")
    print(f"    Sum of Lyapunov exponents: {metrics['lyapunov_sum']:.4f}")
    print(f"    Kaplan-Yorke dimension: {metrics['kaplan_yorke_dim']:.4f}")
    print(f"    Correlation dimension: {metrics['correlation_dim']:.4f}")
    print(f"    KS entropy: {metrics['ks_entropy']:.4f}")
    print(f"    Is chaotic: {metrics['is_chaotic']}")
    
    # 6. Save results
    print("\n[6] Saving results...")
    
    # Save trajectory CSV
    csv_path = output_dir / "aizawa_trajectory.csv"
    DataHandler.save_trajectory_csv(csv_path, result)
    print(f"    Saved: {csv_path}")
    
    # Save metrics CSV
    metrics_path = output_dir / "aizawa_metrics.csv"
    DataHandler.save_metrics_csv(metrics_path, metrics)
    print(f"    Saved: {metrics_path}")
    
    # Save NetCDF
    nc_path = output_dir / "aizawa_simulation.nc"
    config = {'scenario_name': 'Basic Example'}
    DataHandler.save_netcdf(nc_path, result, config, metrics)
    print(f"    Saved: {nc_path}")
    
    # 7. Create visualization
    print("\n[7] Creating visualizations...")
    animator = Animator(fps=30, dpi=150)
    
    # Static plot
    png_path = output_dir / "aizawa_summary.png"
    animator.create_static_plot(
        result, png_path,
        title="Aizawa Strange Attractor",
        metrics=metrics
    )
    print(f"    Saved: {png_path}")
    
    # Animation (optional - takes time)
    create_gif = input("\n    Create GIF animation? (y/n): ").lower().strip() == 'y'
    if create_gif:
        gif_path = output_dir / "aizawa_animation.gif"
        animator.create_animation(
            result, gif_path,
            title="Aizawa Attractor",
            n_frames=120,
            duration_seconds=10.0
        )
        print(f"    Saved: {gif_path}")
    
    print("\n" + "=" * 60)
    print("Done! Check the 'example_outputs' directory for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
