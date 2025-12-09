"""
Tests for pattingalloang core functionality.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from pattingalloang import AizawaSystem, AizawaSolver
from pattingalloang import (
    compute_lyapunov_exponents,
    compute_correlation_dimension,
    compute_kaplan_yorke_dimension,
    compute_kolmogorov_sinai_entropy,
    compute_all_metrics,
)
from pattingalloang.io.config_manager import ConfigManager
from pattingalloang.io.data_handler import DataHandler


class TestAizawaSystem:
    """Test Aizawa attractor system definition."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        system = AizawaSystem()
        assert system.a == 0.95
        assert system.b == 0.7
        assert system.c == 0.6
        assert system.d == 3.5
        assert system.e == 0.25
        assert system.f == 0.1
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        system = AizawaSystem(a=0.9, b=0.8, c=0.5, d=3.0, e=0.3, f=0.15)
        assert system.a == 0.9
        assert system.b == 0.8
        assert system.c == 0.5
        assert system.d == 3.0
        assert system.e == 0.3
        assert system.f == 0.15
    
    def test_derivatives(self):
        """Test derivative computation."""
        system = AizawaSystem()
        state = np.array([0.1, 0.0, 0.0])
        deriv = system.derivatives(state)
        
        assert deriv.shape == (3,)
        assert np.isfinite(deriv).all()
    
    def test_jacobian(self):
        """Test Jacobian matrix computation."""
        system = AizawaSystem()
        state = np.array([0.1, 0.0, 0.0])
        J = system.jacobian(state)
        
        assert J.shape == (3, 3)
        assert np.isfinite(J).all()
    
    def test_params_to_tuple(self):
        """Test parameter conversion to tuple."""
        system = AizawaSystem()
        params = system.params.to_tuple()
        
        assert len(params) == 6
        assert params == (0.95, 0.7, 0.6, 3.5, 0.25, 0.1)
    
    def test_default_initial_conditions(self):
        """Test default initial conditions."""
        system = AizawaSystem()
        ics = system.get_default_initial_conditions()
        
        assert len(ics) == 4
        for ic in ics:
            assert ic.shape == (3,)
    
    def test_repr(self):
        """Test string representation."""
        system = AizawaSystem()
        repr_str = repr(system)
        
        assert "AizawaSystem" in repr_str
        assert "0.95" in repr_str


class TestAizawaSolver:
    """Test JAX-accelerated solver."""
    
    def test_solver_initialization_cpu(self):
        """Test solver initialization with CPU backend."""
        solver = AizawaSolver(dt=0.01, use_gpu=False)
        
        assert solver.dt == 0.01
        assert solver.use_gpu == False
        assert solver.backend == 'cpu'
    
    def test_basic_integration(self):
        """Test basic trajectory integration."""
        system = AizawaSystem()
        solver = AizawaSolver(dt=0.01, use_gpu=False)
        
        result = solver.solve(
            system=system,
            n_steps=1000,
            transient_steps=100,
            verbose=False
        )
        
        assert 'trajectory' in result
        assert 'time' in result
        assert 'x' in result
        assert 'y' in result
        assert 'z' in result
        
        traj = result['trajectory']
        assert traj.shape[1] == 3
        assert np.isfinite(traj).all()
    
    def test_trajectory_shape(self):
        """Test trajectory has correct shape."""
        system = AizawaSystem()
        solver = AizawaSolver(dt=0.01, use_gpu=False)
        
        n_steps = 5000
        transient = 500
        
        result = solver.solve(
            system=system,
            n_steps=n_steps,
            transient_steps=transient,
            verbose=False
        )
        
        expected_points = n_steps
        assert len(result['trajectory']) == expected_points
        assert len(result['time']) == expected_points
    
    def test_attractor_bounds(self):
        """Test that trajectory stays within expected bounds."""
        system = AizawaSystem()
        solver = AizawaSolver(dt=0.01, use_gpu=False)
        
        result = solver.solve(
            system=system,
            n_steps=10000,
            transient_steps=1000,
            verbose=False
        )
        
        traj = result['trajectory']
        
        # Aizawa attractor typically stays within these bounds
        assert np.abs(traj[:, 0]).max() < 5.0  # x
        assert np.abs(traj[:, 1]).max() < 5.0  # y
        assert np.abs(traj[:, 2]).max() < 5.0  # z
    
    def test_batch_integration(self):
        """Test batch trajectory integration."""
        system = AizawaSystem()
        solver = AizawaSolver(dt=0.01, use_gpu=False)
        
        result = solver.solve_batch(
            system=system,
            n_steps=1000,
            initial_conditions=None,  # Use defaults
            transient_steps=100,
            verbose=False
        )
        
        assert 'trajectories' in result
        assert 'n_trajectories' in result
        
        trajs = result['trajectories']
        assert trajs.shape[0] == 4  # Default 4 trajectories
        assert trajs.shape[2] == 3  # 3D
    
    def test_lyapunov_integration(self):
        """Test integration with Jacobian tracking."""
        system = AizawaSystem()
        solver = AizawaSolver(dt=0.01, use_gpu=False)
        
        result = solver.solve_with_lyapunov(
            system=system,
            n_steps=5000,
            verbose=False
        )
        
        assert 'trajectory' in result
        assert 'log_diags' in result
        
        log_diags = result['log_diags']
        assert log_diags.shape[1] == 3
        assert np.isfinite(log_diags).all()


class TestChaosMetrics:
    """Test chaos metric computations."""
    
    @pytest.fixture
    def sample_trajectory(self):
        """Generate sample trajectory for testing."""
        system = AizawaSystem()
        solver = AizawaSolver(dt=0.01, use_gpu=False)
        
        result = solver.solve(
            system=system,
            n_steps=20000,
            transient_steps=2000,
            verbose=False
        )
        
        lyap_result = solver.solve_with_lyapunov(
            system=system,
            n_steps=20000,
            verbose=False
        )
        
        return result, lyap_result
    
    def test_lyapunov_exponents(self, sample_trajectory):
        """Test Lyapunov exponent computation."""
        _, lyap_result = sample_trajectory
        
        lyap = compute_lyapunov_exponents(
            lyap_result['log_diags'],
            dt=0.01
        )
        
        assert len(lyap) == 3
        assert np.isfinite(lyap).all()
        
        # For Aizawa attractor, largest exponent should be positive
        assert lyap[0] > 0
        
        # Exponents should be in descending order
        assert lyap[0] >= lyap[1] >= lyap[2]
    
    def test_kaplan_yorke_dimension(self):
        """Test Kaplan-Yorke dimension computation."""
        # Typical Lyapunov spectrum for Aizawa
        lyap = np.array([0.15, 0.0, -0.5])
        
        d_ky = compute_kaplan_yorke_dimension(lyap)
        
        assert np.isfinite(d_ky)
        assert 2.0 < d_ky < 3.0  # Should be fractal
    
    def test_ks_entropy(self):
        """Test Kolmogorov-Sinai entropy computation."""
        lyap = np.array([0.15, 0.0, -0.5])
        
        h_ks = compute_kolmogorov_sinai_entropy(lyap)
        
        assert np.isfinite(h_ks)
        assert h_ks > 0  # Should be positive for chaotic system
        assert h_ks == 0.15  # Sum of positive exponents
    
    def test_correlation_dimension(self, sample_trajectory):
        """Test correlation dimension computation."""
        result, _ = sample_trajectory
        
        d_corr, r2, _, _ = compute_correlation_dimension(
            result['trajectory'],
            n_samples=2000
        )
        
        assert np.isfinite(d_corr)
        assert 1.5 < d_corr < 3.0  # Typical range for Aizawa
        assert r2 > 0.8  # Good linear fit
    
    def test_all_metrics(self, sample_trajectory):
        """Test comprehensive metrics computation."""
        result, lyap_result = sample_trajectory
        system = result['system']
        
        metrics = compute_all_metrics(
            trajectory=result['trajectory'],
            time_array=result['time'],
            params=system.params.to_tuple(),
            dt=0.01,
            log_diags=lyap_result['log_diags'],
            verbose=False
        )
        
        # Check key metrics exist
        assert 'lyapunov_1' in metrics
        assert 'kaplan_yorke_dim' in metrics
        assert 'correlation_dim' in metrics
        assert 'ks_entropy' in metrics
        assert 'is_chaotic' in metrics
        
        # Check chaotic classification
        assert metrics['is_chaotic'] == True


class TestConfigManager:
    """Test configuration file handling."""
    
    def test_load_config(self):
        """Test loading configuration from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Test config\n")
            f.write("a = 0.95\n")
            f.write("b = 0.7\n")
            f.write("n_steps = 1000\n")
            f.write("use_gpu = false\n")
            f.write("scenario_name = Test\n")
            config_path = f.name
        
        config = ConfigManager.load(config_path)
        
        assert config['a'] == 0.95
        assert config['b'] == 0.7
        assert config['n_steps'] == 1000
        assert config['use_gpu'] == False
        assert config['scenario_name'] == 'Test'
        
        Path(config_path).unlink()
    
    def test_default_config(self):
        """Test default configuration."""
        config = ConfigManager.get_default_config()
        
        assert 'a' in config
        assert 'dt' in config
        assert 'n_steps' in config
        assert config['a'] == 0.95
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config = {'a': 0.95, 'b': 0.7, 'use_gpu': True}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            config_path = f.name
        
        ConfigManager.save(config, config_path)
        
        loaded = ConfigManager.load(config_path)
        assert loaded['a'] == 0.95
        assert loaded['use_gpu'] == True
        
        Path(config_path).unlink()


class TestDataHandler:
    """Test data saving functionality."""
    
    @pytest.fixture
    def sample_result(self):
        """Generate sample result for testing."""
        system = AizawaSystem()
        solver = AizawaSolver(dt=0.01, use_gpu=False)
        
        return solver.solve(
            system=system,
            n_steps=1000,
            transient_steps=100,
            verbose=False
        )
    
    def test_save_trajectory_csv(self, sample_result):
        """Test trajectory CSV saving."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name
        
        DataHandler.save_trajectory_csv(filepath, sample_result)
        
        assert Path(filepath).exists()
        
        # Check content
        import pandas as pd
        df = pd.read_csv(filepath)
        
        assert 'time' in df.columns
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'z' in df.columns
        
        Path(filepath).unlink()
    
    def test_save_metrics_csv(self):
        """Test metrics CSV saving."""
        metrics = {
            'lyapunov_1': 0.15,
            'lyapunov_2': 0.0,
            'kaplan_yorke_dim': 2.3,
            'is_chaotic': True
        }
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            filepath = f.name
        
        DataHandler.save_metrics_csv(filepath, metrics)
        
        assert Path(filepath).exists()
        Path(filepath).unlink()
    
    def test_save_netcdf(self, sample_result):
        """Test NetCDF saving."""
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            filepath = f.name
        
        config = {'scenario_name': 'Test'}
        DataHandler.save_netcdf(filepath, sample_result, config)
        
        assert Path(filepath).exists()
        
        # Check content
        from netCDF4 import Dataset
        with Dataset(filepath, 'r') as nc:
            assert 'time' in nc.variables
            assert 'x' in nc.variables
            assert 'trajectory' in nc.variables
        
        Path(filepath).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
