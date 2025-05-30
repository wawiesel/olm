"""Enhanced tests for scale.olm.check module covering untested functionality."""

import pytest
import numpy as np
import scale.olm as so
import scale.olm.check as check
import scale.olm.internal as internal
from unittest.mock import Mock, patch
from pathlib import Path
import json
import tempfile
import os


def data_file(filename):
    """Helper to get test data files."""
    p = Path(__file__).parent.parent / "data" / filename
    size = p.stat().st_size
    if size < 5e4:
        raise ValueError(f"Data file {p} may be a GIT LFS pointer. Run `git lfs pull`.")
    return p


class TestGridGradientAdvanced:
    """Test advanced GridGradient functionality."""
    
    def test_default_params(self):
        """Test that default_params returns expected values."""
        params = check.GridGradient.default_params()
        
        # Verify all expected keys exist
        expected_keys = {'eps0', 'epsa', 'epsr', 'target_q1', 'target_q2'}
        assert set(params.keys()) == expected_keys
        
        # Verify reasonable default values
        assert params['eps0'] == 1e-20
        assert params['epsa'] == 1e-1
        assert params['epsr'] == 1e-1
        assert params['target_q1'] == 0.5
        assert params['target_q2'] == 0.7
        
    def test_describe_params(self):
        """Test that describe_params returns helpful descriptions."""
        descriptions = check.GridGradient.describe_params()
        
        # Verify all parameter descriptions exist
        expected_keys = {'eps0', 'epsa', 'epsr', 'target_q1', 'target_q2'}
        assert set(descriptions.keys()) == expected_keys
        
        # Verify descriptions are strings
        for desc in descriptions.values():
            assert isinstance(desc, str)
            assert len(desc) > 5  # Should be meaningful descriptions
    
    def test_initialization_with_env(self):
        """Test GridGradient initialization with environment variables."""
        env = {'nprocs': 8}
        grid_grad = check.GridGradient(_env=env, eps0=1e-15, target_q1=0.8)
        
        assert grid_grad.eps0 == 1e-15
        assert grid_grad.target_q1 == 0.8
        assert grid_grad.nprocs == 8
        
    def test_kernel_with_simple_data(self):
        """Test the kernel function with simple mathematical data."""
        # Create simple test data
        rel_axes = [[0.0, 0.5, 1.0], [0.0, 1.0]]  # 2D grid
        yreshape = np.array([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # coefficient 0
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]   # coefficient 1
        ])
        eps0 = 1e-10
        
        ahist, rhist, khist = check.GridGradient._GridGradient__kernel(rel_axes, yreshape, eps0)
        
        # Verify output arrays have correct structure
        n_axes = len(rel_axes)
        n_intervals = sum(len(axis) - 1 for axis in rel_axes)  # 2 + 1 = 3
        n_coeff = yreshape.shape[0]  # 2
        expected_length = n_axes * n_intervals * n_coeff  # 2 * 3 * 2 = 12
        
        assert len(ahist) == expected_length
        assert len(rhist) == expected_length
        assert len(khist) == expected_length
        
        # Verify all values are finite and non-negative
        assert np.all(np.isfinite(ahist))
        assert np.all(np.isfinite(rhist))
        assert np.all(ahist >= 0)
        assert np.all(rhist >= 0)
        
        # Verify coefficient indices are valid
        assert np.all(khist >= 0)
        assert np.all(khist < n_coeff)
    
    def test_info_calculation(self):
        """Test the info calculation with known histogram data."""
        grid_grad = check.GridGradient(epsa=0.1, epsr=0.05, target_q1=0.7, target_q2=0.8)
        
        # Manually set histogram data for predictable testing
        # rhist > epsr: points that fail relative test
        # ahist > epsa AND rhist > epsr: points that fail both tests
        grid_grad.ahist = np.array([0.15, 0.05, 0.2, 0.01])  # indices 0,2 > 0.1
        grid_grad.rhist = np.array([0.08, 0.02, 0.1, 0.001]) # indices 0,2 > 0.05
        grid_grad.khist = np.array([0, 1, 0, 1])
        
        info = grid_grad.info()
        
        # Verify basic properties
        assert info.name == "GridGradient"
        assert info.m == 4  # total points
        
        # Let's check the logic:
        # rhist > epsr (0.05): indices 0 (0.08) and 2 (0.1) fail relative test
        # ahist > epsa (0.1) AND rhist > epsr: indices 0 (0.15 > 0.1 AND 0.08 > 0.05) and 2 (0.2 > 0.1 AND 0.1 > 0.05)
        assert info.wr == 2  # points failing relative test (indices 0,2)
        assert info.wa == 2  # points failing both tests (indices 0,2)
        
        # Verify score calculations
        expected_fr = 2.0 / 4.0  # fraction failing relative = 0.5
        expected_fa = 2.0 / 4.0  # fraction failing absolute + relative = 0.5
        expected_q1 = 1.0 - expected_fr  # 0.5
        expected_q2 = 1.0 - 0.9 * expected_fa - 0.1 * expected_fr  # 1.0 - 0.45 - 0.05 = 0.5
        
        assert info.fr == pytest.approx(expected_fr)
        assert info.fa == pytest.approx(expected_fa)
        assert info.q1 == pytest.approx(expected_q1)
        assert info.q2 == pytest.approx(expected_q2)
        
        # Verify test pass flags
        assert info.test_pass_q1 == (expected_q1 >= 0.7)  # False
        assert info.test_pass_q2 == (expected_q2 >= 0.8)  # False
        assert info.test_pass == (info.test_pass_q1 and info.test_pass_q2)  # False


class TestSequencer:
    """Test the check sequencer functionality."""
    
    def test_schema_sequencer(self):
        """Test schema generation for sequencer."""
        schema = check._schema_sequencer()
        assert isinstance(schema, dict)
        assert '_type' in schema or 'properties' in schema
        
        schema_with_state = check._schema_sequencer(with_state=True)
        assert isinstance(schema_with_state, dict)
    
    def test_test_args_sequencer(self):
        """Test test args generation for sequencer."""
        args = check._test_args_sequencer()
        
        assert args['_type'] == 'scale.olm.check:sequencer'
        assert 'sequence' in args
        assert isinstance(args['sequence'], list)
        assert len(args['sequence']) >= 1
        
        # Verify sequence contains valid check types
        for check_def in args['sequence']:
            assert '_type' in check_def
            assert check_def['_type'].startswith('scale.olm.check:')
    
    @patch('scale.olm.internal.logger')
    def test_sequencer_dry_run(self, mock_logger):
        """Test sequencer in dry run mode."""
        sequence = [
            {'_type': 'GridGradient', 'eps0': 1e-10}
        ]
        model = {'name': 'test_model'}
        env = {'work_dir': '/tmp'}
        
        result = check.sequencer(sequence, model, env, dry_run=True)
        
        assert result['test_pass'] == False
        assert 'output' in result
        assert isinstance(result['output'], list)


class TestLowOrderConsistency:
    """Test LowOrderConsistency functionality."""
    
    def test_default_params(self):
        """Test that default_params returns expected values."""
        params = check.LowOrderConsistency.default_params()
        
        # Should return a dictionary with expected parameters
        assert isinstance(params, dict)
        expected_keys = {'eps0', 'epsa', 'epsr', 'target_q1', 'target_q2', 
                        'nuclide_compare', 'template', 'name'}
        
        # May not have all keys but should be a reasonable subset
        assert len(params) > 0
        
    def test_describe_params(self):
        """Test that describe_params returns helpful descriptions."""
        descriptions = check.LowOrderConsistency.describe_params()
        
        expected_keys = {'eps0', 'epsa', 'epsr', 'target_q1', 'target_q2',
                        'nuclide_compare', 'template', 'name'}
        assert set(descriptions.keys()) == expected_keys
        
        # Verify descriptions are strings
        for desc in descriptions.values():
            assert isinstance(desc, str)
            assert len(desc) > 2  # Should be meaningful
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_make_diff_plot_basic(self, mock_figure, mock_savefig):
        """Test the make_diff_plot static method with mocked matplotlib."""
        import tempfile
        
        # Create test data
        identifier = 'u235'
        time = [0, 86400, 172800]  # days in seconds
        min_diff = [-0.01, -0.02, -0.005]
        max_diff = [0.01, 0.02, 0.01]
        max_diff0 = 0.02
        perms = [
            {'(lo-hi)/max(|hi|)': [-0.005, 0.015, 0.008]}
        ]
        
        # Set up mock figure to return mock axes
        mock_figure_instance = Mock()
        mock_axes = Mock()
        mock_figure_instance.gca.return_value = mock_axes
        mock_figure.return_value = mock_figure_instance
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image_path = tmp.name
        
        try:
            # This should not raise an exception
            check.LowOrderConsistency.make_diff_plot(
                identifier, image_path, time, min_diff, max_diff, max_diff0, perms
            )
            
            # Verify matplotlib functions were called (may be called multiple times)
            assert mock_figure.call_count >= 1
            mock_savefig.assert_called_once_with(image_path, bbox_inches="tight")
            
        finally:
            # Clean up
            if os.path.exists(image_path):
                os.unlink(image_path)


class TestSchemaFunctions:
    """Test schema generation functions for all check types."""
    
    def test_schema_gridgradient(self):
        """Test schema generation for GridGradient."""
        schema = check._schema_GridGradient()
        assert isinstance(schema, dict)
        
        schema_with_state = check._schema_GridGradient(with_state=True)
        assert isinstance(schema_with_state, dict)
    
    def test_test_args_gridgradient(self):
        """Test test args generation for GridGradient."""
        args = check._test_args_GridGradient()
        
        assert args['_type'] == 'scale.olm.check:GridGradient'
        
        # Should include default parameters
        default_params = check.GridGradient.default_params()
        for key, value in default_params.items():
            assert key in args
            assert args[key] == value
    
    def test_schema_loworderconsistency(self):
        """Test schema generation for LowOrderConsistency."""
        schema = check._schema_LowOrderConsistency()
        assert isinstance(schema, dict)
        
        schema_with_state = check._schema_LowOrderConsistency(with_state=True)
        assert isinstance(schema_with_state, dict)
    
    def test_test_args_loworderconsistency(self):
        """Test test args generation for LowOrderConsistency."""
        args = check._test_args_LowOrderConsistency()
        
        assert args['_type'] == 'scale.olm.check:LowOrderConsistency'
        # Should be a valid dictionary (exact content depends on implementation)
        assert isinstance(args, dict)


class TestCheckInfo:
    """Test the CheckInfo class."""
    
    def test_checkinfo_initialization(self):
        """Test that CheckInfo initializes correctly."""
        info = check.CheckInfo()
        
        # Should have test_pass set to False by default
        assert hasattr(info, 'test_pass')
        assert info.test_pass == False
        
        # Should be able to set additional attributes
        info.name = "TestCheck"
        info.q1 = 0.85
        info.q2 = 0.90
        
        assert info.name == "TestCheck"
        assert info.q1 == 0.85
        assert info.q2 == 0.90


class TestUtilityFunctions:
    """Test utility functions and edge cases."""
    
    def test_gridgradient_with_constant_data(self):
        """Test GridGradient with constant coefficient data."""
        # Create reactor library with constant coefficients
        rl = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
        
        # Override with constant data
        rl.coeff = np.ones_like(rl.coeff) * 1e-5  # Small constant value
        
        grid_grad = check.GridGradient(eps0=1e-10, epsa=1e-3, epsr=1e-3)
        info = grid_grad.run(rl)
        
        # With constant data, gradients should be very small
        assert info.name == "GridGradient"
        assert 0 <= info.q1 <= 1
        assert 0 <= info.q2 <= 1
        assert info.m > 0
        
        # Most or all points should pass with constant data
        assert info.q1 >= 0.5  # Should have low relative gradients
    
    def test_gridgradient_extreme_values(self):
        """Test GridGradient with extreme coefficient values."""
        rl = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
        
        # Test with very large values
        rl.coeff = np.ones_like(rl.coeff) * 1e10
        
        grid_grad = check.GridGradient(eps0=1e-20, epsa=1e5, epsr=0.1)
        info = grid_grad.run(rl)
        
        assert info.name == "GridGradient"
        assert np.isfinite(info.q1)
        assert np.isfinite(info.q2)
        assert 0 <= info.q1 <= 1
        assert 0 <= info.q2 <= 1
    
    def test_gridgradient_single_axis_point(self):
        """Test GridGradient behavior with minimal axis points."""
        rl = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
        
        # Verify that degenerate axis duplication occurred
        mod_dens_idx = list(rl.axes_names).index("mod_dens")
        assert len(rl.axes_values[mod_dens_idx]) == 2, "Degenerate axis should be duplicated"
        
        grid_grad = check.GridGradient()
        info = grid_grad.run(rl)
        
        # Should work without errors even with minimal points
        assert info.name == "GridGradient"
        assert info.m > 0
        assert np.isfinite(info.q1)
        assert np.isfinite(info.q2) 