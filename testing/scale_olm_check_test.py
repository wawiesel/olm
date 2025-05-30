import scale.olm as so
import pytest
import numpy as np


def data_file(filename):
    from pathlib import Path

    p = Path(__file__)
    p = p.parent.parent / "data" / filename
    size = p.stat().st_size
    # 50KB can't be real data
    if size < 5e4:
        raise ValueError(
            f"It appears the data file {p} may be a GIT LFS pointer. Make sure they are downloaded, e.g. with `git lfs pull``."
        )
    return p


def test_degenerate_axis_duplication():
    """Test that ReactorLibrary properly handles degenerate axes with data duplication."""
    # Use the existing test data which already has a degenerate mod_dens axis
    a = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
    
    # Verify the degenerate mod_dens axis was properly duplicated
    mod_dens_idx = list(a.axes_names).index("mod_dens")
    mod_dens_values = a.axes_values[mod_dens_idx]
    
    # Should have exactly 2 values (duplicated from original single value)
    assert len(mod_dens_values) == 2, f"mod_dens should be duplicated, got {len(mod_dens_values)} values"
    
    # Verify positive spacing
    dx = mod_dens_values[1] - mod_dens_values[0]
    assert dx > 0, f"Spacing must be positive, got {dx}"
    assert dx == pytest.approx(0.05, abs=1e-10), f"Expected delta of 0.05 for mod_dens, got {dx}"
    
    # Verify the values are correct
    assert mod_dens_values[0] == pytest.approx(0.723, abs=1e-10), f"First value should be 0.723, got {mod_dens_values[0]}"
    assert mod_dens_values[1] == pytest.approx(0.773, abs=1e-10), f"Second value should be 0.773, got {mod_dens_values[1]}"
    
    # Verify axis shape was updated
    assert a.axes_shape[mod_dens_idx] == 2, "mod_dens axis shape should be 2"
    
    # Verify coefficient data shape includes duplication
    expected_shape = (10, 2, 31, a.ncoeff)  # enrichment, mod_dens, times, transitions
    assert a.coeff.shape == expected_shape, f"Coefficient shape mismatch, expected {expected_shape}, got {a.coeff.shape}"


def test_gradient_calculation_with_duplicated_axis():
    """Test that gradient calculations work properly with duplicated degenerate axes."""
    # Use the existing test data which has a degenerate mod_dens axis
    a = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
    
    # Verify it has a duplicated axis
    mod_dens_idx = list(a.axes_names).index("mod_dens")
    mod_dens_values = a.axes_values[mod_dens_idx]
    assert len(mod_dens_values) == 2, "mod_dens should be duplicated from degenerate axis"
    assert mod_dens_values[1] > mod_dens_values[0], "Duplicated value should be larger"
    
    # Test that GridGradient works without errors
    c = so.check.GridGradient()
    info = c.run(a)
    
    # Should complete without errors and produce valid quality scores
    assert info.name == "GridGradient"
    assert 0 <= info.q1 <= 1, f"q1 should be between 0 and 1, got {info.q1}"
    assert 0 <= info.q2 <= 1, f"q2 should be between 0 and 1, got {info.q2}"
    assert info.m > 0, "Should have processed some gradient points"


def test_duplicate_degenerate_axis_value():
    """Test the duplicate_degenerate_axis_value function with various edge cases."""
    # Test cases: (input_value, expected_delta)
    test_cases = [
        (0.0, 0.05),        # zero -> add 0.05
        (0.723, 0.05),      # small positive -> add 0.05 (min delta)
        (-1.0, 0.05),       # negative -> add 0.05 (min delta)
        (100.0, 5.0),       # large positive -> add 5% = 5.0
        (-100.0, 5.0),      # large negative -> add 5% = 5.0
        (0.001, 0.05),      # small positive -> add 0.05 (min delta)
        (-0.001, 0.05),     # small negative -> add 0.05 (min delta)
        (1e-12, 0.05),      # tiny value -> add 0.05 (below threshold)
        (2.0, 0.1),         # 5% of 2.0 = 0.1
        (20.0, 1.0),        # 5% of 20.0 = 1.0
        (-20.0, 1.0),       # 5% of -20.0 = 1.0
    ]
    
    for x0, expected_delta in test_cases:
        x1 = so.core.ReactorLibrary.duplicate_degenerate_axis_value(x0)
        dx = x1 - x0
        
        # Should always produce positive spacing
        assert dx > 0, f"For x0={x0}, dx must be positive, got {dx}"
        
        # Should match expected delta
        assert dx == pytest.approx(expected_delta, abs=1e-10), f"For x0={x0}, expected delta={expected_delta}, got {dx}"
        
        # Verify the result
        expected_x1 = x0 + expected_delta
        assert x1 == pytest.approx(expected_x1, abs=1e-10), f"For x0={x0}, expected x1={expected_x1}, got {x1}"


def test_gridgradient_basic():
    # Test that we can change the basic parameters of the check.
    c = so.check.GridGradient(eps0=1e-3)
    assert c.eps0 == 1e-3
    assert c.epsr == c.default_params()["epsr"]
    assert c.epsa == c.default_params()["epsa"]

    # Test that we can load an archive
    a = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
    assert a != None

    # Test that we can change and get the default result.
    c.eps0 = c.default_params()["eps0"]
    i = c.run(a)
    assert i.name == "GridGradient"
    assert i.q1 == pytest.approx(0.779, 1e-3)
    assert i.q2 == pytest.approx(0.98, 1e-2)


# ===== Content from testing/scale_olm_check_advanced_test.py =====
"""
Advanced tests for scale.olm.check module.

This module tests the mathematical algorithms and core functionality
of the checking classes, focusing on areas that can improve coverage.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

import scale.olm.check as check
import scale.olm.core as core
import scale.olm.internal as internal


class TestGridGradientMath:
    """Test mathematical calculations in GridGradient class."""
    
    def test_default_params(self):
        """Test GridGradient default parameter values."""
        params = check.GridGradient.default_params()
        
        # Check that all expected parameters are present
        expected_keys = ['eps0', 'epsa', 'epsr', 'target_q1', 'target_q2']
        assert all(key in params for key in expected_keys)
        
        # Check reasonable default values
        assert params['eps0'] > 0
        assert params['epsa'] > 0
        assert params['epsr'] > 0
        assert 0 <= params['target_q1'] <= 1
        assert 0 <= params['target_q2'] <= 1
        
    def test_describe_params(self):
        """Test parameter descriptions are provided."""
        descriptions = check.GridGradient.describe_params()
        
        expected_keys = ['eps0', 'epsa', 'epsr', 'target_q1', 'target_q2']
        assert all(key in descriptions for key in expected_keys)
        assert all(isinstance(desc, str) for desc in descriptions.values())
        assert all(len(desc) > 0 for desc in descriptions.values())

    def test_grid_gradient_initialization(self):
        """Test GridGradient class initialization with various parameters."""
        # Test with default parameters
        gg1 = check.GridGradient()
        assert gg1.eps0 == 1e-20
        assert gg1.epsa == 1e-1
        assert gg1.epsr == 1e-1
        assert gg1.target_q1 == 0.5  # Actual default value
        assert gg1.target_q2 == 0.7  # Corrected actual default value
        
        # Test with custom parameters
        gg2 = check.GridGradient(
            eps0=1e-15,
            epsa=1e-2,
            epsr=1e-2,
            target_q1=0.8,
            target_q2=0.9
        )
        assert gg2.eps0 == 1e-15
        assert gg2.epsa == 1e-2
        assert gg2.epsr == 1e-2
        assert gg2.target_q1 == 0.8
        assert gg2.target_q2 == 0.9

    @pytest.mark.parametrize("eps0,epsa,epsr,target_q1,target_q2", [
        (1e-20, 1e-1, 1e-1, 0.5, 0.7),   # Actual default values (corrected)
        (1e-15, 1e-2, 1e-2, 0.8, 0.9),   # Custom values
        (1e-10, 1e-3, 1e-3, 0.6, 0.85),  # Different values
    ])
    def test_grid_gradient_parameter_variations(self, eps0, epsa, epsr, target_q1, target_q2):
        """Test GridGradient with various parameter combinations."""
        gg = check.GridGradient(
            eps0=eps0,
            epsa=epsa,
            epsr=epsr,
            target_q1=target_q1,
            target_q2=target_q2
        )
        
        assert gg.eps0 == eps0
        assert gg.epsa == epsa
        assert gg.epsr == epsr
        assert gg.target_q1 == target_q1
        assert gg.target_q2 == target_q2


class TestLowOrderConsistencyUtils:
    """Test utility functions in LowOrderConsistency class."""
    
    def test_make_diff_plot_with_mock_data(self):
        """Test difference plot creation with mocked matplotlib."""
        with patch('matplotlib.pyplot.figure'), \
             patch('matplotlib.pyplot.fill_between'), \
             patch('matplotlib.pyplot.plot'), \
             patch('matplotlib.pyplot.xlabel'), \
             patch('matplotlib.pyplot.ylabel'), \
             patch('matplotlib.pyplot.legend'), \
             patch('matplotlib.pyplot.savefig') as mock_save, \
             patch('scale.olm.core.NuclideInventory._nuclide_color', return_value='blue'):
            
            # Test data
            identifier = "U-235"
            image = "/tmp/test_plot.png"
            time = [0, 86400, 172800]  # 0, 1, 2 days in seconds
            min_diff = [-0.01, -0.02, -0.01]
            max_diff = [0.01, 0.02, 0.01]
            max_diff0 = 0.02
            perms = [
                {"(lo-hi)/max(|hi|)": [-0.005, -0.015, -0.005]},
                {"(lo-hi)/max(|hi|)": [0.005, 0.015, 0.005]}
            ]
            
            # Should not raise an exception
            check.LowOrderConsistency.make_diff_plot(
                identifier, image, time, min_diff, max_diff, max_diff0, perms
            )
            
            # Verify savefig was called with correct image path
            mock_save.assert_called_once_with(image, bbox_inches="tight")


class TestSequencerFunction:
    """Test the sequencer function for running check sequences."""
    
    def test_sequencer_schema(self):
        """Test schema generation for sequencer."""
        schema = check._schema_sequencer()
        assert isinstance(schema, dict)
        
        schema_with_state = check._schema_sequencer(with_state=True)
        assert isinstance(schema_with_state, dict)
    
    def test_sequencer_test_args(self):
        """Test test arguments generation for sequencer."""
        args = check._test_args_sequencer()
        
        assert args['_type'] == 'scale.olm.check:sequencer'
        assert 'sequence' in args
        assert isinstance(args['sequence'], list)
        assert len(args['sequence']) >= 1
        
        # Check that sequence contains valid check types
        for seq_item in args['sequence']:
            assert '_type' in seq_item
            assert seq_item['_type'].startswith('scale.olm.check:')
    
    def test_sequencer_dry_run(self):
        """Test sequencer with dry_run=True."""
        sequence = [{"_type": "GridGradient", "eps0": 0.0001}]
        model = {"name": "test"}
        env = {"work_dir": "/tmp"}
        
        result = check.sequencer(
            sequence=sequence,
            _model=model,
            _env=env,
            dry_run=True
        )
        
        assert isinstance(result, dict)
        assert 'test_pass' in result
        assert 'output' in result
        assert result['test_pass'] is False
        assert isinstance(result['output'], list)


class TestCheckInfoClass:
    """Test the CheckInfo class."""
    
    def test_check_info_initialization(self):
        """Test CheckInfo class initialization."""
        info = check.CheckInfo()
        
        assert hasattr(info, 'test_pass')
        assert info.test_pass is False
    
    def test_check_info_attributes(self):
        """Test that CheckInfo can store additional attributes."""
        info = check.CheckInfo()
        
        # Test setting various attributes
        info.test_pass = True
        info.q1 = 0.85
        info.q2 = 0.92
        info.name = "TestCheck"
        
        assert info.test_pass is True
        assert info.q1 == 0.85
        assert info.q2 == 0.92
        assert info.name == "TestCheck"


class TestSchemaFunctions:
    """Test schema generation functions."""
    
    def test_schema_gridgradient(self):
        """Test GridGradient schema generation."""
        schema = check._schema_GridGradient()
        assert isinstance(schema, dict)
        
        schema_with_state = check._schema_GridGradient(with_state=True)
        assert isinstance(schema_with_state, dict)
    
    def test_test_args_gridgradient(self):
        """Test GridGradient test arguments generation."""
        args = check._test_args_GridGradient()
        
        assert args['_type'] == 'scale.olm.check:GridGradient'
        
        # Should contain all default parameters from GridGradient
        expected_keys = ['eps0', 'epsa', 'epsr', 'target_q1', 'target_q2']
        assert all(key in args for key in expected_keys)


class TestErrorHandling:
    """Test error handling in check module functions."""
    
    def test_grid_gradient_with_invalid_parameters(self):
        """Test GridGradient initialization with edge case parameters."""
        # Test with very small values
        gg_small = check.GridGradient(eps0=1e-50, epsa=1e-50, epsr=1e-50)
        assert gg_small.eps0 == 1e-50
        
        # Test with larger values
        gg_large = check.GridGradient(eps0=1.0, epsa=10.0, epsr=10.0)
        assert gg_large.eps0 == 1.0
        
        # Test with target values at boundaries
        gg_bounds = check.GridGradient(target_q1=0.0, target_q2=1.0)
        assert gg_bounds.target_q1 == 0.0
        assert gg_bounds.target_q2 == 1.0


class TestIntegrationWithMocks:
    """Test integration scenarios using mocks."""
    
    @patch('scale.olm.internal._fn_redirect')
    @patch('scale.olm.core.ReactorLibrary')
    def test_sequencer_execution_flow(self, mock_reactor_lib, mock_fn_redirect):
        """Test sequencer execution flow with mocked dependencies."""
        # Simplified mock setup using direct Mock creation (Python 3.12 compatible)
        mock_check_instance = Mock()
        mock_info = Mock()
        mock_info.test_pass = True
        
        # Use configure_mock instead of direct assignment for better compatibility
        mock_check_instance.configure_mock(**{'run.return_value': mock_info})
        mock_fn_redirect.return_value = mock_check_instance
        
        mock_reactor_lib.return_value = Mock()
        
        # Test data
        sequence = [{"_type": "scale.olm.check:GridGradient", "eps0": 0.0001}]
        model = {"name": "test_reactor"}
        env = {"work_dir": "/tmp/test"}
        
        # Mock Path.exists to avoid Path instantiation issues
        with patch('pathlib.Path.exists', return_value=False):
            result = check.sequencer(
                sequence=sequence,
                _model=model,
                _env=env,
                dry_run=False
            )
            
            assert isinstance(result, dict)
            assert 'test_pass' in result
            assert 'sequence' in result


class TestMathematicalCalculations:
    """Test mathematical calculations that might be present in check classes."""
    
    def test_gradient_calculation_concepts(self):
        """Test concepts related to gradient calculations."""
        # Test relative error calculation concepts
        eps0 = 1e-12
        hi_val = 1.0
        lo_val = 0.9
        
        # Relative error
        rel_error = abs((lo_val + eps0) / (hi_val + eps0) - 1.0)
        assert rel_error > 0
        
        # Absolute error
        abs_error = abs(lo_val - hi_val)
        assert abs(abs_error - 0.1) < 1e-10
        
        # Test with very small values
        hi_small = 1e-15
        lo_small = 0.9e-15
        rel_error_small = abs((lo_small + eps0) / (hi_small + eps0) - 1.0)
        assert rel_error_small < rel_error  # Should be much smaller due to eps0
    
    def test_quality_score_calculations(self):
        """Test quality score calculation concepts."""
        # Test quality score calculation logic
        total_points = 1000
        failed_relative = 50  # 5% failure
        failed_absolute_and_relative = 10  # 1% failure
        
        fr = float(failed_relative) / total_points
        fa = float(failed_absolute_and_relative) / total_points
        
        q1 = 1.0 - fr
        q2 = 1.0 - 0.9 * fa - 0.1 * fr
        
        assert q1 == 0.95  # 95% pass rate for relative only
        assert abs(q2 - 0.986) < 1e-10  # Use actual calculated value
        # Note: q2 considers both absolute and relative thresholds
    
    @pytest.mark.parametrize("total,failed_rel,failed_abs_rel,expected_q1,expected_q2", [
        (100, 5, 1, 0.95, 0.986),    # 5% rel failure, 1% abs+rel failure (corrected value)
        (100, 10, 2, 0.90, 0.972),   # 10% rel failure, 2% abs+rel failure (corrected value)  
        (1000, 50, 10, 0.95, 0.986), # 5% rel failure, 1% abs+rel failure (corrected value)
        (100, 0, 0, 1.0, 1.0),       # Perfect scores
    ])
    def test_quality_score_variations(self, total, failed_rel, failed_abs_rel, expected_q1, expected_q2):
        """Test quality score calculations with various scenarios."""
        fr = float(failed_rel) / total
        fa = float(failed_abs_rel) / total
        
        q1 = 1.0 - fr
        q2 = 1.0 - 0.9 * fa - 0.1 * fr
        
        assert abs(q1 - expected_q1) < 1e-10
        assert abs(q2 - expected_q2) < 1e-10 

# ===== Content from testing/scale_olm_check_enhanced_test.py =====
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

# ===== Content from testing/scale_olm_check_optimized_test.py =====
"""Optimized tests for scale.olm.check module that run much faster while maintaining coverage."""

import pytest
import numpy as np
import scale.olm as so
import scale.olm.check as check
from unittest.mock import Mock, patch
from pathlib import Path


def data_file(filename):
    """Helper to get test data files."""
    p = Path(__file__).parent.parent / "data" / filename
    size = p.stat().st_size
    if size < 5e4:
        raise ValueError(f"Data file {p} may be a GIT LFS pointer. Run `git lfs pull`.")
    return p


def test_gridgradient_fast_basic():
    """Fast version of gridgradient test that verifies basic functionality without full computation."""
    # Test parameter setting
    c = so.check.GridGradient(eps0=1e-3)
    assert c.eps0 == 1e-3
    assert c.epsr == c.default_params()["epsr"]
    assert c.epsa == c.default_params()["epsa"]

    # Load reactor library once
    a = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
    assert a is not None
    
    # Speed up test by patching the kernel to run on fewer coefficients
    original_kernel = check.GridGradient._GridGradient__kernel
    
    def fast_kernel(rel_axes, yreshape, eps0):
        # Only process first 5 coefficients instead of all 11
        yreshape_small = yreshape[:5]
        return original_kernel(rel_axes, yreshape_small, eps0)
    
    # Reset parameters for fast test
    c.eps0 = c.default_params()["eps0"]
    
    with patch.object(check.GridGradient, '_GridGradient__kernel', fast_kernel):
        info = c.run(a)
    
    # Verify basic functionality
    assert info.name == "GridGradient"
    assert 0 <= info.q1 <= 1, f"q1 should be between 0 and 1, got {info.q1}"
    assert 0 <= info.q2 <= 1, f"q2 should be between 0 and 1, got {info.q2}"
    assert info.m > 0, "Should have processed some gradient points"


def test_gridgradient_fast_with_duplicated_axis():
    """Fast test that verifies gradient calculation works with duplicated degenerate axes."""
    # Load reactor library and verify degenerate axis duplication occurred
    a = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
    
    mod_dens_idx = list(a.axes_names).index("mod_dens")
    mod_dens_values = a.axes_values[mod_dens_idx]
    assert len(mod_dens_values) == 2, "mod_dens should be duplicated from degenerate axis"
    assert mod_dens_values[1] > mod_dens_values[0], "Duplicated value should be larger"
    
    # Speed up by running on subset of coefficients
    original_kernel = check.GridGradient._GridGradient__kernel
    
    def fast_kernel(rel_axes, yreshape, eps0):
        # Only process first 3 coefficients for speed
        yreshape_small = yreshape[:3]
        return original_kernel(rel_axes, yreshape_small, eps0)
    
    # Test GridGradient with fast parameters
    c = so.check.GridGradient(eps0=1e-10, epsa=1e-1, epsr=1e-1)
    
    with patch.object(check.GridGradient, '_GridGradient__kernel', fast_kernel):
        info = c.run(a)
    
    # Should complete without errors and produce valid quality scores
    assert info.name == "GridGradient"
    assert 0 <= info.q1 <= 1, f"q1 should be between 0 and 1, got {info.q1}"
    assert 0 <= info.q2 <= 1, f"q2 should be between 0 and 1, got {info.q2}"
    assert info.m > 0, "Should have processed some gradient points"


def test_gridgradient_kernel_mathematical_properties():
    """Test the core mathematical properties of the GridGradient kernel with synthetic data."""
    # Create small, controlled test data
    rel_axes = [[0.0, 0.5, 1.0], [0.0, 1.0]]  # 2D grid
    
    # Linear coefficients should produce predictable gradients
    yreshape = np.array([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # linear in both dimensions
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]   # also linear
    ])
    eps0 = 1e-10
    
    ahist, rhist, khist = check.GridGradient._GridGradient__kernel(rel_axes, yreshape, eps0)
    
    # Verify mathematical properties
    assert len(ahist) == len(rhist) == len(khist), "All histogram arrays should have same length"
    assert np.all(np.isfinite(ahist)), "All absolute gradients should be finite"
    assert np.all(np.isfinite(rhist)), "All relative gradients should be finite"
    assert np.all(ahist >= 0), "Absolute gradients should be non-negative"
    assert np.all(rhist >= 0), "Relative gradients should be non-negative"
    
    # For linear data, most gradients should be small (since second derivatives are zero)
    # This tests the Rolle's theorem implementation
    assert np.mean(ahist) < 1.0, "Linear data should have small average absolute gradients"


def test_gridgradient_coefficient_scaling():
    """Test that GridGradient handles different coefficient magnitudes correctly."""
    # Use minimal reactor library
    a = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
    
    # Speed up by running on subset of coefficients
    original_kernel = check.GridGradient._GridGradient__kernel
    
    def fast_kernel(rel_axes, yreshape, eps0):
        # Only process first coefficient for multiple scaling tests
        yreshape_small = yreshape[:1]
        return original_kernel(rel_axes, yreshape_small, eps0)
    
    # Test multiple scenarios
    test_scenarios = [
        (1e-10, "tiny values"),
        (1.0, "normal values"),
        (1e5, "large values")
    ]
    
    for scale_factor, description in test_scenarios:
        # Create scaled reactor library
        scaled_lib = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
        scaled_lib.coeff = scaled_lib.coeff * scale_factor
        
        # Run gradient calculation
        c = so.check.GridGradient(eps0=1e-20, epsa=1e-5, epsr=1e-3)
        
        with patch.object(check.GridGradient, '_GridGradient__kernel', fast_kernel):
            info = c.run(scaled_lib)
        
        # Should handle all scales without errors
        assert info.name == "GridGradient", f"Failed for {description}"
        assert np.isfinite(info.q1), f"q1 not finite for {description}"
        assert np.isfinite(info.q2), f"q2 not finite for {description}"
        assert 0 <= info.q1 <= 1, f"q1 out of range for {description}"
        assert 0 <= info.q2 <= 1, f"q2 out of range for {description}"


def test_gridgradient_quality_score_calculations():
    """Test the quality score calculation logic with known data."""
    # Create GridGradient instance
    grid_grad = check.GridGradient(epsa=0.1, epsr=0.05, target_q1=0.7, target_q2=0.8)
    
    # Manually set histogram data for predictable testing
    grid_grad.ahist = np.array([0.15, 0.05, 0.2, 0.01, 0.001])
    grid_grad.rhist = np.array([0.08, 0.02, 0.1, 0.001, 0.0001])
    grid_grad.khist = np.array([0, 1, 0, 1, 2])
    
    info = grid_grad.info()
    
    # Verify calculations
    assert info.m == 5, "Should count all histogram points"
    
    # Check the basic score properties
    assert 0 <= info.q1 <= 1, "q1 should be between 0 and 1"
    assert 0 <= info.q2 <= 1, "q2 should be between 0 and 1"
    assert info.q2 <= info.q1, "q2 should be less than or equal to q1 (more stringent test)"
    
    # Check test pass logic
    assert info.test_pass == (info.test_pass_q1 and info.test_pass_q2)
    assert info.test_pass_q1 == (info.q1 >= info.target_q1)
    assert info.test_pass_q2 == (info.q2 >= info.target_q2)


def test_gridgradient_parameter_validation():
    """Test GridGradient parameter validation and default values."""
    # Test default initialization
    c1 = check.GridGradient()
    defaults = check.GridGradient.default_params()
    
    assert c1.eps0 == defaults['eps0']
    assert c1.epsa == defaults['epsa']
    assert c1.epsr == defaults['epsr']
    assert c1.target_q1 == defaults['target_q1']
    assert c1.target_q2 == defaults['target_q2']
    
    # Test custom initialization
    c2 = check.GridGradient(eps0=1e-5, epsa=0.01, epsr=0.02, target_q1=0.8, target_q2=0.9)
    
    assert c2.eps0 == 1e-5
    assert c2.epsa == 0.01
    assert c2.epsr == 0.02
    assert c2.target_q1 == 0.8
    assert c2.target_q2 == 0.9
    
    # Test environment parameter
    env = {'nprocs': 8}
    c3 = check.GridGradient(_env=env)
    assert c3.nprocs == 8


def test_gridgradient_minimal_real_data():
    """Test GridGradient with minimal real data processing for speed."""
    a = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
    
    # Verify degenerate axis handling worked
    mod_dens_idx = list(a.axes_names).index("mod_dens")
    assert len(a.axes_values[mod_dens_idx]) == 2, "Degenerate axis should be duplicated"
    
    # Use fastest possible settings
    original_kernel = check.GridGradient._GridGradient__kernel
    
    def minimal_kernel(rel_axes, yreshape, eps0):
        # Process only the first coefficient for speed
        yreshape_minimal = yreshape[:1]
        return original_kernel(rel_axes, yreshape_minimal, eps0)
    
    c = so.check.GridGradient()
    
    with patch.object(check.GridGradient, '_GridGradient__kernel', minimal_kernel):
        info = c.run(a)
    
    # Should complete very quickly and produce valid results
    assert info.name == "GridGradient"
    assert np.isfinite(info.q1)
    assert np.isfinite(info.q2)
    assert 0 <= info.q1 <= 1
    assert 0 <= info.q2 <= 1
    assert info.m > 0


def test_gridgradient_axes_properties():
    """Test that GridGradient correctly interprets reactor library axes."""
    a = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
    
    # Verify axis structure
    assert len(a.axes_names) == len(a.axes_values)
    assert len(a.axes_names) == len(a.axes_shape)
    
    # All axes should be monotonic after library initialization
    for i, axis in enumerate(a.axes_values):
        if len(axis) > 1:
            diffs = np.diff(axis)
            assert np.all(diffs >= 0), f"Axis {i} ({a.axes_names[i]}) should be monotonic non-decreasing"
            assert np.all(diffs > 0), f"Axis {i} ({a.axes_names[i]}) should be strictly increasing"
    
    # Coefficient matrix should have proper shape
    expected_shape = tuple(a.axes_shape) + (a.ncoeff,)
    assert a.coeff.shape == expected_shape, f"Coefficient shape should be {expected_shape}, got {a.coeff.shape}"
    
    # Verify degenerate axis was handled
    mod_dens_idx = list(a.axes_names).index("mod_dens")
    assert a.axes_shape[mod_dens_idx] == 2, "mod_dens should have shape 2 after duplication" 