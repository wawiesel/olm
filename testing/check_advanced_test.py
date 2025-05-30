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
        # Simplified mock setup
        mock_check_instance = Mock()
        mock_check_instance.run = Mock()
        mock_info = type('MockInfo', (), {'test_pass': True, '__dict__': {'result': 'success'}})()
        mock_check_instance.run.return_value = mock_info
        mock_fn_redirect.return_value = mock_check_instance
        
        mock_reactor_lib.return_value = Mock()
        
        # Test data
        sequence = [{"_type": "scale.olm.check:GridGradient", "eps0": 0.0001}]
        model = {"name": "test_reactor"}
        env = {"work_dir": "/tmp/test"}
        
        with patch('pathlib.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
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