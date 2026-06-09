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
    
    def test_default_params_enhanced(self):
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
        
    def test_describe_params_enhanced(self):
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
    def test_sequencer_dry_run_enhanced(self, mock_logger):
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
    
    def test_default_params_enhanced_loc(self):
        """Test that default_params returns expected values."""
        params = check.LowOrderConsistency.default_params()
        
        # Should return a dictionary with expected parameters
        assert isinstance(params, dict)
        expected_keys = {
            'eps0',
            'epsa',
            'epsr',
            'metric',
            'target_q1',
            'target_q2',
            'nuclide_compare',
            'template',
            'name',
        }
        assert set(params.keys()) == expected_keys
        assert params['metric'] == 'grams_per_initial_hm'
        
    def test_describe_params_enhanced_loc(self):
        """Test that describe_params returns helpful descriptions."""
        descriptions = check.LowOrderConsistency.describe_params()
        
        expected_keys = {'eps0', 'epsa', 'epsr', 'metric', 'target_q1', 'target_q2',
                        'nuclide_compare', 'template', 'name'}
        assert set(descriptions.keys()) == expected_keys
        
        # Verify descriptions are strings
        for desc in descriptions.values():
            assert isinstance(desc, str)
            assert len(desc) > 2  # Should be meaningful
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_make_scaled_difference_plot_basic(self, mock_figure, mock_savefig):
        """Test the scaled-difference plot with mocked matplotlib."""
        import tempfile
        
        # Create test data
        identifier = 'u235'
        time = [0, 86400, 172800]  # days in seconds
        min_scaled_difference = [-0.01, -0.02, -0.005]
        max_scaled_difference = [0.01, 0.02, 0.01]
        max_abs_scaled_difference = 0.02
        perms = [
            {"scaled_difference": [-0.005, 0.015, 0.008]}
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
            check.LowOrderConsistency.make_scaled_difference_plot(
                identifier,
                image_path,
                time,
                min_scaled_difference,
                max_scaled_difference,
                max_abs_scaled_difference,
                perms,
            )
            
            # Verify matplotlib functions were called (may be called multiple times)
            assert mock_figure.call_count >= 1
            mock_savefig.assert_called_once_with(image_path, bbox_inches="tight")
            
        finally:
            # Clean up
            if os.path.exists(image_path):
                os.unlink(image_path)

    def test_amounts_to_grams_per_initial_hm(self):
        """Test conversion from inventory amounts to g/gIHM."""
        amounts = np.array([
            [[2.0, 3.0], [4.0, 5.0]],
            [[6.0, 7.0], [8.0, 9.0]],
        ])
        loc = check.LowOrderConsistency(_dry_run=True)
        loc.names = ['0092235', '0008016']
        loc.composition_manager = so.core.CompositionManager({
            '0092235': {
                'mass': 10.0,
                'atomicNumber': 92,
                'element': 'U',
                'isomericState': 0,
                'massNumber': 235,
            },
            '0008016': {
                'mass': 20.0,
                'atomicNumber': 8,
                'element': 'O',
                'isomericState': 0,
                'massNumber': 16,
            },
        })
        loc.initialhm_list = [1.0, 2.0]

        result = loc._amounts_to_grams_per_initial_hm(amounts)

        masses = np.array([10.0, 20.0])
        expected = amounts * masses[None, None, :] / np.array(
            [1.0e6, 2.0e6]
        )[:, None, None]
        assert np.allclose(result, expected)

    def test_amounts_to_atom_fraction(self):
        """Test legacy atom-fraction metric conversion."""
        amounts = np.array([[[1.0, 3.0], [2.0, 2.0]]])
        loc = check.LowOrderConsistency(_dry_run=True)

        result = loc._amounts_to_atom_fraction(amounts)

        expected = np.array([[[0.25, 0.75], [0.5, 0.5]]])
        assert np.allclose(result, expected)

    def test_difference_arrays(self):
        """Test LowOrderConsistency pointwise error arrays."""
        lo = np.array([1.0, 3.0, 6.0])
        hi = np.array([2.0, 2.0, 3.0])

        ahist, rhist = check.LowOrderConsistency._difference_arrays(lo, hi, 0.0)

        assert np.allclose(ahist, [1.0, 1.0, 3.0])
        assert np.allclose(rhist, [0.5, 0.5, 1.0])

    def test_scaled_difference(self):
        """Test LowOrderConsistency scaled time-series difference."""
        lo = np.array([1.0, 3.0, 6.0])
        hi = np.array([2.0, 2.0, 3.0])

        scaled_difference = check.LowOrderConsistency._scaled_difference(lo, hi)

        assert np.allclose(scaled_difference, (lo - hi) / 3.0)

    def test_scaled_difference_ylabel(self):
        """Test LowOrderConsistency scaled time-series difference label."""
        assert check.LowOrderConsistency._scaled_difference_ylabel() == (
            "(lo - hi) / max(hi) (%)"
        )

    def test_matching_time_indices_requires_all_high_times_in_low(self):
        """Test LOW order extra times are ignored after matching HIGH order times."""
        indices = check.LowOrderConsistency._matching_time_indices(
            [0.0, 10.0, 20.0],
            [0.0, 5.0, 10.0, 15.0, 20.0],
        )

        assert indices == [0, 2, 4]

    def test_matching_time_indices_requires_initial_time_in_low(self):
        """Test LOW order results must include the initial time."""
        with pytest.raises(ValueError, match="LOW order list of times"):
            check.LowOrderConsistency._matching_time_indices(
                [0.0, 10.0, 20.0],
                [5.0, 10.0, 20.0],
            )

    def test_load_ii_json_aligns_lower_extra_times(self, tmp_path):
        """Test LOW order extra time points are dropped before comparison."""
        nuclide_data = {
            '0092235': {
                'mass': 235.0,
                'atomicNumber': 92,
                'element': 'U',
                'isomericState': 0,
                'massNumber': 235,
            },
            '0008016': {
                'mass': 16.0,
                'atomicNumber': 8,
                'element': 'O',
                'isomericState': 0,
                'massNumber': 16,
            },
        }
        definitions = {'nuclideVectors': {'v': ['0092235', '0008016']}}
        hi = {
            'data': {'nuclides': nuclide_data},
            'definitions': definitions,
            'responses': {
                'system': {
                    'time': [0.0, 10.0, 20.0],
                    'amount': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    'nuclideVectorHash': 'v',
                }
            },
        }
        lo = {
            'data': {'nuclides': nuclide_data},
            'definitions': definitions,
            'responses': {
                'case(1)': {
                    'time': [0.0, 5.0, 10.0, 15.0, 20.0],
                    'amount': [
                        [10.0, 20.0],
                        [50.0, 60.0],
                        [30.0, 40.0],
                        [70.0, 80.0],
                        [50.0, 60.0],
                    ],
                    'nuclideVectorHash': 'v',
                }
            },
        }
        hi_path = tmp_path / 'hi.ii.json'
        lo_path = tmp_path / 'lo.ii.json'
        hi_path.write_text(json.dumps(hi))
        lo_path.write_text(json.dumps(lo))
        loc = check.LowOrderConsistency(_dry_run=True)
        loc.lo_case = 1

        loc._LowOrderConsistency__load_ii_json([(hi_path, lo_path)])

        assert loc.time_list == [0.0, 10.0, 20.0]
        np.testing.assert_allclose(
            loc.lo_list[0],
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
        )
        np.testing.assert_allclose(
            loc.hi_list[0],
            hi['responses']['system']['amount'],
        )

    @patch('scale.olm.core.RelAbsHistogram.plot_hist')
    def test_info_defaults_to_grams_per_initial_hm(self, mock_plot_hist, tmp_path):
        """Test that LowOrderConsistency scores use g/gIHM by default."""
        loc = check.LowOrderConsistency(_dry_run=True, nuclide_compare=[])
        loc.run_success = True
        loc.time_list = [0.0, 86400.0]
        loc.hi_list = [np.array([[2.0, 3.0], [4.0, 1.0]])]
        loc.lo_list = [np.array([[1.0, 4.0], [5.0, 2.0]])]
        loc.names = ['0092235', '0008016']
        loc.composition_manager = so.core.CompositionManager({
            '0092235': {
                'mass': 235.0,
                'atomicNumber': 92,
                'element': 'U',
                'isomericState': 0,
                'massNumber': 235,
            },
            '0008016': {
                'mass': 16.0,
                'atomicNumber': 8,
                'element': 'O',
                'isomericState': 0,
                'massNumber': 16,
            },
        })
        loc.initialhm_list = [2.0]
        loc.ii_json_list = []
        loc.work_path = tmp_path
        loc.check_path = tmp_path

        info = loc.info()

        expected_hi = np.array([loc.hi_list[0]]) * np.array(
            [235.0, 16.0]
        )[None, None, :] / 2.0e6
        expected_lo = np.array([loc.lo_list[0]]) * np.array(
            [235.0, 16.0]
        )[None, None, :] / 2.0e6
        assert info.metric == 'grams_per_initial_hm'
        assert info.units == 'g/gIHM'
        assert np.allclose(loc.hi, expected_hi)
        assert np.allclose(loc.lo, expected_lo)
        assert info.m == 4
        assert info.hist_image == str(tmp_path / 'hist.png')
        assert mock_plot_hist.call_args.kwargs['xlabel'] == (
            r"$\log_{10} |lo/hi-1|$"
        )
        assert mock_plot_hist.call_args.kwargs['ylabel'] == (
            r"$\log_{10} |hi-lo|$ [g/gIHM]"
        )


class TestSchemaFunctions:
    """Test schema generation functions for all check types."""
    
    def test_schema_gridgradient_enhanced(self):
        """Test GridGradient schema generation."""
        schema = check._schema_GridGradient()
        assert isinstance(schema, dict)
        assert '_type' in schema or 'properties' in schema
        
        schema_with_state = check._schema_GridGradient(with_state=True)
        assert isinstance(schema_with_state, dict)
    
    def test_test_args_gridgradient_enhanced(self):
        """Test GridGradient test arguments generation."""
        args = check._test_args_GridGradient()
        
        assert args['_type'] == 'scale.olm.check:GridGradient'
        assert 'eps0' in args
        assert 'target_q1' in args
        assert 'target_q2' in args
        
        # Verify that target values are in reasonable range
        assert 0 <= args['target_q1'] <= 1
        assert 0 <= args['target_q2'] <= 1
    
    def test_schema_loworderconsistency(self):
        """Test schema generation for LowOrderConsistency."""
        schema = check._schema_LowOrderConsistency()
        assert isinstance(schema, dict)
        assert schema['properties']['metric']['enum'] == [
            'grams_per_initial_hm',
            'atom_fraction',
        ]
        
        schema_with_state = check._schema_LowOrderConsistency(with_state=True)
        assert isinstance(schema_with_state, dict)
    
    def test_test_args_loworderconsistency(self):
        """Test test args generation for LowOrderConsistency."""
        args = check._test_args_LowOrderConsistency()
        
        assert args['_type'] == 'scale.olm.check:LowOrderConsistency'
        assert args['metric'] == 'grams_per_initial_hm'
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
