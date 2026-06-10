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
        expected_keys = {'eps0', 'epsa', 'epsr', 'target_q_r', 'target_q_ar'}
        assert set(params.keys()) == expected_keys
        
        # Verify reasonable default values
        assert params['eps0'] == 1e-20
        assert params['epsa'] == 1e-1
        assert params['epsr'] == 1e-1
        assert params['target_q_r'] == 0.5
        assert params['target_q_ar'] == 0.7
        
    def test_describe_params_enhanced(self):
        """Test that describe_params returns helpful descriptions."""
        descriptions = check.GridGradient.describe_params()
        
        # Verify all parameter descriptions exist
        expected_keys = {'eps0', 'epsa', 'epsr', 'target_q_r', 'target_q_ar'}
        assert set(descriptions.keys()) == expected_keys
        
        # Verify descriptions are strings
        for desc in descriptions.values():
            assert isinstance(desc, str)
            assert len(desc) > 5  # Should be meaningful descriptions
    
    def test_initialization_with_env(self):
        """Test GridGradient initialization with environment variables."""
        env = {'nprocs': 8}
        grid_grad = check.GridGradient(_env=env, eps0=1e-15, target_q_r=0.8)
        
        assert grid_grad.eps0 == 1e-15
        assert grid_grad.target_q_r == 0.8
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
        grid_grad = check.GridGradient(epsa=0.1, epsr=0.05, target_q_r=0.7, target_q_ar=0.8)
        
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
        assert info.w_r == 2  # points failing relative test (indices 0,2)
        assert info.w_ar == 2  # points failing both tests (indices 0,2)
        
        # Verify score calculations
        expected_fr = 2.0 / 4.0  # fraction failing relative = 0.5
        expected_fa = 2.0 / 4.0  # fraction failing absolute + relative = 0.5
        expected_q_r = 1.0 - expected_fr  # 0.5
        expected_q_ar = 1.0 - 0.9 * expected_fa - 0.1 * expected_fr  # 1.0 - 0.45 - 0.05 = 0.5
        
        assert info.f_r == pytest.approx(expected_fr)
        assert info.f_ar == pytest.approx(expected_fa)
        assert info.q_r == pytest.approx(expected_q_r)
        assert info.q_ar == pytest.approx(expected_q_ar)
        
        # Verify test pass flags
        assert info.test_pass_q_r == (expected_q_r >= 0.7)  # False
        assert info.test_pass_q_ar == (expected_q_ar >= 0.8)  # False
        assert info.test_pass == (info.test_pass_q_r and info.test_pass_q_ar)  # False

    def test_quality_summary_discards_points_below_eps0(self):
        """Test points below eps0 on both axes do not affect q-scores."""
        summary = check._quality_summary_from_histograms(
            ahist=np.array([1.0e-13, 1.0e-5, 1.0e-4]),
            rhist=np.array([1.0e-13, 1.0e-13, 1.0e-2]),
            eps0=1.0e-12,
            epsa=1.0e-6,
            epsr=1.0e-3,
            target_q_r=0.9,
            target_q_ar=0.95,
        )

        assert summary["m"] == 2
        assert summary["w_r"] == 1
        assert summary["w_ar"] == 1
        assert summary["f_r"] == pytest.approx(0.5)
        assert summary["f_ar"] == pytest.approx(0.5)
        assert summary["q_r"] == pytest.approx(0.5)
        assert summary["q_ar"] == pytest.approx(0.5)
        assert not summary["test_pass"]

    def test_quality_summary_all_points_below_eps0_passes(self):
        """Test a fully discarded set has perfect q-scores and zero counts."""
        summary = check._quality_summary_from_histograms(
            ahist=np.array([1.0e-13, 2.0e-13]),
            rhist=np.array([1.0e-13, 2.0e-13]),
            eps0=1.0e-12,
            epsa=1.0e-6,
            epsr=1.0e-3,
            target_q_r=0.9,
            target_q_ar=0.95,
        )

        assert summary["m"] == 0
        assert summary["w_r"] == 0
        assert summary["w_ar"] == 0
        assert summary["q_r"] == pytest.approx(1.0)
        assert summary["q_ar"] == pytest.approx(1.0)
        assert summary["test_pass"]
        assert summary["mean_abs_diff"] == pytest.approx(0.0)
        assert summary["mean_rel_diff"] == pytest.approx(0.0)


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
            'convergence',
            'target_q_r',
            'target_q_ar',
            'nuclide_compare',
            'nuclide_scaled_difference_min_abs_ylim',
            'template',
            'name',
        }
        assert set(params.keys()) == expected_keys
        assert params['metric'] == 'grams_per_initial_hm'
        assert params['nuclide_scaled_difference_min_abs_ylim'] is None
        
    def test_describe_params_enhanced_loc(self):
        """Test that describe_params returns helpful descriptions."""
        descriptions = check.LowOrderConsistency.describe_params()
        
        expected_keys = {
            'eps0',
            'epsa',
            'epsr',
            'metric',
            'convergence',
            'target_q_r',
            'target_q_ar',
            'nuclide_compare',
            'nuclide_scaled_difference_min_abs_ylim',
            'template',
            'name',
        }
        assert set(descriptions.keys()) == expected_keys
        
        # Verify descriptions are strings
        for desc in descriptions.values():
            assert isinstance(desc, str)
            assert len(desc) > 2  # Should be meaningful

    def test_nuclide_scaled_difference_ylim_defaults_to_epsr(self):
        """Test omitted nuclide scaled-difference y-limit uses epsr."""
        loc = check.LowOrderConsistency(_dry_run=True, epsr=0.004)
        loc.run_success = False

        info = loc.info()

        assert info.nuclide_scaled_difference_min_abs_ylim == 0.004

    def test_nuclide_scaled_difference_ylim_override_in_check_info(self):
        """Test explicit nuclide scaled-difference y-limit is written to check info."""
        loc = check.LowOrderConsistency(
            _dry_run=True,
            epsr=0.004,
            nuclide_scaled_difference_min_abs_ylim=0.025,
        )
        loc.run_success = False

        info = loc.info()

        assert info.nuclide_scaled_difference_min_abs_ylim == 0.025
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.ylim')
    @patch('matplotlib.pyplot.figure')
    def test_make_scaled_difference_plot_basic(
        self, mock_figure, mock_ylim, mock_savefig
    ):
        """Test the scaled-difference plot with mocked matplotlib."""
        import tempfile
        
        # Create test data
        identifier = 'u235'
        time = [0, 86400, 172800]  # days in seconds
        min_scaled_difference = [-0.001, -0.002, -0.0005]
        max_scaled_difference = [0.001, 0.002, 0.001]
        max_abs_scaled_difference = 0.002
        perms = [
            {"scaled_difference": [-0.0005, 0.0015, 0.0008]}
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
                min_abs_ylim=0.01,
            )
            
            # Verify matplotlib functions were called (may be called multiple times)
            assert mock_figure.call_count >= 1
            mock_ylim.assert_called_once_with(-1.0, 1.0)
            mock_savefig.assert_called_once_with(image_path, bbox_inches="tight")
            
        finally:
            # Clean up
            if os.path.exists(image_path):
                os.unlink(image_path)

    @patch('scale.olm.core.RelAbsHistogram.plot_hist')
    @patch.object(check.LowOrderConsistency, 'make_time_quality_plot')
    def test_time_quality_failure_fails_check(
        self, mock_time_quality_plot, mock_hist_plot, tmp_path
    ):
        """Test every high-order time point must satisfy q_r/q_ar targets."""
        loc = check.LowOrderConsistency(
            metric='atom_fraction',
            target_q_r=0.7,
            target_q_ar=0.7,
            nuclide_compare=[],
            _dry_run=True,
        )
        hi = np.full((1, 4, 10), 0.1)
        lo = hi.copy()
        lo[0, 1, :] = 0.0
        lo[0, 1, 0] = 1.0
        loc.hi_list = hi
        loc.lo_list = lo
        loc.time_list = [0.0, 86400.0, 172800.0, 259200.0]
        loc.burnup_list = [0.0, 10000.0, 20000.0, 30000.0]
        loc.work_path = tmp_path
        loc.check_path = tmp_path
        loc.run_success = True

        info = loc.info()

        assert info.q_r == pytest.approx(0.0)
        assert info.q_ar == pytest.approx(0.0)
        assert info.test_pass_q_r is False
        assert info.test_pass_q_ar is False
        assert info.test_pass_initial is True
        assert info.test_pass_initial_q_r is True
        assert info.test_pass_initial_q_ar is True
        assert info.minimum_time_quality['q_r'] == pytest.approx(0.0)
        assert info.minimum_time_quality['q_ar'] == pytest.approx(0.0)
        assert info.minimum_time_quality['test_pass'] is False
        assert info.minimum_q_r_time_quality['score'] == pytest.approx(0.0)
        assert info.minimum_q_r_time_quality['target'] == pytest.approx(0.7)
        assert info.minimum_q_r_time_quality['test_pass_score'] is False
        assert info.minimum_q_ar_time_quality['score'] == pytest.approx(0.0)
        assert info.minimum_q_ar_time_quality['target'] == pytest.approx(0.7)
        assert info.minimum_q_ar_time_quality['test_pass_score'] is False
        assert info.test_pass_time_q_r is False
        assert info.test_pass_time_q_ar is False
        assert info.test_pass_time is False
        assert info.test_pass is False
        assert info.time_quality[1]['q_r'] == pytest.approx(0.0)
        assert info.time_quality[1]['q_ar'] == pytest.approx(0.0)
        assert info.time_quality[1]['burnup_gwd_per_mtihm'] == pytest.approx(10.0)
        assert info.worst_time_quality['index'] == 1
        assert info.worst_time_quality['time_days'] == pytest.approx(1.0)
        assert info.worst_time_quality['burnup'] == pytest.approx(10000.0)
        assert info.worst_time_quality['limiting_score'] == 'q_r'
        assert info.worst_time_quality['limiting_score_shortfall'] == pytest.approx(
            0.7
        )
        assert info.first_failed_time_quality['index'] == 1
        assert info.first_failed_time_quality['time_days'] == pytest.approx(1.0)
        assert info.first_failed_time_quality['burnup_gwd_per_mtihm'] == (
            pytest.approx(10.0)
        )
        mock_time_quality_plot.assert_called_once()
        mock_hist_plot.assert_called_once()

    @patch('scale.olm.core.RelAbsHistogram.plot_hist')
    @patch.object(check.LowOrderConsistency, 'make_time_quality_plot')
    def test_initial_identity_failure_fails_test_one(
        self, mock_time_quality_plot, mock_hist_plot, tmp_path
    ):
        """Test t=0 q_r/q_ar must be exactly 1.000."""
        loc = check.LowOrderConsistency(
            metric='atom_fraction',
            target_q_r=0.0,
            target_q_ar=0.0,
            nuclide_compare=[],
            _dry_run=True,
        )
        hi = np.full((1, 2, 4), 0.25)
        lo = hi.copy()
        lo[0, 0, :] = [0.26, 0.24, 0.25, 0.25]
        loc.hi_list = hi
        loc.lo_list = lo
        loc.time_list = [0.0, 86400.0]
        loc.burnup_list = [0.0, 1000.0]
        loc.work_path = tmp_path
        loc.check_path = tmp_path
        loc.run_success = True

        info = loc.info()

        assert info.initial_time_quality['q_r'] < 1.0
        assert info.initial_time_quality['q_ar'] < 1.0
        assert info.test_pass_initial_q_r is False
        assert info.test_pass_initial_q_ar is False
        assert info.test_pass_initial is False
        assert info.test_pass_time is True
        assert info.test_pass is False
        assert loc._failure_reasons(info) == [
            'test 1.1 failed: q_r at t=0 must be 1.000',
            'test 1.2 failed: q_ar at t=0 must be 1.000',
        ]
        mock_time_quality_plot.assert_called_once()
        mock_hist_plot.assert_called_once()

    def test_burnup_list_from_assemble_uses_history_endpoints(self):
        """Test report burnups come from replay history, not the F33 grid."""
        assemble_data = {
            'space': {
                'burnup': {
                    'grid': [0.0, 5.0, 15.0, 25.0],
                },
            },
            'points': [
                {
                    'history': {
                        'burndata': [
                            {'power': 40.0, 'burn': 1.0},
                            {'power': 40.0, 'burn': 2.0},
                        ],
                    },
                },
                {
                    'history': {
                        'burndata': [
                            {'power': 42.0, 'burn': 1.0},
                            {'power': 42.0, 'burn': 2.0},
                        ],
                    },
                },
            ],
        }

        burnups = check.LowOrderConsistency._burnup_list_from_assemble(
            assemble_data
        )

        assert burnups == pytest.approx([0.0, 41.0, 123.0])

    def test_convergence_summary_uses_worst_time_scores(self):
        """Test convergence rows summarize the lowest time-dependent q scores."""
        info = check.CheckInfo()
        info.nlib = 2
        info.nburn = 4
        info.q_r = 0.95
        info.q_ar = 0.96
        info.test_pass = False
        info.test_pass_q_r = True
        info.test_pass_q_ar = True
        info.test_pass_time = False
        info.mean_abs_diff = 0.0
        info.mean_rel_diff = 0.0
        info.time_quality = [
            {
                'time': 0.0,
                'time_days': 0.0,
                'burnup_gwd_per_mtihm': 0.0,
                'q_r': 0.80,
                'q_ar': 0.99,
                'target_q_r': 0.70,
                'target_q_ar': 0.95,
                'test_pass': True,
                'limiting_score': 'q_r',
                'limiting_score_shortfall': 0.0,
            },
            {
                'time': 86400.0,
                'time_days': 1.0,
                'burnup_gwd_per_mtihm': 10.0,
                'q_r': 0.60,
                'q_ar': 0.90,
                'target_q_r': 0.70,
                'target_q_ar': 0.95,
                'test_pass': False,
                'limiting_score': 'q_r',
                'limiting_score_shortfall': 0.10,
            },
        ]

        row = check.LowOrderConsistency._convergence_summary(info)

        assert 'overall_q_r' not in row
        assert 'overall_q_ar' not in row
        assert row['q_r'] == pytest.approx(0.60)
        assert row['q_ar'] == pytest.approx(0.90)
        assert row['result'] == 'fail q_r/q_ar'
        assert row['time_days'] == pytest.approx(1.0)
        assert row['burnup_gwd_per_mtihm'] == pytest.approx(10.0)
        assert row['time_quality'] == info.time_quality

    @patch.object(check.LowOrderConsistency, 'make_convergence_quality_plot')
    @patch.object(check.LowOrderConsistency, 'make_time_quality_plot')
    def test_write_convergence_diagnostics_adds_plots_and_history(
        self, mock_time_plot, mock_convergence_plot, tmp_path
    ):
        """Test final convergence diagnostics include history rows and plot paths."""
        loc = check.LowOrderConsistency(_dry_run=True, convergence={})
        loc.target_q_r = 0.7
        loc.target_q_ar = 0.95
        loc.check_path = tmp_path
        info = check.CheckInfo()
        info.nlib = 2
        info.nburn = 2
        info.time_quality = [
            {'time': 0.0, 'time_days': 0.0, 'q_r': 0.8, 'q_ar': 0.96},
            {'time': 86400.0, 'time_days': 1.0, 'q_r': 0.75, 'q_ar': 0.95},
        ]
        info.time_quality_image = str(tmp_path / 'q_r-q_ar-by-time.png')
        first = {
            'nlib': 1,
            'nburn': 1,
            'q_r': 0.6,
            'q_ar': 0.9,
            'time_quality': [
                {'time_days': 0.0, 'q_r': 0.7, 'q_ar': 0.95},
                {'time_days': 1.0, 'q_r': 0.6, 'q_ar': 0.9},
            ],
        }
        final = {
            'nlib': 2,
            'nburn': 2,
            'q_r': 0.75,
            'q_ar': 0.95,
            'time_quality': info.time_quality,
        }
        info.nlib_history = [first]
        info.nburn_history = [first, final]

        loc._write_convergence_diagnostics(info)

        assert info.convergence_history == [first, final]
        assert info.convergence_quality_image == str(
            tmp_path / 'q_r-q_ar-convergence.png'
        )
        mock_time_plot.assert_called_once()
        assert mock_time_plot.call_args.kwargs['convergence_history'] == [first]
        mock_convergence_plot.assert_called_once_with(
            tmp_path / 'q_r-q_ar-convergence.png',
            [first, final],
            0.7,
            0.95,
        )

    def test_convergence_max_status_does_not_fail_check(self):
        """Test convergence max is search metadata, not a failure reason."""
        loc = check.LowOrderConsistency(
            _dry_run=True,
            convergence={
                'nlib_start': 1,
                'nlib_max': 4,
                'nburn_start': 1,
                'nburn_max': 8,
                'q_r_stop_criteria': 0.005,
                'q_ar_stop_criteria': 0.005,
            },
        )
        info = check.CheckInfo()
        info.q_r = 0.8881592039800995
        info.q_ar = 0.9879410639112132
        info.target_q_r = 0.7
        info.target_q_ar = 0.95
        info.test_pass_q_r = True
        info.test_pass_q_ar = True
        info.test_pass_time = True
        info.nlib = 4
        info.nburn = 1
        info.test_pass_nlib = False
        info.test_pass_nburn = True
        info.test_pass_initial = True
        info.nlib_delta_q_r = 0.05375430539609638
        info.nlib_delta_q_ar = 0.006346727898966731
        info.nlib_convergence_stop = 'max'

        info.convergence_status = loc._convergence_status(info)
        reasons = loc._failure_reasons(info)

        assert info.convergence_status['nlib']['result'] == 'max'
        assert info.convergence_status['nlib']['delta_q_r_text'] == (
            '5.375e-02 / 5.000e-03'
        )
        assert info.convergence_status['nlib']['reason'] == (
            'reached nlib_max before another stop criterion'
        )
        assert info.convergence_status['nburn']['result'] == 'selected'
        assert reasons == []

    def test_failure_reason_preserves_incomplete_check_error(self):
        """Test early check failures preserve the original calculation error."""
        loc = check.LowOrderConsistency(
            _dry_run=True,
            convergence={
                'nlib_start': 1,
                'nlib_max': 16,
                'nburn_start': 1,
                'nburn_max': 8,
                'q_r_stop_criteria': 0.005,
                'q_ar_stop_criteria': 0.01,
            },
        )
        info = check.CheckInfo()
        info.nlib = 1
        info.nburn = 1
        info.test_pass_nlib = False
        info.test_pass_nburn = False
        info.run_error = (
            "***Error: the interpolated value of      99.2 calculated for "
            "cycle 160 is outside the range of       0.0 to      98.7 for "
            "library uox_quick"
        )

        info.convergence_status = loc._convergence_status(info)
        reasons = loc._failure_reasons(info)

        assert info.convergence_status['nlib']['result'] == 'error'
        assert info.convergence_status['nlib']['reason'] == 'check did not complete'
        assert len(reasons) == 1
        assert reasons[0].startswith('check calculation failed: ***Error:')
        assert 'outside the range' in reasons[0]

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
        assert mock_plot_hist.call_args.kwargs['epsr'] == loc.epsr
        assert mock_plot_hist.call_args.kwargs['epsa'] == loc.epsa
        assert mock_plot_hist.call_args.kwargs['eps0'] == loc.eps0

    def test_convergence_range_checks(self):
        """Test LowOrderConsistency rejects inverted convergence ranges."""
        with pytest.raises(ValueError, match='nlib_max'):
            check.LowOrderConsistency(
                _dry_run=True,
                convergence={
                    'nlib_start': 4,
                    'nlib_max': 2,
                },
            )

        with pytest.raises(ValueError, match='nburn_max'):
            check.LowOrderConsistency(
                _dry_run=True,
                convergence={
                    'nburn_start': 4,
                    'nburn_max': 2,
                },
            )

    def test_convergence_check_path_uses_convergence_subdirectories(self, tmp_path):
        """Test convergence runs isolate outputs under _convergence."""
        loc = check.LowOrderConsistency(
            _dry_run=True,
            convergence={
                'nlib_start': 1,
                'nlib_max': 4,
                'nburn_start': 1,
                'nburn_max': 2,
            },
        )
        loc.work_path = tmp_path
        loc.base_check_path = tmp_path / 'check' / 'loc'
        loc.convergence_check_path = loc.base_check_path / '_convergence'

        loc._set_check_path_for_convergence(nlib=2, nburn=2)

        assert loc.check_path == (
            tmp_path
            / 'check'
            / 'loc'
            / '_convergence'
            / 'nlib0002'
            / 'nburn0002'
        )
        assert loc.check_dir == Path('check/loc/_convergence/nlib0002/nburn0002')

    def test_final_convergence_outputs_copy_to_nominal_path(self, tmp_path):
        """Test selected convergence outputs are copied to the normal check path."""
        loc = check.LowOrderConsistency(_dry_run=True, convergence={})
        loc.work_path = tmp_path
        loc.base_check_path = tmp_path / 'check' / 'loc'
        loc.convergence_check_path = loc.base_check_path / '_convergence'
        loc.check_path = loc.convergence_check_path / 'nlib0002' / 'nburn0004'
        loc.check_dir = loc.check_path.relative_to(loc.work_path)
        source_point = loc.check_path / 'uox.arc'
        source_point.mkdir(parents=True)
        (source_point / 'uox.arc.inp').write_text('input')
        (loc.check_path / 'hist.png').write_text('hist')
        loc.base_check_path.mkdir(parents=True, exist_ok=True)
        (loc.base_check_path / 'stale.txt').write_text('stale')

        info = check.CheckInfo()
        info.hist_image = str(loc.check_path / 'hist.png')
        info.time_quality_image = str(loc.check_path / 'q_r-q_ar-by-time.png')
        info.nuclide_compare = {
            'u235': {
                'image': str(loc.check_path / 'u235-scaled-difference.png'),
            },
        }

        loc._copy_final_check_outputs_to_nominal(info)

        assert (loc.base_check_path / '_convergence').exists()
        assert not (loc.base_check_path / 'stale.txt').exists()
        assert (loc.base_check_path / 'uox.arc' / 'uox.arc.inp').read_text() == (
            'input'
        )
        assert (loc.base_check_path / 'hist.png').read_text() == 'hist'
        assert info.hist_image == str(loc.base_check_path / 'hist.png')
        assert info.time_quality_image == str(
            loc.base_check_path / 'q_r-q_ar-by-time.png'
        )
        assert info.nuclide_compare['u235']['image'] == str(
            loc.base_check_path / 'u235-scaled-difference.png'
        )
        assert loc.check_path == loc.base_check_path
        assert loc.check_dir == Path('check/loc')

    def test_scores_converged_uses_q_score_deltas(self):
        """Test convergence is based on q_r and q_ar score changes."""
        loc = check.LowOrderConsistency(
            _dry_run=True,
            convergence={
                'q_r_stop_criteria': 0.01,
                'q_ar_stop_criteria': 0.02,
            },
        )
        previous = check.CheckInfo()
        previous.q_r = 0.90
        previous.q_ar = 0.95
        current = check.CheckInfo()
        current.q_r = 0.905
        current.q_ar = 0.969

        assert loc._scores_converged(previous, current, fixed_grid=False)

        current.q_ar = 0.971
        assert not loc._scores_converged(previous, current, fixed_grid=False)

    def test_scores_converged_when_scores_cross_targets(self):
        """Test convergence stops when q_r/q_ar rise through pass targets."""
        loc = check.LowOrderConsistency(
            _dry_run=True,
            target_q_r=0.7,
            target_q_ar=0.95,
            convergence={
                'q_r_stop_criteria': 0.001,
                'q_ar_stop_criteria': 0.001,
            },
        )
        previous = check.CheckInfo()
        previous.q_r = 0.60
        previous.q_ar = 0.90
        current = check.CheckInfo()
        current.q_r = 0.71
        current.q_ar = 0.96

        assert loc._convergence_stop_reason(
            previous, current, fixed_grid=False
        ) == 'target'
        assert loc._scores_converged(previous, current, fixed_grid=False)

    def test_run_nlib_convergence_doubles_until_scores_converge(self, monkeypatch):
        """Test nlib convergence runs until q-score deltas meet the stop criteria."""
        loc = check.LowOrderConsistency(
            _dry_run=True,
            convergence={
                'nlib_start': 1,
                'nlib_max': 4,
                'q_r_stop_criteria': 0.01,
                'q_ar_stop_criteria': 0.01,
            },
        )
        scores = {
            1: (0.50, 0.60),
            2: (0.70, 0.80),
            4: (0.705, 0.805),
        }
        calls = []

        def fake_run_once(do_run, nlib, nburn):
            calls.append((nlib, nburn))
            info = check.CheckInfo()
            info.nlib = nlib
            info.nburn = nburn
            info.q_r, info.q_ar = scores[nlib]
            info.test_pass = True
            info.test_pass_q_r = True
            info.test_pass_q_ar = True
            info.mean_abs_diff = 0.0
            info.mean_rel_diff = 0.0
            return info

        monkeypatch.setattr(loc, '_run_once', fake_run_once)

        info = loc._run_nlib_convergence(do_run=False, nburn=1)

        assert calls == [(1, 1), (2, 1), (4, 1)]
        assert info.nlib == 4
        assert info.nlib_converged
        assert info.test_pass_nlib
        assert info.nlib_convergence_stop == 'delta'
        assert info.nlib_delta_q_r == pytest.approx(0.005)
        assert info.nlib_delta_q_ar == pytest.approx(0.005)
        assert [row['nlib'] for row in info.nlib_history] == [1, 2, 4]

    def test_run_nlib_convergence_stops_when_scores_cross_targets(
        self, monkeypatch
    ):
        """Test nlib search stops when time q-scores rise above targets."""
        loc = check.LowOrderConsistency(
            _dry_run=True,
            target_q_r=0.7,
            target_q_ar=0.95,
            convergence={
                'nlib_start': 1,
                'nlib_max': 4,
                'q_r_stop_criteria': 0.001,
                'q_ar_stop_criteria': 0.001,
            },
        )
        scores = {
            1: (0.60, 0.90),
            2: (0.71, 0.96),
            4: (0.72, 0.97),
        }
        calls = []

        def fake_run_once(do_run, nlib, nburn):
            calls.append((nlib, nburn))
            info = check.CheckInfo()
            info.nlib = nlib
            info.nburn = nburn
            info.q_r, info.q_ar = scores[nlib]
            info.test_pass = True
            info.test_pass_q_r = info.q_r >= loc.target_q_r
            info.test_pass_q_ar = info.q_ar >= loc.target_q_ar
            info.mean_abs_diff = 0.0
            info.mean_rel_diff = 0.0
            return info

        monkeypatch.setattr(loc, '_run_once', fake_run_once)

        info = loc._run_nlib_convergence(do_run=False, nburn=1)

        assert calls == [(1, 1), (2, 1)]
        assert info.nlib == 2
        assert info.nlib_converged
        assert info.nlib_convergence_stop == 'target'

    def test_run_nburn_convergence_uses_converged_nlib(self, monkeypatch):
        """Test nburn convergence holds nlib fixed after nlib convergence."""
        loc = check.LowOrderConsistency(
            _dry_run=True,
            convergence={
                'nburn_start': 1,
                'nburn_max': 4,
                'q_r_stop_criteria': 0.01,
                'q_ar_stop_criteria': 0.01,
            },
        )
        nlib_info = check.CheckInfo()
        nlib_info.nlib = 2
        nlib_info.nburn = 1
        nlib_info.q_r = 0.50
        nlib_info.q_ar = 0.60
        nlib_info.test_pass = True
        nlib_info.test_pass_q_r = True
        nlib_info.test_pass_q_ar = True
        nlib_info.mean_abs_diff = 0.0
        nlib_info.mean_rel_diff = 0.0
        nlib_info.nlib_history = [{'nlib': 2, 'nburn': 1}]
        nlib_info.nlib_converged = True
        nlib_info.test_pass_nlib = True
        scores = {
            2: (0.70, 0.80),
            4: (0.705, 0.805),
        }
        calls = []

        def fake_run_once(do_run, nlib, nburn):
            calls.append((nlib, nburn))
            info = check.CheckInfo()
            info.nlib = nlib
            info.nburn = nburn
            info.q_r, info.q_ar = scores[nburn]
            info.test_pass = True
            info.test_pass_q_r = True
            info.test_pass_q_ar = True
            info.mean_abs_diff = 0.0
            info.mean_rel_diff = 0.0
            return info

        monkeypatch.setattr(loc, '_run_once', fake_run_once)

        info = loc._run_nburn_convergence(do_run=False, nlib_info=nlib_info)

        assert calls == [(2, 2), (2, 4)]
        assert info.nlib == 2
        assert info.nburn == 4
        assert info.nburn_converged
        assert info.test_pass_nburn
        assert info.nburn_convergence_stop == 'delta'
        assert info.nlib_history == nlib_info.nlib_history
        assert [row['nburn'] for row in info.nburn_history] == [1, 2, 4]

    def test_run_without_convergence_omits_convergence_status(self, monkeypatch):
        """Test single-run LOC output has no convergence fields by default."""
        loc = check.LowOrderConsistency(_dry_run=True)
        calls = []

        def fake_run_once(do_run, nlib, nburn):
            calls.append((nlib, nburn))
            info = check.CheckInfo()
            info.test_pass = True
            info.nlib = nlib
            info.nburn = nburn
            info.nlib_converged = True
            info.nburn_converged = True
            info.test_pass_nlib = True
            info.test_pass_nburn = True
            return info

        monkeypatch.setattr(loc, '_run_once', fake_run_once)

        info = loc.run(reactor_library=None)

        assert calls == [(1, 1)]
        assert info.test_pass
        assert not hasattr(info, 'nlib')
        assert not hasattr(info, 'nburn')
        assert not hasattr(info, 'test_pass_nlib')
        assert not hasattr(info, 'test_pass_nburn')

    @patch('scale.olm.internal._execute_makefile')
    def test_run_lo_order_writes_check_convergence(self, mock_execute, tmp_path):
        """Test LOW order input data exposes nlib and nburn to templates."""
        work_path = tmp_path / 'work'
        work_path.mkdir()
        template_path = tmp_path / 'loc-template.jt.inp'
        template_path.write_text(
            'libs=[ {{_arpinfo.name}} ]\n'
            'options{ nburn={{convergence_control.nburn}} }\n'
            'cycle{ nlib={{convergence_control.nlib}} }\n'
        )
        assemble_data = {
            'points': [
                {
                    'files': {
                        'lib': 'check/loc/uox.arc.h5',
                        'ii_json': 'hi.ii.json',
                    },
                    'history': {
                        'initialhm': 1.0,
                    },
                    'comp': {'system': {}},
                    '_arpinfo': {
                        'interpvars': {
                            'mod_dens': 0.7,
                        },
                    },
                }
            ]
        }
        (work_path / 'assemble.olm.json').write_text(json.dumps(assemble_data))
        env = {
            'config_file': str(tmp_path / 'config.olm.json'),
            'work_dir': str(work_path),
            'nprocs': 1,
        }
        loc = check.LowOrderConsistency(
            name='loc',
            template=template_path.name,
            convergence={},
            _model={'name': 'test-model'},
            _env=env,
        )
        loc.lo_case = 1

        ii_json_list = loc._LowOrderConsistency__run_lo_order(
            do_run=False,
            nlib=2,
            nburn=4,
        )

        check_dir = work_path / 'check' / 'loc' / 'uox.arc'
        data = json.loads((check_dir / 'data.olm.json').read_text())
        rendered = (check_dir / 'uox.arc.inp').read_text()
        assert data['check']['convergence'] == {'nburn': 4, 'nlib': 2}
        assert data['convergence_control'] == {'nburn': 4, 'nlib': 2}
        assert data['_arpinfo']['name'] == 'test-model'
        assert data['lumped0d'] == {}
        assert 'libs=[ test-model ]' in rendered
        assert 'options{ nburn=4 }' in rendered
        assert 'cycle{ nlib=2 }' in rendered
        assert ii_json_list == [
            (work_path / 'hi.ii.json', check_dir / 'uox.arc.ii.json')
        ]
        mock_execute.assert_called_once()

    @patch('scale.olm.internal._execute_makefile')
    def test_run_lo_order_omits_convergence_template_data_by_default(
        self, mock_execute, tmp_path
    ):
        """Test default LOW order inputs do not require convergence template data."""
        work_path = tmp_path / 'work'
        work_path.mkdir()
        template_path = tmp_path / 'loc-template.jt.inp'
        template_path.write_text(
            'options{ mtu={{history.initialhm}} }\n'
            'cycle{ burn=1 }\n'
        )
        assemble_data = {
            'points': [
                {
                    'files': {
                        'lib': 'check/loc/uox.arc.h5',
                        'ii_json': 'hi.ii.json',
                    },
                    'history': {
                        'initialhm': 1.0,
                    },
                    'comp': {'system': {}},
                    '_arpinfo': {
                        'interpvars': {
                            'mod_dens': 0.7,
                        },
                    },
                }
            ]
        }
        (work_path / 'assemble.olm.json').write_text(json.dumps(assemble_data))
        env = {
            'config_file': str(tmp_path / 'config.olm.json'),
            'work_dir': str(work_path),
            'nprocs': 1,
        }
        loc = check.LowOrderConsistency(
            name='loc',
            template=template_path.name,
            _model={'name': 'test-model'},
            _env=env,
        )
        loc.lo_case = 1

        loc._LowOrderConsistency__run_lo_order(
            do_run=False,
            nlib=1,
            nburn=1,
        )

        check_dir = work_path / 'check' / 'loc' / 'uox.arc'
        data = json.loads((check_dir / 'data.olm.json').read_text())
        rendered = (check_dir / 'uox.arc.inp').read_text()
        assert 'convergence' not in data['check']
        assert 'convergence_control' not in data
        assert data['lumped0d'] == {}
        assert 'options{ mtu=1.000000000000e+00 }' in rendered
        assert 'nburn=' not in rendered
        assert 'nlib=' not in rendered
        mock_execute.assert_called_once()

    @staticmethod
    def _write_library_burnup_check_fixture(tmp_path, burnup_list):
        work_path = tmp_path / 'work'
        work_path.mkdir()
        template_path = tmp_path / 'loc-template.jt.inp'
        template_path.write_text(
            'options{ mtu={{history.initialhm}} }\n'
            '{% for row in history.burndata %}power={{row.power}}\n{% endfor %}'
        )
        assemble_data = {
            'burnup_rtol': 2.0e-2,
            'points': [
                {
                    'files': {
                        'lib': 'check/loc/uox.arc.h5',
                        'ii_json': 'hi.ii.json',
                    },
                    'history': {
                        'initialhm': 1.0,
                        'burndata': [
                            {
                                'power': 39.93052,
                                'burn': 2.5,
                            },
                        ],
                    },
                    'comp': {'system': {}},
                    '_arpinfo': {
                        'interpvars': {
                            'mod_dens': 0.7,
                        },
                        'burnup_list': burnup_list,
                    },
                }
            ]
        }
        (work_path / 'assemble.olm.json').write_text(json.dumps(assemble_data))
        env = {
            'config_file': str(tmp_path / 'config.olm.json'),
            'work_dir': str(work_path),
            'nprocs': 1,
        }
        return check.LowOrderConsistency(
            name='loc',
            template=template_path.name,
            _model={'name': 'test-model'},
            _env=env,
        )

    @patch('scale.olm.internal._execute_makefile')
    def test_run_lo_order_allows_f33_midpoint_grid_for_nlib_one(
        self, mock_execute, tmp_path
    ):
        """Test F33 midpoint burnups can cover a later history endpoint."""
        loc = self._write_library_burnup_check_fixture(tmp_path, [0.0, 49.0])
        loc.lo_case = 1

        loc._LowOrderConsistency__run_lo_order(
            do_run=False,
            nlib=1,
            nburn=1,
        )

        mock_execute.assert_called_once()
        rendered = (
            tmp_path
            / 'work'
            / 'check'
            / 'loc'
            / 'uox.arc'
            / 'uox.arc.inp'
        ).read_text()
        assert 'power=3.993052000000e+01' in rendered

    @patch('scale.olm.internal._execute_makefile')
    def test_run_lo_order_rejects_nlib_past_library_burnup(
        self, mock_execute, tmp_path
    ):
        """Test uncovered library interpolation burnup fails before SCALE."""
        loc = self._write_library_burnup_check_fixture(tmp_path, [0.0, 49.0])
        loc.lo_case = 1

        with pytest.raises(
            ValueError,
            match="nlib=2.*requires low-order library interpolation",
        ):
            loc._LowOrderConsistency__run_lo_order(
                do_run=False,
                nlib=2,
                nburn=1,
            )

        mock_execute.assert_not_called()

    def test_lumped0d_uox_template_conditionally_uses_convergence_control(self):
        """Test stored lumped0d UOX template does not require convergence data."""
        tm = check.core.TemplateManager()
        data = {
            '_': {
                'env': {
                    'work_dir': '/work',
                },
            },
            '_arpinfo': {
                'name': 'uox_quick',
                'interpvars': {
                    'mod_dens': 0.7,
                },
            },
            'history': {
                'initialhm': 1.0,
                'burndata': [
                    {
                        'power': 40.0,
                        'burn': 1.0,
                    },
                ],
            },
            'comp': {
                'system': {
                    'uo2': {
                        'iso': {
                            'u234': 0.01,
                            'u235': 4.0,
                            'u236': 0.02,
                            'u238': 95.97,
                        },
                    },
                },
            },
        }

        rendered = tm.expand('model/origami/lumped0d-uox.jt.inp', data)
        assert 'nburn=' not in rendered
        assert 'nlib=' not in rendered

        data['convergence_control'] = {'nburn': 4, 'nlib': 2}
        rendered = tm.expand('model/origami/lumped0d-uox.jt.inp', data)
        assert 'nburn=4' in rendered
        assert 'nlib=2' in rendered


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
        assert 'target_q_r' in args
        assert 'target_q_ar' in args
        
        # Verify that target values are in reasonable range
        assert 0 <= args['target_q_r'] <= 1
        assert 0 <= args['target_q_ar'] <= 1
    
    def test_schema_loworderconsistency(self):
        """Test schema generation for LowOrderConsistency."""
        schema = check._schema_LowOrderConsistency()
        assert isinstance(schema, dict)
        assert schema['properties']['metric']['enum'] == [
            'grams_per_initial_hm',
            'atom_fraction',
        ]
        assert schema['properties']['nuclide_scaled_difference_min_abs_ylim'][
            'anyOf'
        ] == [{'type': 'number'}, {'type': 'null'}]
        assert (
            schema['properties']['nuclide_scaled_difference_min_abs_ylim']['default']
            is None
        )
        
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
        info.q_r = 0.85
        info.q_ar = 0.90
        
        assert info.name == "TestCheck"
        assert info.q_r == 0.85
        assert info.q_ar == 0.90


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
        assert 0 <= info.q_r <= 1
        assert 0 <= info.q_ar <= 1
        assert info.m > 0
        
        # Most or all points should pass with constant data
        assert info.q_r >= 0.5  # Should have low relative gradients
    
    def test_gridgradient_extreme_values(self):
        """Test GridGradient with extreme coefficient values."""
        rl = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))
        
        # Test with very large values
        rl.coeff = np.ones_like(rl.coeff) * 1e10
        
        grid_grad = check.GridGradient(eps0=1e-20, epsa=1e5, epsr=0.1)
        info = grid_grad.run(rl)
        
        assert info.name == "GridGradient"
        assert np.isfinite(info.q_r)
        assert np.isfinite(info.q_ar)
        assert 0 <= info.q_r <= 1
        assert 0 <= info.q_ar <= 1
    
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
        assert np.isfinite(info.q_r)
        assert np.isfinite(info.q_ar)
