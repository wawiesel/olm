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
            'convergence',
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
        
        expected_keys = {
            'eps0',
            'epsa',
            'epsr',
            'metric',
            'convergence',
            'target_q1',
            'target_q2',
            'nuclide_compare',
            'template',
            'name',
        }
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

    @patch('scale.olm.core.RelAbsHistogram.plot_hist')
    @patch.object(check.LowOrderConsistency, 'make_time_quality_plot')
    def test_time_quality_failure_fails_check(
        self, mock_time_quality_plot, mock_hist_plot, tmp_path
    ):
        """Test every high-order time point must satisfy q1/q2 targets."""
        loc = check.LowOrderConsistency(
            metric='atom_fraction',
            target_q1=0.7,
            target_q2=0.7,
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

        assert info.q1 == pytest.approx(0.75)
        assert info.q2 == pytest.approx(0.75)
        assert info.test_pass_q1 is True
        assert info.test_pass_q2 is True
        assert info.test_pass_time is False
        assert info.test_pass is False
        assert info.time_quality[1]['q1'] == pytest.approx(0.0)
        assert info.time_quality[1]['q2'] == pytest.approx(0.0)
        assert info.time_quality[1]['burnup_gwd_per_mtihm'] == pytest.approx(10.0)
        assert info.worst_time_quality['index'] == 1
        assert info.worst_time_quality['time_days'] == pytest.approx(1.0)
        assert info.worst_time_quality['burnup'] == pytest.approx(10000.0)
        assert info.worst_time_quality['limiting_score'] == 'q1'
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

    @patch.object(check.LowOrderConsistency, 'make_time_quality_plot')
    def test_time_quality_plot_includes_convergence_history(
        self, mock_time_quality_plot, tmp_path
    ):
        """Test convergence diagnostics are added as shaded q1/q2 time curves."""
        loc = check.LowOrderConsistency(
            target_q1=0.7,
            target_q2=0.95,
            _dry_run=True,
        )
        final_time_quality = [
            {'time': 0.0, 'time_days': 0.0, 'q1': 1.0, 'q2': 1.0},
            {'time': 86400.0, 'time_days': 1.0, 'q1': 0.9, 'q2': 0.97},
        ]
        background_time_quality = [
            {'time': 0.0, 'time_days': 0.0, 'q1': 1.0, 'q2': 1.0},
            {'time': 86400.0, 'time_days': 1.0, 'q1': 0.8, 'q2': 0.96},
        ]
        info = check.CheckInfo()
        info.time_quality = final_time_quality
        info.time_quality_image = str(tmp_path / 'q1-q2-by-time.png')
        info.nlib = 2
        info.nburn = 1
        info.nlib_history = [
            {'nlib': 1, 'nburn': 1, 'time_quality': background_time_quality},
            {'nlib': 2, 'nburn': 1, 'time_quality': final_time_quality},
        ]
        info.nburn_history = []

        loc._write_time_quality_plot(info)

        mock_time_quality_plot.assert_called_once()
        assert mock_time_quality_plot.call_args.kwargs['background'] == [
            {'label': 'nlib_history', 'time_quality': background_time_quality}
        ]

    @patch.object(check.LowOrderConsistency, 'make_convergence_time_quality_plot')
    def test_convergence_time_quality_uses_worst_scores_over_time(
        self, mock_convergence_plot, tmp_path
    ):
        """Test convergence rows summarize minimum q1/q2 across time."""
        loc = check.LowOrderConsistency(
            target_q1=0.7,
            target_q2=0.95,
            convergence={},
            _dry_run=True,
        )
        loc.work_path = tmp_path
        loc.base_check_path = tmp_path / 'check' / 'loc'
        loc.base_check_path.mkdir(parents=True)
        info = check.CheckInfo()
        info.nlib_history = [
            {
                'nlib': 1,
                'nburn': 1,
                'time_quality': [
                    {
                        'time': 0.0,
                        'time_days': 0.0,
                        'burnup_gwd_per_mtu': 0.0,
                        'q1': 1.0,
                        'q2': 1.0,
                        'target_q1': 0.7,
                        'target_q2': 0.95,
                        'limiting_score': 'q1',
                        'limiting_score_value': 1.0,
                        'limiting_score_target': 0.7,
                        'limiting_score_shortfall': 0.0,
                    },
                    {
                        'time': 86400.0,
                        'time_days': 1.0,
                        'burnup_gwd_per_mtu': 10.0,
                        'q1': 0.69,
                        'q2': 0.98,
                        'target_q1': 0.7,
                        'target_q2': 0.95,
                        'limiting_score': 'q1',
                        'limiting_score_value': 0.69,
                        'limiting_score_target': 0.7,
                        'limiting_score_shortfall': 0.01,
                    },
                    {
                        'time': 172800.0,
                        'time_days': 2.0,
                        'burnup_gwd_per_mtu': 20.0,
                        'q1': 0.8,
                        'q2': 0.94,
                        'target_q1': 0.7,
                        'target_q2': 0.95,
                        'limiting_score': 'q2',
                        'limiting_score_value': 0.94,
                        'limiting_score_target': 0.95,
                        'limiting_score_shortfall': 0.01,
                    },
                ],
            },
        ]
        info.nburn_history = []

        loc._write_convergence_time_quality(info)

        row = info.convergence_time_quality[0]
        assert row['nlib'] == 1
        assert row['nburn'] == 1
        assert row['q1'] == pytest.approx(0.69)
        assert row['q2'] == pytest.approx(0.94)
        assert row['pass'] == 'fail q1/q2'
        assert row['failed_scores'] == ['q1', 'q2']
        assert row['time_days'] == pytest.approx(1.0)
        assert row['burnup_gwd_per_mtu'] == pytest.approx(10.0)
        mock_convergence_plot.assert_called_once_with(
            loc.base_check_path / 'q1-q2-by-convergence.png',
            info.convergence_time_quality,
            loc.target_q1,
            loc.target_q2,
        )

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

    def test_convergence_check_path_uses_subdirectories(self, tmp_path):
        """Test convergence runs isolate outputs in parameter-specific directories."""
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

        loc._set_check_path_for_convergence(nlib=2, nburn=2)

        assert loc.check_path == tmp_path / 'check' / 'loc' / 'nlib0002' / 'nburn0002'
        assert loc.check_dir == Path('check/loc/nlib0002/nburn0002')

    def test_scores_converged_uses_q_score_deltas(self):
        """Test convergence is based on q1 and q2 score changes."""
        loc = check.LowOrderConsistency(
            _dry_run=True,
            convergence={
                'q1_stop_criteria': 0.01,
                'q2_stop_criteria': 0.02,
            },
        )
        previous = check.CheckInfo()
        previous.q1 = 0.90
        previous.q2 = 0.95
        current = check.CheckInfo()
        current.q1 = 0.905
        current.q2 = 0.969

        assert loc._scores_converged(previous, current, fixed_grid=False)

        current.q2 = 0.971
        assert not loc._scores_converged(previous, current, fixed_grid=False)

    def test_run_nlib_convergence_doubles_until_scores_converge(self, monkeypatch):
        """Test nlib convergence runs until q-score deltas meet the stop criteria."""
        loc = check.LowOrderConsistency(
            _dry_run=True,
            convergence={
                'nlib_start': 1,
                'nlib_max': 4,
                'q1_stop_criteria': 0.01,
                'q2_stop_criteria': 0.01,
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
            info.q1, info.q2 = scores[nlib]
            info.test_pass = True
            info.test_pass_q1 = True
            info.test_pass_q2 = True
            info.mean_abs_diff = 0.0
            info.mean_rel_diff = 0.0
            return info

        monkeypatch.setattr(loc, '_run_once', fake_run_once)

        info = loc._run_nlib_convergence(do_run=False, nburn=1)

        assert calls == [(1, 1), (2, 1), (4, 1)]
        assert info.nlib == 4
        assert info.nlib_converged
        assert info.test_pass_nlib
        assert info.nlib_delta_q1 == pytest.approx(0.005)
        assert info.nlib_delta_q2 == pytest.approx(0.005)
        assert [row['nlib'] for row in info.nlib_history] == [1, 2, 4]

    def test_run_nburn_convergence_uses_converged_nlib(self, monkeypatch):
        """Test nburn convergence holds nlib fixed after nlib convergence."""
        loc = check.LowOrderConsistency(
            _dry_run=True,
            convergence={
                'nburn_start': 1,
                'nburn_max': 4,
                'q1_stop_criteria': 0.01,
                'q2_stop_criteria': 0.01,
            },
        )
        nlib_info = check.CheckInfo()
        nlib_info.nlib = 2
        nlib_info.nburn = 1
        nlib_info.q1 = 0.50
        nlib_info.q2 = 0.60
        nlib_info.test_pass = True
        nlib_info.test_pass_q1 = True
        nlib_info.test_pass_q2 = True
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
            info.q1, info.q2 = scores[nburn]
            info.test_pass = True
            info.test_pass_q1 = True
            info.test_pass_q2 = True
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
            'options{ nburn={{check.convergence.nburn}} }\n'
            'cycle{ nlib={{check.convergence.nlib}} }\n'
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
        assert 'options{ mtu=1.000000000000e+00 }' in rendered
        assert 'nburn=' not in rendered
        assert 'nlib=' not in rendered
        mock_execute.assert_called_once()


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
