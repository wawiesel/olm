"""Focused tests for LowOrderConsistency convergence helpers."""

import numpy as np
import pytest
from unittest.mock import patch

import scale.olm.check as check


@patch('scale.olm.core.RelAbsHistogram.plot_hist')
@patch.object(check.LowOrderConsistency, 'make_time_quality_plot')
def test_initial_identity_failure_fails_test_one(
    mock_time_quality_plot, mock_hist_plot, tmp_path
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
    assert info.test_pass_initial is False
    assert info.test_pass_time is True
    assert info.test_pass is False
    assert loc._failure_reasons(info) == [
        'test 1.1 failed: q_r at t=0 must be 1.000',
        'test 1.2 failed: q_ar at t=0 must be 1.000',
    ]
    mock_time_quality_plot.assert_called_once()
    mock_hist_plot.assert_called_once()


def test_final_convergence_outputs_copy_to_nominal_path(tmp_path):
    loc = check.LowOrderConsistency(_dry_run=True, convergence={})
    loc.work_path = tmp_path
    loc.base_check_path = tmp_path / 'check' / 'loc'
    loc.convergence_check_path = loc.base_check_path / '_convergence'
    loc.check_path = loc.convergence_check_path / 'nlib0002' / 'nburn0004'
    source_point = loc.check_path / 'uox.arc'
    source_point.mkdir(parents=True)
    (source_point / 'uox.arc.inp').write_text('input')
    (loc.check_path / 'hist.png').write_text('hist')
    loc.base_check_path.mkdir(parents=True, exist_ok=True)
    (loc.base_check_path / 'stale.txt').write_text('stale')
    info = check.CheckInfo()
    info.hist_image = str(loc.check_path / 'hist.png')
    info.time_quality_image = str(loc.check_path / 'q_r-q_ar-by-time.png')

    loc._copy_final_check_outputs_to_nominal(info)

    assert not (loc.base_check_path / 'stale.txt').exists()
    assert (
        loc.base_check_path / 'uox.arc' / 'uox.arc.inp'
    ).read_text() == 'input'
    assert info.hist_image == str(loc.base_check_path / 'hist.png')
    assert info.time_quality_image == str(
        loc.base_check_path / 'q_r-q_ar-by-time.png'
    )


def test_convergence_summary_status_and_diagnostics(tmp_path):
    loc = check.LowOrderConsistency(
        _dry_run=True,
        target_q_r=0.7,
        target_q_ar=0.95,
        convergence={'nlib_max': 4, 'nburn_max': 4},
    )
    loc.check_path = tmp_path
    info = check.CheckInfo()
    info.nlib = 4
    info.nburn = 1
    info.nlib_delta_q_r = 0.01
    info.nlib_delta_q_ar = 0.02
    info.nlib_convergence_stop = 'max'
    info.time_quality = [
        {
            'time': 0.0,
            'time_days': 0.0,
            'q_r': 0.8,
            'q_ar': 0.96,
            'target_q_r': 0.7,
            'target_q_ar': 0.95,
            'test_pass': True,
            'limiting_score': 'q_r',
            'limiting_score_shortfall': 0.0,
        },
        {
            'time': 86400.0,
            'time_days': 1.0,
            'q_r': 0.6,
            'q_ar': 0.90,
            'target_q_r': 0.7,
            'target_q_ar': 0.95,
            'test_pass': False,
            'limiting_score': 'q_r',
            'limiting_score_shortfall': 0.1,
        },
    ]
    info.time_quality_image = str(tmp_path / 'q_r-q_ar-by-time.png')
    info.nlib_history = [dict(nlib=1, nburn=1, q_r=0.5, q_ar=0.8)]
    info.nburn_history = [dict(nlib=4, nburn=1, q_r=0.6, q_ar=0.9)]

    status = loc._convergence_status(info)
    row = loc._convergence_summary(info)
    loc._write_convergence_diagnostics(info)

    assert status['nlib']['result'] == 'max'
    assert row['result'] == 'fail q_r/q_ar'
    assert row['time_days'] == pytest.approx(1.0)
    assert [r['nlib'] for r in info.convergence_history] == [1, 4]
    assert (tmp_path / 'q_r-q_ar-convergence.png').exists()
