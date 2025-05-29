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
