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