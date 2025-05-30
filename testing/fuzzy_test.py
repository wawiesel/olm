"""
Fuzzy testing for OLM mathematical functions.

This module uses Hypothesis for property-based testing to verify mathematical
properties and correctness across a wide range of inputs.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

import scale.olm.core as core


class TestAxisHandlingFuzzy:
    """Fuzzy tests for axis handling and mathematical properties."""

    @given(x0=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False, allow_infinity=False))
    @settings(max_examples=1000)
    def test_degenerate_axis_creates_valid_increasing_sequence(self, x0):
        """Property: Degenerate axis handling should create a valid increasing sequence."""
        # Skip extreme values that might cause numerical issues
        assume(abs(x0) < 1e10)
        assume(not (abs(x0) < 1e-15 and x0 != 0.0))
        
        x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(x0)
        
        # Core mathematical properties
        assert x1 > x0, f"Second value ({x1}) must be greater than first ({x0})"
        assert x1 != x0, f"Values must be distinct: {x0} vs {x1}"
        assert np.isfinite(x1), f"Result must be finite: {x1}"
        
        # Create axis and verify it's properly increasing
        axis = np.array([x0, x1])
        assert len(np.unique(axis)) == 2, "Axis must have distinct values"
        assert np.all(np.diff(axis) > 0), "Axis must be strictly increasing"

    @given(x0=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=300)
    def test_gradient_calculation_works(self, x0):
        """Property: Resulting axis should enable gradient calculations."""
        assume(abs(x0) < 1e6)
        assume(not (abs(x0) < 1e-12 and x0 != 0.0))
        
        x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(x0)
        axis_values = np.array([x0, x1])
        
        # Gradient calculation should succeed and produce valid results
        try:
            gradient = np.gradient(axis_values)
            assert len(gradient) == 2, "Gradient should have same length as input"
            assert np.all(np.isfinite(gradient)), "Gradient values must be finite"
            assert np.all(gradient > 0), "Gradient should be positive for increasing axis"
        except Exception as e:
            pytest.fail(f"Gradient calculation failed for axis {axis_values}: {e}")

    @given(x0=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_spacing_is_reasonable(self, x0):
        """Property: Spacing between values should be reasonable for numerical work."""
        assume(abs(x0) < 100)
        
        x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(x0)
        spacing = x1 - x0
        
        # Spacing should be positive and not too small for numerical stability
        assert spacing > 0, "Spacing must be positive"
        assert spacing > 1e-15, "Spacing should be above machine epsilon for stability"
        
        # For interpolation/extrapolation, spacing should be reasonable
        # For very small values, relative spacing can be large, which is fine
        if abs(x0) > 1.0:
            relative_spacing = spacing / abs(x0)
            assert relative_spacing < 1, "For large values, relative spacing should be reasonable"


class TestNumericalPropertiesFuzzy:
    """Fuzzy tests for numerical properties and stability."""

    @given(values=arrays(np.float64, shape=st.integers(1, 50), 
                        elements=st.floats(min_value=-1e6, max_value=1e6, 
                                         allow_nan=False, allow_infinity=False)))
    @settings(max_examples=200)
    def test_rounding_stability(self, values):
        """Property: Rounding operations should be stable and consistent."""
        assume(len(values) > 0)
        assume(not np.any(np.isnan(values)))
        assume(not np.any(np.isinf(values)))
        
        # Test rounding to 6 decimal places (used in extract_axes)
        rounded = np.round(values, 6)
        
        # Rounding should be idempotent
        double_rounded = np.round(rounded, 6)
        np.testing.assert_array_equal(rounded, double_rounded, 
                                      "Rounding should be idempotent")
        
        # Rounding should preserve array properties
        assert len(rounded) == len(values), "Length should be preserved"
        assert rounded.shape == values.shape, "Shape should be preserved"
        
        # Rounded values should be close to original
        max_diff = np.max(np.abs(values - rounded))
        assert max_diff <= 1e-6, f"Rounding error too large: {max_diff}"

    @given(axis_data=arrays(np.float64, shape=st.integers(2, 20),
                           elements=st.floats(min_value=-1e3, max_value=1e3,
                                            allow_nan=False, allow_infinity=False)))
    @settings(max_examples=100)
    def test_monotonic_axis_gradient_stability(self, axis_data):
        """Property: Monotonic axes should have stable gradient calculations."""
        assume(len(axis_data) >= 2)
        assume(not np.any(np.isnan(axis_data)))
        assume(not np.any(np.isinf(axis_data)))
        
        # Create a proper monotonic axis by sorting and removing near-duplicates
        sorted_data = np.sort(axis_data)
        unique_data = [sorted_data[0]]
        
        for val in sorted_data[1:]:
            if abs(val - unique_data[-1]) > 1e-10:  # Avoid near-duplicates
                unique_data.append(val)
        
        if len(unique_data) >= 2:
            axis = np.array(unique_data)
            
            # Properties of a good monotonic axis
            assert np.all(np.diff(axis) > 0), "Axis should be strictly increasing"
            assert len(np.unique(axis)) == len(axis), "All values should be unique"
            
            # Gradient should work without issues
            try:
                gradient = np.gradient(axis)
                assert len(gradient) == len(axis), "Gradient length should match axis length"
                assert np.all(np.isfinite(gradient)), "All gradient values should be finite"
                assert np.all(gradient > 0), "Gradient should be positive for increasing axis"
            except Exception as e:
                pytest.fail(f"Gradient failed for monotonic axis: {e}")


class TestAxisTransformationFuzzy:
    """Fuzzy tests for axis transformation properties."""

    @given(shape=st.tuples(st.integers(1, 10), st.integers(1, 10), st.integers(1, 10)))
    @settings(max_examples=100)
    def test_shape_transformation_consistency(self, shape):
        """Property: Shape transformations should be consistent and valid."""
        original_shape = list(shape)
        
        # Apply transformation logic (convert 1s to 2s for degenerate axes)
        transformed_shape = [2 if size == 1 else size for size in original_shape]
        
        # Properties that should hold
        assert len(transformed_shape) == len(original_shape), "Shape length preserved"
        assert all(s >= 2 for s in transformed_shape), "All dimensions should be >= 2"
        assert all(s >= orig for s, orig in zip(transformed_shape, original_shape)), \
               "Transformed shape should not shrink dimensions"
        
        # Count transformations
        degenerate_axes = sum(1 for s in original_shape if s == 1)
        changes = sum(1 for orig, trans in zip(original_shape, transformed_shape) if orig != trans)
        assert changes == degenerate_axes, "Should transform exactly the degenerate axes"

    @given(coefficient_shape=st.tuples(st.integers(1, 5), st.integers(1, 5)))
    @settings(max_examples=50)
    def test_coefficient_expansion_validity(self, coefficient_shape):
        """Property: Coefficient expansion should preserve data structure."""
        # Simulate coefficient array expansion for degenerate axes
        original_coeff = np.random.rand(*coefficient_shape)
        
        for axis in range(len(coefficient_shape)):
            if coefficient_shape[axis] == 1:
                # Test the expansion operation used in ReactorLibrary
                expanded_coeff = np.repeat(original_coeff, 2, axis=axis)
                
                # Verify expansion properties
                expected_shape = list(coefficient_shape)
                expected_shape[axis] = 2
                assert expanded_coeff.shape == tuple(expected_shape), \
                       f"Shape should be {expected_shape}, got {expanded_coeff.shape}"
                
                # Data should be preserved (duplicated along the axis)
                assert not np.array_equal(expanded_coeff, original_coeff) or coefficient_shape[axis] != 1, \
                       "Expansion should change the array when axis size was 1"


class TestRealWorldScenariosFuzzy:
    """Fuzzy tests for real-world reactor physics scenarios."""

    @given(reactor_param=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_reactor_parameter_axis_handling(self, reactor_param):
        """Property: Reactor parameters should create valid interpolation axes."""
        # Test with values similar to mod_dens, burnup, enrichment, etc.
        x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(reactor_param)
        
        # Create parameter axis
        param_axis = np.array([reactor_param, x1])
        
        # Essential properties for reactor physics interpolation
        assert np.all(param_axis >= 0), "Reactor parameters should be non-negative"
        assert np.all(np.diff(param_axis) > 0), "Parameter axis should be increasing"
        assert len(np.unique(param_axis)) == 2, "Parameter values should be distinct"
        
        # Should work for linear interpolation
        try:
            # Test that we can do basic interpolation operations
            mid_point = (param_axis[0] + param_axis[1]) / 2
            assert param_axis[0] < mid_point < param_axis[1], "Midpoint should be between endpoints"
            
            # Test gradient for finite differences
            gradient = np.gradient(param_axis)
            assert np.all(gradient > 0), "Gradient should be positive"
        except Exception as e:
            pytest.fail(f"Interpolation setup failed for parameter {reactor_param}: {e}")

    @given(values=st.lists(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                          min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_multiple_parameter_axis_consistency(self, values):
        """Property: Multiple parameters should create consistent axis systems."""
        assume(len(values) > 0)
        assume(all(v >= 0 for v in values))
        
        # Process multiple parameters as might happen in a reactor library
        processed_axes = []
        for param in values:
            if abs(param) < 1e-10:  # Treat very small as zero
                param = 0.0
            x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(param)
            axis = np.array([param, x1])
            processed_axes.append(axis)
        
        # All axes should be valid
        for i, axis in enumerate(processed_axes):
            assert len(axis) == 2, f"Axis {i} should have exactly 2 points"
            assert np.all(np.diff(axis) > 0), f"Axis {i} should be strictly increasing"
            assert np.all(np.isfinite(axis)), f"Axis {i} should have finite values"
            
            # Gradient should work
            try:
                grad = np.gradient(axis)
                assert np.all(grad > 0), f"Gradient for axis {i} should be positive"
            except Exception as e:
                pytest.fail(f"Gradient failed for axis {i} with values {axis}: {e}")


class TestEdgeCaseRobustnessFuzzy:
    """Fuzzy tests for edge case robustness."""

    @given(st.one_of(st.just(0.0), st.just(-0.0), 
                     st.floats(min_value=-1e-10, max_value=1e-10, allow_nan=False, allow_infinity=False)))
    @settings(max_examples=100)
    def test_near_zero_handling(self, x0):
        """Property: Near-zero values should be handled robustly."""
        x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(x0)
        
        # Basic correctness
        assert x1 > x0, "Second value should be greater than first"
        assert np.isfinite(x1), "Result should be finite"
        
        # Should work in gradient calculations
        axis = np.array([x0, x1])
        gradient = np.gradient(axis)
        assert np.all(np.isfinite(gradient)), "Gradient should be finite"
        assert np.all(gradient > 0), "Gradient should be positive"

    @given(x0=st.floats(min_value=1e6, max_value=1e9, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_large_value_handling(self, x0):
        """Property: Large values should be handled without overflow."""
        x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(x0)
        
        # No overflow or invalid results
        assert np.isfinite(x1), "Large value result should be finite"
        assert x1 > x0, "Second value should be greater"
        
        # Spacing should be reasonable relative to magnitude
        spacing = x1 - x0
        relative_spacing = spacing / x0
        assert 0 < relative_spacing < 1, "Relative spacing should be reasonable" 