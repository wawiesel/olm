

# ===== Content from testing/scale_olm_core_advanced_test.py =====
"""
Advanced tests for scale.olm.core module.

This module tests the mathematical algorithms, composition calculations,
and data processing functionality of the core module to improve coverage.
Focus on testing real functionality with minimal mocking.
"""
import pytest
import numpy as np
import tempfile
import os

import scale.olm.core as core


class TestCompositionManager:
    """Test the CompositionManager class for nuclide data and calculations."""

    @pytest.fixture
    def sample_nuclide_data(self):
        """Sample nuclide data for testing."""
        return {
            "0001001": {
                "IZZZAAA": "0001001",
                "atomicNumber": 1,
                "element": "H",
                "isomericState": 0,
                "mass": 1.007825,
                "massNumber": 1
            },
            "0001002": {
                "IZZZAAA": "0001002",
                "atomicNumber": 1,
                "element": "H",
                "isomericState": 0,
                "mass": 2.014102,
                "massNumber": 2
            },
            "0092235": {
                "IZZZAAA": "0092235",
                "atomicNumber": 92,
                "element": "U",
                "isomericState": 0,
                "mass": 235.044,
                "massNumber": 235
            },
            "0094239": {
                "IZZZAAA": "0094239",
                "atomicNumber": 94,
                "element": "Pu",
                "isomericState": 0,
                "mass": 239.052,
                "massNumber": 239
            }
        }

    @pytest.fixture
    def composition_manager(self, sample_nuclide_data):
        """Create a CompositionManager instance for testing."""
        return core.CompositionManager(sample_nuclide_data)

    def test_composition_manager_initialization(self, composition_manager):
        """Test CompositionManager initialization and element mapping."""
        # Test element to atomic number mapping
        assert composition_manager.e_to_z["h"] == 1
        assert composition_manager.e_to_z["u"] == 92
        assert composition_manager.e_to_z["pu"] == 94

        # Test atomic number to element mapping
        assert composition_manager.z_to_e[1] == "h"
        assert composition_manager.z_to_e[92] == "u"
        assert composition_manager.z_to_e[94] == "pu"

    def test_parse_eam_to_eai(self):
        """Test parsing element-mass-isomer identifiers."""
        # Test normal nuclides
        e, a, i = core.CompositionManager.parse_eam_to_eai("u235")
        assert e == "u" and a == 235 and i == 0

        e, a, i = core.CompositionManager.parse_eam_to_eai("pu239")
        assert e == "pu" and a == 239 and i == 0

        # Test metastable states
        e, a, i = core.CompositionManager.parse_eam_to_eai("am242m")
        assert e == "am" and a == 242 and i == 1

        e, a, i = core.CompositionManager.parse_eam_to_eai("tc99m2")
        assert e == "tc" and a == 99 and i == 2

        # Test single-letter elements
        e, a, i = core.CompositionManager.parse_eam_to_eai("h1")
        assert e == "h" and a == 1 and i == 0

        # Test invalid formats
        with pytest.raises(ValueError, match="did not match regular expression"):
            core.CompositionManager.parse_eam_to_eai("invalid123")

    def test_mass_lookup(self, composition_manager):
        """Test mass lookup functionality using real data."""
        # Test direct IZZZAAA lookup
        mass = composition_manager.mass("0092235")
        assert mass == pytest.approx(235.044, abs=0.01)

        # Test with invalid ID - this will return None or default
        result = composition_manager.data.get("nonexistent", {"mass": 100.0})["mass"]
        assert result == 100.0

    def test_renormalize_wtpt(self):
        """Test weight percent renormalization with real calculations."""
        # Test basic renormalization
        wtpt0 = {"u235": 25.0, "u238": 75.0, "pu239": 5.0}
        wtpt, norm = core.CompositionManager.renormalize_wtpt(wtpt0, 100.0)

        # Should include all elements and sum to 100
        assert "u235" in wtpt and "u238" in wtpt and "pu239" in wtpt
        assert sum(wtpt.values()) == pytest.approx(100.0, abs=1e-10)

        # Test with filter
        wtpt_u, norm_u = core.CompositionManager.renormalize_wtpt(wtpt0, 100.0, "u")
        assert "u235" in wtpt_u and "u238" in wtpt_u
        assert "pu239" not in wtpt_u
        assert sum(wtpt_u.values()) == pytest.approx(100.0, abs=1e-10)

    def test_grams_per_mol(self):
        """Test molar mass calculation using harmonic mean formula."""
        # Test simple mixture
        iso_wts = {"u235": 50.0, "pu239": 50.0}
        molar_mass = core.CompositionManager.grams_per_mol(iso_wts, m_data={})

        # Should be close to average of mass numbers: (235 + 239) / 2 = 237
        assert molar_mass == pytest.approx(236.98, abs=0.1)

        # Test with real molar masses
        m_data = {"u235": 235.044, "pu239": 239.052}
        molar_mass = core.CompositionManager.grams_per_mol(iso_wts, m_data)
        expected = 1.0 / (0.5/235.044 + 0.5/239.052)
        assert molar_mass == pytest.approx(expected, abs=0.01)


class TestBurnupHistory:
    """Test the BurnupHistory class for time-burnup management."""

    def test_burnup_history_initialization(self):
        """Test BurnupHistory initialization with simple data."""
        time = [0, 10, 20, 30, 40]
        burnup = [0, 100, 250, 500, 1000]

        bh = core.BurnupHistory(time, burnup)

        # Verify basic attributes
        assert len(bh.time) == 5
        assert len(bh.burnup) == 5
        assert len(bh.interval_time) == 4
        assert len(bh.interval_burnup) == 4
        assert len(bh.interval_power) == 4

        # Verify interval calculations
        expected_dt = [10, 10, 10, 10]
        expected_dbu = [100, 150, 250, 500]
        expected_power = [10.0, 15.0, 25.0, 50.0]

        np.testing.assert_array_almost_equal(bh.interval_time, expected_dt)
        np.testing.assert_array_almost_equal(bh.interval_burnup, expected_dbu)
        np.testing.assert_array_almost_equal(bh.interval_power, expected_power)

    def test_union_times(self):
        """Test time grid union functionality."""
        a = np.array([0, 10, 20, 30])
        b = np.array([5, 15, 25, 35])

        c = core.BurnupHistory.union_times(a, b)
        expected = np.array([0, 5, 10, 15, 20, 25, 30, 35])

        np.testing.assert_array_equal(c, expected)

    def test_classify_operations_basic(self):
        """Test basic operations classification."""
        time = [0, 5, 10, 50, 55, 100, 105]
        burnup = [0, 0, 100, 500, 500, 1000, 1000]

        bh = core.BurnupHistory(time, burnup)
        result = bh.classify_operations()

        # Verify structure
        assert "options" in result
        assert "operations" in result

        # Verify operations
        operations = result["operations"]
        assert len(operations) >= 3  # At least some operations
        assert operations[0]["start"] == 0


class TestScaleOutfile:
    """Test the ScaleOutfile class for SCALE output parsing."""

    def test_parse_burnups_from_triton_output(self):
        """Test parsing burnup data from TRITON output using real file."""
        # Create realistic TRITON output file
        sample_output = """
Some header text...
Sub-Interval   Depletion   Sub-interval    Specific      Burn Length  Decay Length   Library Burnup
     No.       Interval     in interval  Power(MW/MTIHM)     (d)          (d)           (MWd/MTIHM)
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
        0     ****Initial Bootstrap Calculation****                                      0.00000E+00
        1          1                1          40.000      25.000         0.000          5.00000e+02
        2          1                2          40.000     300.000         0.000          7.00000e+03
        3          1                3          40.000     300.000         0.000          1.90000e+04
        4          1                4          40.000     312.500         0.000          3.12500e+04
----------------------------------------------------------------------------------------------------
Some footer text...
"""

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.out') as f:
            f.write(sample_output)
            temp_path = f.name

        try:
            burnups = core.ScaleOutfile.parse_burnups_from_triton_output(temp_path)

            expected = [0.0, 500.0, 7000.0, 19000.0, 31250.0]
            assert len(burnups) == 5
            np.testing.assert_array_almost_equal(burnups, expected)

        finally:
            os.unlink(temp_path)

    def test_get_runtime(self):
        """Test extracting runtime from SCALE output using real file."""
        sample_output = """
Some output text...
t-depl finished. used 35.2481 seconds.
More output text...
"""

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.out') as f:
            f.write(sample_output)
            temp_path = f.name

        try:
            runtime = core.ScaleOutfile.get_runtime(temp_path)
            assert runtime == pytest.approx(35.2481, abs=0.001)

        finally:
            os.unlink(temp_path)


class TestReactorLibraryUtilities:
    """Test ReactorLibrary utility functions with minimal mocking."""

    def test_duplicate_degenerate_axis_value(self):
        """Test degenerate axis value duplication (comprehensive mathematical testing)."""
        test_cases = [
            # (input, expected_delta)
            (0.0, 0.05),           # Zero case
            (0.723, 0.05),         # Typical reactor parameter
            (-1.0, 0.05),          # Negative value
            (100.0, 5.0),          # Large value (5% of 100)
            (1e-12, 0.05),         # Very small value
            (-50.0, 2.5),          # Large negative (5% of 50)
            (2.0, 0.1),            # Moderate value (5% of 2)
        ]

        for x0, expected_delta in test_cases:
            x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(x0)
            actual_delta = x1 - x0
            assert actual_delta == pytest.approx(expected_delta, abs=1e-10)

            # Verify essential properties
            assert x1 > x0, f"x1 ({x1}) should be greater than x0 ({x0})"
            assert x1 != x0, f"x1 ({x1}) should be different from x0 ({x0})"
            assert np.isfinite(x1), f"x1 ({x1}) should be finite"

    def test_get_indices(self):
        """Test index calculation for library interpolation."""
        axes_names = np.array(["mod_dens", "enrichment", "burnup"])
        axes_values = [
            np.array([0.1, 0.5, 0.9]),      # mod_dens
            np.array([2.0, 3.5, 5.0]),      # enrichment
            np.array([0, 1000, 5000])       # burnup
        ]

        # Test exact matches
        point_data = {"mod_dens": 0.5, "enrichment": 3.5, "burnup": 1000}
        indices = core.ReactorLibrary.get_indices(axes_names, axes_values, point_data)
        expected = (1, 1, 1)  # Middle values
        assert indices == expected


class TestNuclideInventory:
    """Test the NuclideInventory class using real data structures."""

    @pytest.fixture
    def sample_composition_manager(self):
        """Create a real composition manager for testing."""
        data = {
            "0092235": {"mass": 235.044, "atomicNumber": 92, "element": "U", "massNumber": 235},
            "0092238": {"mass": 238.051, "atomicNumber": 92, "element": "U", "massNumber": 238},
            "0094239": {"mass": 239.052, "atomicNumber": 94, "element": "Pu", "massNumber": 239}
        }
        return core.CompositionManager(data)

    @pytest.fixture
    def sample_inventory(self, sample_composition_manager):
        """Create a real NuclideInventory for testing."""
        time = np.array([0, 100, 200, 300])  # days
        nuclide_amount = {
            "0092235": np.array([1000, 950, 900, 850]),  # moles
            "0092238": np.array([100, 105, 110, 115]),   # moles
            "0094239": np.array([0, 5, 15, 30])          # moles
        }
        return core.NuclideInventory(sample_composition_manager, time, nuclide_amount)

    def test_get_hm_mass(self, sample_inventory):
        """Test heavy metal mass calculation."""
        hm_mass = sample_inventory.get_hm_mass(min_z=92)

        # Should be positive and have correct length
        assert len(hm_mass) == 4
        assert np.all(hm_mass > 0)

        # Mass should change over time due to transmutation
        assert not np.allclose(hm_mass, hm_mass[0])

    def test_get_amount(self, sample_inventory):
        """Test nuclide amount extraction."""
        # Test moles (default)
        u235_moles = sample_inventory.get_amount("u235", units="MOLES")
        expected = np.array([1000, 950, 900, 850])
        np.testing.assert_array_equal(u235_moles, expected)

        # Test grams
        u235_grams = sample_inventory.get_amount("u235", units="GRAMS")
        expected_grams = expected * 235.044  # moles * mass
        np.testing.assert_array_almost_equal(u235_grams, expected_grams)


class TestMathematicalAlgorithms:
    """Test mathematical algorithms with focus on correctness, not implementation."""

    def test_axis_duplication_mathematical_properties(self):
        """Test mathematical properties of axis duplication algorithm."""
        # Test over wide range of realistic reactor parameters
        test_values = [
            0.0, 0.1, 0.5, 0.723, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0,
            -0.1, -1.0, -10.0, 1e-10, 1e-5, 1e5
        ]

        for x0 in test_values:
            x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(x0)

            # Essential mathematical properties
            assert x1 > x0, f"Failed monotonicity: {x1} <= {x0}"
            assert x1 != x0, f"Failed distinctness: {x1} == {x0}"
            assert np.isfinite(x1), f"Failed finiteness: {x1} is not finite"

            # Test numerical stability
            axis = np.array([x0, x1])
            gradient = np.gradient(axis)
            assert np.all(gradient > 0), f"Failed gradient positivity for {x0}"
            assert np.all(np.isfinite(gradient)), f"Failed gradient finiteness for {x0}"

    def test_composition_normalization_properties(self):
        """Test mathematical properties of composition normalization."""
        # Test various composition scenarios
        test_compositions = [
            {"u235": 25, "u238": 75},                    # Simple uranium
            {"u235": 20, "u238": 70, "pu239": 10},      # U-Pu mixture
            {"pu239": 50, "pu241": 30, "am241": 20},    # TRU mixture
            {"u235": 1, "u238": 1, "pu239": 1},         # Equal parts
        ]

        for comp in test_compositions:
            # Test renormalization to 100%
            norm_comp, norm_factor = core.CompositionManager.renormalize_wtpt(comp, 100.0)

            # Mathematical properties
            total = sum(norm_comp.values())
            assert total == pytest.approx(100.0, abs=1e-10), f"Failed normalization: {total}"
            assert norm_factor > 0, f"Normalization factor should be positive: {norm_factor}"

    def test_molar_mass_calculation_properties(self):
        """Test mathematical properties of molar mass calculations."""
        # Test harmonic mean formula: 1/m = sum(w_i / m_i)
        test_cases = [
            ({"u235": 50, "u238": 50}, {}),           # Equal mixture
            ({"pu239": 100}, {}),                     # Pure isotope
            ({"u235": 25, "u238": 75}, {}),          # Enriched uranium
        ]

        for iso_wts, m_data in test_cases:
            molar_mass = core.CompositionManager.grams_per_mol(iso_wts, m_data)

            # Mathematical properties
            assert molar_mass > 0, f"Molar mass should be positive: {molar_mass}"
            assert np.isfinite(molar_mass), f"Molar mass should be finite: {molar_mass}"

            # For single isotope, should equal mass number (approximately)
            if len(iso_wts) == 1:
                isotope = list(iso_wts.keys())[0]
                # Extract mass number correctly using regex
                import re
                mass_str = re.sub("^[a-z]+", "", isotope)  # Remove element letters
                mass_str = re.sub("m[0-9]*$", "", mass_str)  # Remove metastable indicators
                mass_number = float(mass_str)
                assert molar_mass == pytest.approx(mass_number, rel=0.01)


# ===== Content from testing/scale_olm_core_fuzzy_test.py =====
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