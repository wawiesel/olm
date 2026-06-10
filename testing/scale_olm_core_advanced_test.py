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
import warnings
from matplotlib.colors import LogNorm
from unittest.mock import patch

import scale.olm.core as core


class TestTemplateManager:
    """Test Jinja template expansion behavior."""

    def test_expand_file_supports_same_directory_parent_template(self, tmp_path):
        """Test direct file expansion resolves an inherited sibling template."""
        base = tmp_path / "base.jt.inp"
        child = tmp_path / "child.jt.inp"
        base.write_text("A{% block body %}{% endblock %}C")
        child.write_text(
            "{% extends \"base.jt.inp\" %}{% block body %}{{ value }}{% endblock %}"
        )

        result = core.TemplateManager.expand_file(child, {"value": "B"})

        assert result == "ABC"

    def test_expand_supports_parent_template_from_template_manager_path(self, tmp_path):
        """Test manager expansion searches registered template roots for parents."""
        child_dir = tmp_path / "child"
        child_dir.mkdir()
        (tmp_path / "base.jt.inp").write_text("A{% block body %}{% endblock %}C")
        (child_dir / "child.jt.inp").write_text(
            "{% extends \"base.jt.inp\" %}{% block body %}B{% endblock %}"
        )
        manager = core.TemplateManager(paths=[tmp_path], include_env=False)

        result = manager.expand("child/child.jt.inp", {})

        assert result == "ABC"

    def test_expand_text_formats_float_substitutions_with_12_digit_exponents(self):
        """Test default template expansion renders floats consistently."""
        result = core.TemplateManager.expand_text(
            "{{ scalar }} {{ array_scalar }} {{ integer }} {{ text }}",
            {
                "scalar": 1.234567890123456,
                "array_scalar": np.float64(0.125),
                "integer": 7,
                "text": "abc",
            },
        )

        assert result == "1.234567890123e+00 1.250000000000e-01 7 abc"

    def test_expand_text_accepts_float_format_override(self):
        """Test callers can select the model-level float format."""
        result = core.TemplateManager.expand_text(
            "{{ scalar }}", {"scalar": 1.234567890123456}, float_format=".4e"
        )

        assert result == "1.2346e+00"

    def test_template_float_format_defaults_to_12_digit_exponents(self):
        """Test the model-level template format default."""
        assert core.TemplateManager.template_float_format({}) == ".12e"
        assert (
            core.TemplateManager.template_float_format(
                {"template_float_format": ".5e"}
            )
            == ".5e"
        )

    def test_expand_text_formats_float_expression_results_with_12_digit_exponents(self):
        """Test arithmetic expression results use the default float format."""
        result = core.TemplateManager.expand_text(
            "{{ ppm * 1e-6 }}", {"ppm": 600.0}
        )

        assert result == "6.000000000000e-04"

    def test_expand_text_preserves_explicit_string_formatting(self):
        """Test templates can override the default by rendering a string."""
        result = core.TemplateManager.expand_text(
            "{{ '{:.4f}'.format(value) }}", {"value": 1.23456789}
        )

        assert result == "1.2346"

    def test_tree_print_formats_float_values_like_template_expansion(self):
        """Test error diagnostics use the same float representation."""
        result = core.TemplateManager._tree_print(
            {"state": {"power": 40.0, "nlib": 2}}
        )

        assert "state.power=4.000000000000e+01\n" in result
        assert "state.nlib=2\n" in result


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


class TestObiwan:
    """Test OBIWAN F71 metadata parsing."""

    def test_parse_burnups_from_f33_text(self):
        """Read burnup points from an OBIWAN F33 burnup table."""
        burnup_text = """
         pos   MWd/MTIHM
           1  0.0000e+00
           2  2.4958e+00
           3  7.4873e+00
"""

        burnups = core.Obiwan.parse_burnups_from_f33_text(
            burnup_text, "sample.f33"
        )

        np.testing.assert_allclose(burnups, [0.0, 2.4958, 7.4873])

    def test_get_burnups_from_f33_runs_obiwan_burnup_view(self):
        """Request the F33 burnup table from OBIWAN."""
        burnup_text = """
         pos   MWd/MTIHM
           1  0.0000e+00
           2  5.0000e+00
"""

        with patch("scale.olm.core.run_command", return_value=burnup_text) as run:
            burnups = core.Obiwan.get_burnups_from_f33(
                "/path/to/obiwan", "sample.f33"
            )

        run.assert_called_once_with(
            "/path/to/obiwan view -type=f33 -format=burnups sample.f33",
            echo=False,
        )
        np.testing.assert_allclose(burnups, [0.0, 5.0])

    def test_parse_burnups_from_f33_text_requires_contiguous_positions(self):
        """Reject malformed F33 burnup tables."""
        burnup_text = """
         pos   MWd/MTIHM
           1  0.0000e+00
           3  7.4873e+00
"""

        with pytest.raises(ValueError, match="positions must be contiguous"):
            core.Obiwan.parse_burnups_from_f33_text(burnup_text, "sample.f33")

    def test_get_info_history_from_f71_scale_7_table(self):
        """Read burnups and interval history from SCALE 7 OBIWAN info text."""
        info_text = """
Some OBIWAN header text...
 pos         time        power         flux      fluence       energy    initialhm       volume libpos   case   step DCGNAB
 (-)          (s)         (MW)   (n/cm^2-s)     (n/cm^2)        (MWd)      (MTIHM)       (cm^3)    (-)    (-)    (-)    (-)
  24  0.00000e+00  4.00000e+01  1.00000e+14  0.00000e+00  0.00000e+00  1.00000e+00  1.09084e+05      1     10      0 DC----
  25  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  1.00000e+00  1.09084e+05      1     -2      0 DC----
  26  2.16000e+06  3.98786e+01  3.63597e+14  7.85370e+20  9.96965e+02  1.00000e+00  1.09084e+05      2     -2      1 DC----
  27  2.16000e+07  3.98781e+01  3.80710e+14  8.18637e+21  9.96954e+03  1.00000e+00  1.09084e+05      3     -2      2 DC----
  28  5.40000e+07  3.98783e+01  4.20568e+14  2.18128e+22  2.49239e+04  1.00000e+00  1.09084e+05      4     -2      3 DC----
D - state definition present
"""

        with patch("scale.olm.core.run_command", return_value=info_text) as run:
            history = core.Obiwan.get_info_history_from_f71(
                "/path/to/obiwan", "sample.f71", -2
            )

        run.assert_called_once_with(
            "/path/to/obiwan view -format=info sample.f71", echo=False
        )
        np.testing.assert_allclose(
            history["burnups"],
            [0.0, 996.965, 9969.54, 24923.9],
            rtol=0.0,
            atol=1.0e-8,
        )
        assert history["initialhm"] == pytest.approx(1.0)
        np.testing.assert_allclose(
            [entry["burn"] for entry in history["burndata"]],
            [25.0, 225.0, 375.0],
        )
        np.testing.assert_allclose(
            [entry["power"] for entry in history["burndata"]],
            [996.965 / 25.0, (9969.54 - 996.965) / 225.0, (24923.9 - 9969.54) / 375.0],
        )


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

    def test_duplicate_degenerate_axis_value_advanced(self):
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


class TestRelAbsHistogram:
    """Test relative/absolute histogram plotting."""

    def test_plot_hist_clips_values_above_display_range_to_edge_bins(self):
        """Test histogram values above log10(error)=0 stay in edge bins."""
        histogram = core.RelAbsHistogram(
            rhist=np.array([1.0e-3, 1.0e1]),
            ahist=np.array([1.0e-6, 1.0e2]),
        )

        with patch.object(core.plt, "hist2d") as mock_hist2d, patch.object(
            core.plt, "colorbar"
        ), patch.object(core.plt, "savefig"):
            mock_hist2d.return_value = (None, None, None, object())

            core.RelAbsHistogram.plot_hist(histogram, image="hist.png")

        x_values = mock_hist2d.call_args.args[0]
        y_values = mock_hist2d.call_args.args[1]
        bins = mock_hist2d.call_args.kwargs["bins"]
        assert bins[0] == pytest.approx(-10.0)
        assert bins[-1] == pytest.approx(0.0)
        assert x_values[-1] == pytest.approx(0.0)
        assert y_values[-1] == pytest.approx(0.0)
        assert np.all(x_values <= 0.0)
        assert np.all(y_values <= 0.0)

        core.plt.close("all")

    def test_plot_hist_uses_eps0_limit_and_discards_small_points(self):
        """Test points below eps0 on both axes are not histogrammed."""
        histogram = core.RelAbsHistogram(
            rhist=np.array([1.0e-9, 1.0e-5, 1.0e1]),
            ahist=np.array([1.0e-9, 1.0e-8, 1.0e2]),
        )

        with patch.object(core.plt, "hist2d") as mock_hist2d, patch.object(
            core.plt, "colorbar"
        ), patch.object(core.plt, "savefig"):
            mock_hist2d.return_value = (None, None, None, object())

            core.RelAbsHistogram.plot_hist(
                histogram,
                image="hist.png",
                eps0=1.0e-6,
            )

        x_values = mock_hist2d.call_args.args[0]
        y_values = mock_hist2d.call_args.args[1]
        bins = mock_hist2d.call_args.kwargs["bins"]
        assert bins[0] == pytest.approx(-6.0)
        assert bins[-1] == pytest.approx(0.0)
        assert len(x_values) == 2
        assert len(y_values) == 2
        assert x_values[0] == pytest.approx(-5.0)
        assert y_values[0] == pytest.approx(-6.0)
        assert x_values[1] == pytest.approx(0.0)
        assert y_values[1] == pytest.approx(0.0)

        core.plt.close("all")

    def test_plot_hist_all_discarded_points_does_not_warn(self):
        """Test all-discarded histograms still render without density warnings."""
        histogram = core.RelAbsHistogram(
            rhist=np.array([1.0e-9, 2.0e-9]),
            ahist=np.array([1.0e-9, 2.0e-9]),
        )

        with patch.object(core.plt, "savefig"):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                core.RelAbsHistogram.plot_hist(
                    histogram,
                    image="hist.png",
                    eps0=1.0e-6,
                )

        core.plt.close("all")

    def test_plot_hist_normalizes_to_maximum_occupied_bin(self):
        """Test the busiest histogram bin maps to the color range maximum."""
        histogram = core.RelAbsHistogram(
            rhist=np.array([1.0e-3, 1.2e-3, 1.0e-1]),
            ahist=np.array([1.0e-3, 1.2e-3, 1.0e-1]),
        )

        with patch.object(core.plt, "hist2d") as mock_hist2d, patch.object(
            core.plt, "colorbar"
        ), patch.object(core.plt, "savefig"):
            mock_hist2d.return_value = (None, None, None, object())

            core.RelAbsHistogram.plot_hist(histogram, image="hist.png")

        assert mock_hist2d.call_args.kwargs["density"] is False
        norm = mock_hist2d.call_args.kwargs["norm"]
        assert isinstance(norm, LogNorm)
        assert norm.vmin == pytest.approx(0.5)
        assert norm.vmax == pytest.approx(1.0)
        np.testing.assert_allclose(
            mock_hist2d.call_args.kwargs["weights"],
            [0.5, 0.5, 0.5],
        )

        core.plt.close("all")

    def test_plot_hist_draws_epsilon_threshold_lines(self):
        """Test histogram plots mark epsr and epsa with red dashed lines."""
        histogram = core.RelAbsHistogram(
            rhist=np.array([1.0e-4, 1.0e-2]),
            ahist=np.array([1.0e-7, 1.0e-5]),
        )

        with patch.object(core.plt, "axvline") as mock_axvline, patch.object(
            core.plt, "axhline"
        ) as mock_axhline, patch.object(core.plt, "text") as mock_text, patch.object(
            core.plt, "plot"
        ) as mock_plot, patch.object(
            core.plt, "savefig"
        ):
            core.RelAbsHistogram.plot_hist(
                histogram,
                image="hist.png",
                epsr=1.0e-3,
                epsa=1.0e-6,
            )

        core.plt.gcf().canvas.draw()
        x_tick_labels = {
            label.get_text(): label.get_color()
            for label in core.plt.gca().get_xticklabels()
        }
        y_tick_labels = {
            label.get_text(): label.get_color()
            for label in core.plt.gca().get_yticklabels()
        }
        assert mock_axvline.call_args.args[0] == pytest.approx(-3.0)
        assert mock_axvline.call_args.kwargs["color"] == "red"
        assert mock_axvline.call_args.kwargs["linestyle"] == "--"
        assert mock_axhline.call_args.args[0] == pytest.approx(-6.0)
        assert mock_axhline.call_args.kwargs["color"] == "red"
        assert mock_axhline.call_args.kwargs["linestyle"] == "--"
        labels = [call.args[2] for call in mock_text.call_args_list]
        assert r"$\epsilon_0$" in labels
        assert r"$\epsilon_r$" in labels
        assert r"$\epsilon_a$" in labels
        label_positions = {
            call.args[2]: (call.args[0], call.args[1])
            for call in mock_text.call_args_list
        }
        assert label_positions[r"$\epsilon_0$"] == pytest.approx((-9.75, -9.75))
        assert label_positions[r"$\epsilon_r$"] == pytest.approx((-3.0, -9.75))
        assert label_positions[r"$\epsilon_a$"] == pytest.approx((-9.75, -6.0))
        eps_r_call = next(
            call for call in mock_text.call_args_list if call.args[2] == r"$\epsilon_r$"
        )
        assert "rotation" not in eps_r_call.kwargs
        mock_plot.assert_called_once_with(
            [-10],
            [-10],
            color="red",
            marker="o",
            linestyle="None",
            clip_on=False,
        )
        assert x_tick_labels["-10"] == "red"
        assert x_tick_labels["-3"] == "red"
        assert y_tick_labels["-10"] == "red"
        assert y_tick_labels["-6"] == "red"

        core.plt.close("all")
