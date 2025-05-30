import scale.olm.core as core
import pytest


def test_single_nuclide_default_molar_mass():
    """Tests that the function correctly calculates the grams per mole for a single nuclide using the default molar mass data."""
    iso_wts = {"am241": 100.0}
    result = core.CompositionManager.grams_per_mol(iso_wts)
    assert result == pytest.approx(241.0568, abs=1e-6)


def test_multiple_nuclides_default_molar_mass():
    """Tests that the function correctly calculates the grams per mole for multiple nuclides using the default molar mass data."""
    iso_wts = {"am241": 50.0, "pu239": 50.0}
    result = core.CompositionManager.grams_per_mol(iso_wts, {})
    assert result == pytest.approx(1.0 / (0.5 / 241 + 0.5 / 239), abs=1e-6)


def test_empty_input_dictionary():
    """Tests that the function returns 0.0 when given an empty input dictionary."""
    result = core.CompositionManager.grams_per_mol({})
    assert result == pytest.approx(0.0, abs=1e-6)


def test_zero_mass():
    """If you would like to force using iso_wts that do not add up to 1.0, then
    include a dummy nuclide like 'xxx0'."""
    iso_wts = {}
    result = core.CompositionManager.grams_per_mol({"u235": 0.999, "xxx0": 0.001}, {})
    assert result == pytest.approx(235.23523523523525, abs=1e-6)
