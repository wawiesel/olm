import scale.olm as olm
import pytest
from pydantic import ValidationError


def test_constpower_burndata():
    # Test that the burnup sequence is correct.
    power = 40.0
    gwd_burnups = [0.0, 10.0, 20.0]
    time = olm.generate.time.constpower_burndata({"specific_power": power}, gwd_burnups)

    # Generate output midpoint burnups.
    bu = 0.0
    mid_burnups = []
    for bd in time["burndata"]:
        dbu = bd["burn"] * bd["power"] / 1000.0
        mid_burnups.append(bu + dbu / 2.0)
        bu += dbu

    # Double loop to look for approximate matches.
    for t in gwd_burnups:
        found = False
        for b in mid_burnups:
            if t == pytest.approx(b, 1.0):
                found = True
        assert found, "No approximate match found for {} in {}".format(
            t, ",".join(str(e) for e in mid_burnups)
        )


def test_constpower_burndata_sorts_burnups_and_handles_single_point():
    result = olm.generate.time.constpower_burndata(
        {"specific_power": 40.0}, [20.0, 0.0, 10.0]
    )

    assert result == {
        "burndata": [
            {"power": 40.0, "burn": 250.0},
            {"power": 40.0, "burn": 250.0},
            {"power": 40.0, "burn": 250.0},
        ]
    }

    assert olm.generate.time.constpower_burndata(
        {"specific_power": 40.0}, [0.0]
    ) == {"burndata": [{"power": 40.0, "burn": 0}]}


def test_constpower_burndata_requires_zero_initial_burnup():
    with pytest.raises(ValueError, match="Burnup step 0.0"):
        olm.generate.time.constpower_burndata({"specific_power": 40.0}, [1.0, 2.0])


def test_constpower_burndata_rejects_nonpositive_power():
    with pytest.raises(ValidationError):
        olm.generate.time.constpower_burndata({"specific_power": 0.0}, [0.0])


def test_static_pass_through_drops_state():
    assert olm.generate.static.pass_through(
        _type="scale.olm.generate.static:pass_through",
        state={"specific_power": 40.0},
        xslib="v7.1",
    ) == {"xslib": "v7.1"}


def test_uo2_formula_enrichment_limits():
    with pytest.raises(ValueError, match="must be <=10%"):
        olm.generate.comp.uo2_vera({"enrichment": 10.1})

    with pytest.raises(ValueError, match="must be <=20%"):
        olm.generate.comp.uo2_nuregcr5625({"enrichment": 20.1})


def test_mox_ornltm2003_2_uses_custom_uo2_vector():
    comp = olm.generate.comp.mox_ornltm2003_2(
        state={"pu239_frac": 70.0, "pu_frac": 5.0},
        uo2={"iso": {"u235": 4.0, "u238": 96.0}},
    )

    assert comp["_input"]["uo2"]["iso"]["u235"] == pytest.approx(4.0)
    assert comp["_input"]["uo2"]["iso"]["u238"] == pytest.approx(96.0)
    assert comp["hmo2"]["iso"]["u235"] == pytest.approx(3.8)
    assert comp["hmo2"]["iso"]["u238"] == pytest.approx(91.2)


def test_mox_composition_validation_rejects_unknown_zone_set():
    with pytest.raises(ValueError, match="must be BWR2016/PWR2016"):
        olm.generate.comp.mox_multizone_2023(
            state={"pu239_frac": 70.0, "pu_frac": 5.0},
            zone_names="UNKNOWN",
            zone_pins=[1, 1, 1, 1],
        )


def test_mox_composition_validation_rejects_empty_uo2_iso_vector():
    with pytest.raises(ValidationError):
        olm.generate.comp.mox_ornltm2003_2(
            state={"pu239_frac": 70.0, "pu_frac": 5.0},
            uo2={"iso": {}},
        )
