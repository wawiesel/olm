import scale.olm as olm
import pytest


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


def test_mox_ornltm2003_2_uses_custom_uo2_vector():
    comp = olm.generate.comp.mox_ornltm2003_2(
        state={"pu239_frac": 70.0, "pu_frac": 5.0},
        uo2={"iso": {"u235": 4.0, "u238": 96.0}},
    )

    assert comp["_input"]["uo2"]["iso"]["u235"] == pytest.approx(4.0)
    assert comp["_input"]["uo2"]["iso"]["u238"] == pytest.approx(96.0)
    assert comp["hmo2"]["iso"]["u235"] == pytest.approx(3.8)
    assert comp["hmo2"]["iso"]["u238"] == pytest.approx(91.2)
