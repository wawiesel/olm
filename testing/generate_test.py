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
