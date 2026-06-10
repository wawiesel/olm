import scale.olm as olm
import pytest
from pydantic import ValidationError


def test_jt_expander_uses_packaged_template(tmp_path):
    result = olm.generate.root.jt_expander(
        template="model/triton/pin-uox.jt.inp",
        static={
            "_type": "scale.olm.generate.static:pass_through",
            "addnux": 2,
            "xslib": "v7-56",
            "pitch": 1.26,
            "fuelr": 0.4095,
            "cladr": 0.4750,
        },
        states={
            "_type": "scale.olm.generate.states:full_hypercube",
            "coolant_density": [0.70],
            "enrichment": [3.0],
            "specific_power": [40.0],
            "boron_ppm": [600.0],
        },
        comp={
            "fuel": {
                "_type": "scale.olm.generate.comp:uo2_nuregcr5625",
                "density": 10.4,
            }
        },
        time={
            "_type": "scale.olm.generate.time:constpower_burndata",
            "gwd_burnups": [0.0, 1.0],
        },
        _model={"name": "uox_quick"},
        _env={
            "config_file": str(tmp_path / "config.olm.json"),
            "work_dir": str(tmp_path / "_work"),
        },
    )

    input_file = tmp_path / "_work" / result["perms"][0]["input_file"]
    text = input_file.read_text()
    assert "pincell model" in text
    assert "read burndata" in text
    assert "den=1.040000000000e+01" in text


def test_jt_expander_uses_model_template_float_format(tmp_path):
    template = tmp_path / "model.jt.inp"
    template.write_text("value={{static.value}}")

    result = olm.generate.root.jt_expander(
        template=template.name,
        static={
            "_type": "scale.olm.generate.static:pass_through",
            "value": 1.23456789,
        },
        states={"_type": "scale.olm.generate.states:full_hypercube", "case": [1.0]},
        comp={"_type": "scale.olm.generate.static:pass_through"},
        time={"_type": "scale.olm.generate.static:pass_through"},
        _model={"name": "format_test", "template_float_format": ".4e"},
        _env={
            "config_file": str(tmp_path / "config.olm.json"),
            "work_dir": str(tmp_path / "_work"),
        },
    )

    input_file = tmp_path / "_work" / result["perms"][0]["input_file"]
    assert input_file.read_text() == "value=1.2346e+00"


def test_constpower_burndata():
    power = 40.0
    gwd_burnups = [0.0, 10.0, 20.0]
    time = olm.generate.time.constpower_burndata({"specific_power": power}, gwd_burnups)

    assert time == {
        "burndata": [
            {"power": 40.0, "burn": 250.0},
            {"power": 40.0, "burn": 250.0},
        ],
        "final_burnup_padding_gwd": 0.0,
    }


def test_constpower_burndata_sorts_burnups_and_handles_single_point():
    result = olm.generate.time.constpower_burndata(
        {"specific_power": 40.0}, [20.0, 0.0, 10.0]
    )

    assert result == {
        "burndata": [
            {"power": 40.0, "burn": 250.0},
            {"power": 40.0, "burn": 250.0},
        ],
        "final_burnup_padding_gwd": 0.0,
    }

    assert olm.generate.time.constpower_burndata(
        {"specific_power": 40.0}, [0.0]
    ) == {
        "burndata": [{"power": 40.0, "burn": 0}],
        "final_burnup_padding_gwd": 0.0,
    }


def test_constpower_burndata_adds_explicit_final_padding():
    result = olm.generate.time.constpower_burndata(
        {"specific_power": 40.0},
        [0.0, 10.0, 20.0],
        final_burnup_padding_gwd=1.0,
    )

    assert result == {
        "burndata": [
            {"power": 40.0, "burn": 250.0},
            {"power": 40.0, "burn": 250.0},
            {"power": 40.0, "burn": 25.0},
        ],
        "final_burnup_padding_gwd": 1.0,
    }


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
