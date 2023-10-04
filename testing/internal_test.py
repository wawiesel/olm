import scale.olm as olm
import pytest


def test_get_function_handle():
    """Tests getting a function handle based on _type strings which will be in the OLM JSON data files."""

    _type = "scale.olm.generate.comp:uo2_simple"
    data = {"state": {"enrichment": 5.0}}
    comp = olm.internal._get_function_handle(_type)(**data)
    x = comp["uo2"]["iso"]

    assert x["u234"] == pytest.approx(0.0)
    assert x["u235"] == pytest.approx(5.0)
    assert x["u236"] == pytest.approx(0.0)
    assert x["u238"] == pytest.approx(95.0)
