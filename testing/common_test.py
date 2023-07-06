import scale.olm.common as common
import pytest


def test_get_function_handle():
    # This demonstrates how we call fuel composition functions from strings in the
    # olm.json file.
    s = "scale.olm.generate:fuelcomp_uox_simple"
    d = {"state": {"enrichment": 5.0}}
    x = common.get_function_handle(s)(**d)
    assert x["u234"] == pytest.approx(0.0)
    assert x["u235"] == pytest.approx(5.0)
    assert x["u236"] == pytest.approx(0.0)
    assert x["u238"] == pytest.approx(95.0)
