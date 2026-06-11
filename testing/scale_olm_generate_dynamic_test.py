import pytest

from scale.olm.generate import dynamic


def test_scipy_interp_accepts_case_insensitive_method():
    result = dynamic.scipy_interp(
        state_var="coolant_density",
        data_pairs=[(0.3, 0.4), (0.7, 0.5), (1.1, 0.6)],
        state={"coolant_density": 0.7},
        method="PCHIP",
    )

    assert result == pytest.approx(0.5)


def test_scipy_interp_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unsupported scipy_interp method=nearest"):
        dynamic.scipy_interp(
            state_var="coolant_density",
            data_pairs=[(0.3, 0.4), (0.7, 0.5), (1.1, 0.6)],
            state={"coolant_density": 0.7},
            method="nearest",
        )
