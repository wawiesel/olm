import scale.olm.check as check
import scale.olm.core as core
import pytest


def data_file(filename):
    from pathlib import Path

    p = Path(__file__)
    p = p.parent.parent / "data" / filename
    size = p.stat().st_size
    # 50KB can't be real data
    if size < 5e4:
        raise ValueError(
            f"It appears the data file {p} may be a GIT LFS pointer. Make sure they are downloaded, e.g. with `git lfs pull``."
        )
    return p


def test_gridgradient_basic():
    # Test that we can change the basic parameters of the check.
    c = check.GridGradient(eps0=1e-3)
    assert c.eps0 == 1e-3
    assert c.epsr == c.default_params()["epsr"]
    assert c.epsa == c.default_params()["epsa"]

    # Test that we can load an archive
    a = core.ReactorLibrary(data_file("w17x17.arc.h5"))
    assert a != None

    # Test that we can change and get the default result.
    c.eps0 = c.default_params()["eps0"]
    i = c.run(a)
    assert i.name == "GridGradient"
    assert i.q1 == pytest.approx(0.630, 1e-3)
    assert i.q2 == pytest.approx(0.926, 1e-3)
