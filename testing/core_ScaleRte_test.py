import scale.olm.core as core
import pytest
from unittest.mock import MagicMock
from pathlib import Path
import os


def test_if_path_exists(mocker):
    """Tests retrieving the version of the SCALE Runtime Environment and other
    basic initialization infor."""

    # Mock the subprocess.run method to return a known version.
    mocker.patch("subprocess.run", return_value=MagicMock(stdout="SCALE version 1.0.0"))
    mocker.patch("pathlib.Path.exists", return_value=True)

    # Unset this so we do not pick it up in testing the data directory.
    os.unsetenv("DATA")

    # Assert that the initialization is correct.
    scalerte_path = Path("/path/to/bin/scalerte")
    runner = core.ScaleRte(scalerte_path)
    assert runner.version == "1.0.0"
    assert runner.data_dir == Path("/path/to/data")
    assert runner.scalerte_path == scalerte_path


def test_data_env_var(mocker):
    """Tests processing the DATA environment variable."""

    # Unset this so we do not pick it up in testing the data directory.
    test_data = "/this/very/specific/data"
    os.environ["DATA"] = test_data

    # Assert that the initialization is correct.
    scalerte_path = Path("/path/to/bin/scalerte")
    data_dir = core.ScaleRte._get_data_dir(scalerte_path)
    assert str(data_dir) == str(test_data)


def test_version_if_path_does_not_exist(mocker):
    """Tests throwing when scalerte_path does not exist."""

    # Patch the existence to always be false.
    mocker.patch("pathlib.Path.exists", return_value=False)
    with pytest.raises(ValueError):
        core.ScaleRte("/path/to/scale/bin/scalerte")
