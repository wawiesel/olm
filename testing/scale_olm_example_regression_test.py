"""Local-only regression tests for example problem consistency."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import scale.olm.internal as internal


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "testing" / "baselines" / "example_consistency.json"


def _runtime_available():
    scale_dir = os.environ.get("SCALE_DIR")
    if scale_dir:
        scale_path = Path(scale_dir)
        if (scale_path / "bin" / "scalerte").exists() and (
            scale_path / "bin" / "obiwan"
        ).exists():
            return True
    return Path(os.environ.get("OLM_SCALERTE", "")).exists() and Path(
        os.environ.get("OLM_OBIWAN", "")
    ).exists()


def _load_baseline():
    return json.loads(BASELINE_PATH.read_text())


def _check_summary(check_output):
    time_quality = check_output["time_quality"]
    return {
        "q1": check_output["q1"],
        "q2": check_output["q2"],
        "min_time_q1": min(row["q1"] for row in time_quality),
        "min_time_q2": min(row["q2"] for row in time_quality),
        "worst_time_days": check_output["worst_time_quality"]["time_days"],
        "worst_burnup_gwd_per_mtu": check_output["worst_time_quality"].get(
            "burnup_gwd_per_mtu"
        ),
        "test_pass": check_output["test_pass"],
        "test_pass_time": check_output["test_pass_time"],
    }


def _assert_not_regressed(actual, expected):
    floors = expected["floors"]
    assert actual["q1"] >= floors["q1"]
    assert actual["q2"] >= floors["q2"]
    assert actual["min_time_q1"] >= floors["min_time_q1"]
    assert actual["min_time_q2"] >= floors["min_time_q2"]


@pytest.mark.integration
def test_example_consistency_local_scale_regression(tmp_path):
    """Run configured examples with local SCALE and compare consistency floors."""
    if os.environ.get("SCALE_OLM_RUN_EXAMPLE_REGRESSION") != "1":
        pytest.skip("set SCALE_OLM_RUN_EXAMPLE_REGRESSION=1 to run local SCALE examples")
    if not _runtime_available():
        pytest.skip("SCALE runtime not configured with SCALE_DIR or OLM_* executables")

    baseline = _load_baseline()
    actual_examples = []
    for expected in baseline["examples"]:
        work_dir = tmp_path / expected["name"] / "_work"
        env = {
            "OLM_WORK_DIR": str(work_dir),
            "SCALE_OLM_DO_RUN": "True",
        }
        with patch.dict(os.environ, env, clear=False):
            internal.create(
                config_file=str(PROJECT_ROOT / expected["config"]),
                generate=True,
                run=True,
                assemble=True,
                check=True,
                report=False,
                nprocs=int(os.environ.get("SCALE_OLM_EXAMPLE_NPROCS", "3")),
            )

        check_data = json.loads((work_dir / "check.olm.json").read_text())
        check_output = check_data["sequence"][expected["check_index"]]
        actual = {
            "name": expected["name"],
            "config": expected["config"],
            "check_index": expected["check_index"],
            **_check_summary(check_output),
        }
        actual_examples.append(actual)
        _assert_not_regressed(actual, expected)

    if os.environ.get("SCALE_OLM_UPDATE_EXAMPLE_BASELINE") == "1":
        updated = {
            "description": baseline["description"],
            "examples": [
                {
                    **actual,
                    "floors": {
                        "q1": actual["q1"],
                        "q2": actual["q2"],
                        "min_time_q1": actual["min_time_q1"],
                        "min_time_q2": actual["min_time_q2"],
                    },
                }
                for actual in actual_examples
            ],
        }
        BASELINE_PATH.write_text(json.dumps(updated, indent=4) + "\n")
