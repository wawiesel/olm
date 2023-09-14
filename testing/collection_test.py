import os
import pytest

os.environ["SCALE_LOG_LEVEL"] = "30"
import glob
from pathlib import Path
import scale.olm.internal as internal


def all_reactors():
    root = str(Path(__file__).parent.parent)
    return [Path(x) for x in glob.glob(root + "/collection/*/config.olm.json")]


@pytest.mark.parametrize("reactor", all_reactors())
def test_collection_generates_inputs(reactor):
    print(f"testing reactor={reactor}")
    test_dir = Path(__file__).parent
    work_dir = str(Path(test_dir) / "_work")
    os.environ["OLM_WORK_DIR"] = work_dir
    try:
        internal.create(
            config_file=reactor,
            generate=True,
            run=False,
            assemble=False,
            check=False,
            report=False,
            nprocs=1,
        )
    except:
        assert False, f"Error generating {reactor} at {work_dir}"


if __name__ == "__main__":
    import sys

    test_collection_generates_inputs(sys.argv[1])
