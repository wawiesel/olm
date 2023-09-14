import os

os.environ["SCALE_LOG_LEVEL"] = "30"
import glob
from pathlib import Path
import scale.olm.internal as internal


def test_collection_generates_inputs(subset=[]):
    test_dir = str(Path(__file__).parent)
    root = str(Path(__file__).parent.parent)
    # print(root)
    for x in glob.glob(root + "/collection/*/config.olm.json"):
        reactor = Path(x).parent.stem
        if len(subset) == 0 or (reactor in subset):
            print(f"testing reactor={reactor}")
            work_dir = str(Path(test_dir) / "_work" / reactor)
            os.environ["OLM_WORK_DIR"] = work_dir
            try:
                internal.create(
                    config_file=x,
                    generate=True,
                    run=False,
                    assemble=False,
                    check=False,
                    report=False,
                    nprocs=1,
                )
            except:
                assert False, f"Error generating {reactor} from {x} at {work_dir}"


if __name__ == "__main__":
    import sys

    subset = []
    if len(sys.argv) > 1:
        subset = sys.argv[1:]
    test_collection_generates_inputs(subset)
