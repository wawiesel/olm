from pathlib import Path
import scale.olm.internal as internal
import scale.olm.core as core
import glob
import json


def makefile(dry_run, _model, _env):
    """Generate a Makefile and run it.

    Args:
        _env: Dictionary with environment like work_dir and scalerte.

    """

    work_path = Path(_env["work_dir"])
    base_dir = "perms"
    base_path = work_path / base_dir
    input_list = []
    with open(work_path / "generate.olm.json", "r") as f:
        g = json.load(f)
        for perms in g["perms"]:
            input = Path(perms["input_file"]).relative_to(base_dir)
            input_list.append(str(input))

    return internal._execute_makefile(
        dry_run, _env, base_path=base_path, input_list=input_list
    )
