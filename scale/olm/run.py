from pathlib import Path
import scale.olm.internal as internal
import scale.olm.core as core
import glob
import json
from typing import Literal

__all__ = ["makefile"]

_TYPE_MAKEFILE = "scale.olm.run:makefile"


def _schema_makefile(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_MAKEFILE, with_state=with_state)
    return _schema


def _test_args_makefile(with_state: bool = False):
    return {"_type": _TYPE_MAKEFILE, "dry_run": False}


def makefile(
    dry_run: bool = False,
    _model: dict = {},
    _env: dict = {},
    _type: Literal[_TYPE_MAKEFILE] = None,
):
    """Generate a Makefile and run it.

    Args:
        _dry_run: Whether or not to run calcs.

        _env: Dictionary with environment like work_dir and scalerte.

        _model: Model data


    """
    if not "work_dir" in _env:
        td = core.TempDir()
        work_path = td.path / "_work"
    else:
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
