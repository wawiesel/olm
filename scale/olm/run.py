from pathlib import Path
import scale.olm.internal as internal
import scale.olm.core as core
import glob
import json


def _runtime_in_hours(runtime):
    """Convert runtime in seconds to well-formatted runtime in hours."""
    return "{:.2g}".format(runtime / 3600.0)


def makefile(dry_run, _model, _env, base_dir="perms"):
    """Generate a Makefile and run it.

    Args:
        _model: Dictionary with model info.
        _env: Dictionary with environment like work_dir and scalerte.

    """
    if not "scalerte" in _env:
        internal._raise_scalerte_error()

    scalerte = _env["scalerte"]

    work_path = Path(_env["work_dir"])
    input_listing = ""
    with open(work_path / "generate.olm.json", "r") as f:
        g = json.load(f)
        for perms in g["perms"]:
            input = Path(perms["input_file"]).relative_to("perms")
            input_listing += " " + str(input)

    contents = f"""
outputs = $(patsubst %.inp, %.out, {input_listing})

.PHONY: all

all: $(outputs)

%.out: %.inp
\t@rm -f $@.FAILED
\t{scalerte} $<
\t@grep 'Error' $@ && mv -f $@ $@.FAILED && echo "^^^^^^^^^^^^^^^^ errors from $<" || true

clean:
\trm -f $(outputs)
"""

    base_path = work_path / base_dir
    make_file = base_path / "Makefile"
    with open(make_file, "w") as f:
        f.write(contents)

    version = core.ScaleRunner(scalerte).version
    internal.logger.info("Running SCALE", version=version)

    nprocs = _env["nprocs"]
    command_line = f"cd {base_path} && make -j {nprocs}"
    if dry_run:
        internal.logger.warning("No SCALE runs will be performed because dry_run=True!")
    else:
        internal.run_command(command_line)

    # Get file listing.
    runs = list()
    total_runtime = 0
    for input in [Path(x) for x in sorted(glob.glob(str(base_path) + "/*/*.inp"))]:
        output = input.with_suffix(".out")
        success = output.exists()
        runtime = core.ScaleOutfile.get_runtime(output) if success else 3.6e6
        total_runtime += runtime
        runs.append(
            {
                "input_file": str(input.relative_to(work_path)),
                "output_file": str(output.relative_to(work_path)),
                "success": success,
                "runtime_hrs": _runtime_in_hours(runtime),
            }
        )

    return {
        "scalerte": str(scalerte),
        "base_dir": base_dir,
        "dry_run": dry_run,
        "nprocs": nprocs,
        "command_line": command_line,
        "version": version,
        "runs": runs,
        "total_runtime_hrs": _runtime_in_hours(total_runtime),
    }
