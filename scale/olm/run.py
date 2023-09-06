from pathlib import Path
import scale.olm.common as common
import scale.olm.core as core
import glob


def __runtime_in_hours(runtime):
    """Convert runtime in seconds to well-formatted runtime in hours."""
    return "{:.2g}".format(runtime / 3600.0)


def makefile(model, dry_run, nprocs):
    """Generate a Makefile and run it.

    Args:
        model: dictionary with work_dir and scalerte.
        dry_run: pass :math:`-n` to make
        nprocs: pass :math:`-j nprocs` to make
    """
    scalerte = model["scalerte"]

    contents = f"""
outputs = $(patsubst %.inp, %.out, $(wildcard */*.inp))

.PHONY: all

all: $(outputs)

%.out: %.inp
\t@rm -f $@.FAILED
\t{scalerte} $<
\t@grep 'Error' $@ && mv -f $@ $@.FAILED && echo "^^^^^^^^^^^^^^^^ errors from $<" || true

clean:
\trm -f $(outputs)
"""

    work_dir = model["work_dir"]
    file = Path(work_dir) / "Makefile"
    with open(file, "w") as f:
        f.write(contents)

    version = common.get_scale_version(scalerte)
    core.logger.info(f"Running SCALE version {version}")

    command_line = f"cd {work_dir} && make -j {nprocs}"
    if dry_run:
        core.logger.warning("No SCALE runs will be performed because dry_run=True!")
    else:
        common.run_command(command_line)

    # Get file listing.
    perms = list()
    total_runtime = 0
    for input in [Path(x) for x in sorted(glob.glob(str(work_dir) + "/*/*.inp"))]:
        output = input.with_suffix(".out")
        success = output.exists()
        runtime = common.get_runtime(output) if success else 3.6e6
        total_runtime += runtime
        perms.append(
            {
                "input": str(input.relative_to(work_dir)),
                "output": str(output.relative_to(work_dir)),
                "success": success,
                "runtime_hrs": __runtime_in_hours(runtime),
            }
        )

    return {
        "scalerte": scalerte,
        "work_dir": work_dir,
        "dry_run": dry_run,
        "command_line": command_line,
        "version": version,
        "perms": perms,
        "total_runtime_hrs": __runtime_in_hours(total_runtime),
    }
