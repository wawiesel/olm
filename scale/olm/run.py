from pathlib import Path
import scale.olm.common as common


def __makefile_input_desc(status):
    rows = list()
    for s in status:
        rows.append([s, status[s]])
    return common.rst_table("Makefile-based SCALE run info", [25, 75], 0, rows)


def makefile(model, dry_run, nprocs):
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

    command_line = f"cd {work_dir} && make -j {nprocs}"
    if dry_run:
        common.logger.warning("No SCALE runs will be performed because dry_run=True!")
    else:
        common.run_command(command_line)

    status = {
        "scalerte": scalerte,
        "work_dir": work_dir,
        "file": str(file.relative_to(work_dir)),
        "dry_run": dry_run,
        "command_line": command_line,
    }

    status["input_desc"] = __makefile_input_desc(status)
    return status
