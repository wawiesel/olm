from pathlib import Path
import scale.olm.common as common


def makefile(model, nprocs):
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
    common.run_command(command_line)

    return {
        "scalerte": scalerte,
        "work_dir": work_dir,
        "file": str(file.relative_to(work_dir)),
        "command_line": command_line,
    }
