from pathlib import Path
import scale.olm.common as common


def makefile(model, cmd, nprocs):
    contents = f"""
outputs = $(patsubst %.inp, %.out, $(wildcard */*.inp))

.PHONY: all

all: $(outputs)

%.out: %.inp
\t{cmd} $<
"""

    work_dir = model["work_dir"]
    file = Path(work_dir) / "Makefile"
    with open(file, "w") as f:
        f.write(contents)

    command_line = f"cd {work_dir} && make -j {nprocs}"
    common.logger.info(f"Running command_line='{command_line}' ...")

    # subprocess.Popen.capture_output(command_line,shell=True)

    return {"file": str(file)}
