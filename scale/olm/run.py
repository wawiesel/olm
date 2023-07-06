from pathlib import Path
import scale.olm.common as common
import subprocess


def makefile(model, nprocs):
    scalerte = model["scalerte"]

    contents = f"""
outputs = $(patsubst %.inp, %.out, $(wildcard */*.inp))

.PHONY: all

all: $(outputs)

%.out: %.inp
\t@rm -f $@.FAILED
\t{scalerte} $< || echo $< finished
\t@grep 'Error' $@ && mv -f $@ $@.FAILED && echo "^^^^^^^^^^^^^^^^ errors from $<" 

clean:
\trm -f $(outputs)
"""

    work_dir = model["work_dir"]
    file = Path(work_dir) / "Makefile"
    with open(file, "w") as f:
        f.write(contents)

    command_line = f"cd {work_dir} && make -j {nprocs}"
    common.logger.info(f"Running command_line='{command_line}' ...")

    p = subprocess.Popen(
        command_line,
        shell=True,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        encoding="utf-8",
    )
    while True:
        line = p.stdout.readline()
        if "Error" in line:
            common.logger.error(line.strip())
        else:
            common.logger.info(line.strip())
        if not line:
            break

    return {"scalerte": scalerte, "file": str(file), "command_line": command_line}
