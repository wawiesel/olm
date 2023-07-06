import click
import sys
import scale.olm.common as common
import json
import structlog
import os
from pathlib import Path


# ---------------------------------------------------------------------------------------
# OLM
# ---------------------------------------------------------------------------------------
#
# This is the entry point to run OLM as a command line interface.
#
@click.group()
def cli():
    pass


# ---------------------------------------------------------------------------------------
# OLM DO
# ---------------------------------------------------------------------------------------
#
# This is the entry point to run OLM DO command.
#
@click.command(name="do")
@click.argument("config_file", metavar="config-olm.json", type=click.Path(exists=True))
@click.option(
    "--generate/--nogenerate",
    default=False,
    help="whether to perform input generation",
)
@click.option(
    "--run/--norun",
    default=False,
    help="whether to perform runs",
)
@click.option(
    "--build/--nobuild",
    default=False,
    help="whether to build the reactor library",
)
@click.option(
    "--check/--nocheck",
    default=False,
    help="whether to check the generated library",
)
@click.option(
    "--report/--noreport",
    default=False,
    help="whether to create the report documentation",
)
def command_do(config_file, generate, run, build, check, report):
    # Cycle through these modes.
    modes = []
    if generate:
        modes.append("generate")
    if run:
        modes.append("run")
    if build:
        modes.append("build")
    if check:
        modes.append("check")
    if report:
        modes.append("report")

    # Load the input data.
    with open(config_file, "r") as f:
        data = json.load(f)
    data["model"]["config_file"] = config_file

    # Update paths in the model block.
    model = common.update_model(data["model"])

    # Run each enabled mode in sequence.
    for mode in modes:
        output = common.fn_redirect({"model": model, **data[mode]})
        output_file = str(Path(model["work_dir"]) / mode) + ".json"
        common.logger.info(f"Writing {output_file} ...")
        with open(output_file, "w") as f:
            f.write(json.dumps(output, indent=4))


cli.add_command(command_do)


# ---------------------------------------------------------------------------------------
# OLM LINK
# ---------------------------------------------------------------------------------------
#
# This is the entry point to run OLM LINK command.
#
import scale.olm.link as link


@click.command(name="link")
@click.argument("name", metavar="LIB-NAME")
@click.option(
    "--path",
    "-p",
    "paths",
    multiple=True,
    type=click.Path(exists=True),
    help="path to prepend to SCALE_OLM_PATH",
)
@click.option(
    "--env/--noenv",
    default=True,
    help="whether to allow using the environment variable SCALE_OLM_PATH or not",
)
@click.option(
    "--dest",
    "-d",
    default=os.getcwd(),
    type=click.Path(exists=True),
    help="destination directory (default: current)",
)
@click.option(
    "--format",
    "-f",
    type=str,
    default="archive",
    help="destination format for the libraries (arpdata.txt, archive)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="just emit commands without running them",
)
def command_link(name, paths, env, dest, format, dry_run):
    try:
        registry = common.create_registry(paths, env)

        if not name in registry:
            raise ValueError("name={} not found in provided paths!".format(name))

        libinfo = registry[name]
        link.make_available(dry_run, libinfo, dest, format)

        return 0

    except ValueError as ve:
        common.logger.error(str(ve))
        return str(ve)


cli.add_command(command_link)


# ---------------------------------------------------------------------------------------
# OLM CHECK
# ---------------------------------------------------------------------------------------
#
# This is the entry point to run OLM CHECK command.
#
import scale.olm.check as check


def methods_help(*methods):
    desc = "Run checking method NAME with options OPTS in JSON string format. The following checks are supported.\n\n"
    for m in methods:
        desc += m.__name__ + " '" + json.dumps(m.default_params()) + "' where "
        params = m.describe_params()
        for p in params:
            desc += p + " is " + params[p] + ", "
        desc = desc[:-2]  # remove last comma
        desc += ".\n\n"  # add period and newlines
    return desc


@click.command(name="check")
@click.argument("archive_file", metavar="archive.h5", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    "output_file",
    default="check.json",
    type=str,
    metavar="FILE",
    help="File to write results.",
)
@click.option(
    "--sequence",
    "-s",
    "text_sequence",
    type=str,
    metavar="'{\".type\": NAME, <OPTIONS>}'",
    multiple=True,
    help=methods_help(check.GridGradient, check.Continuity),
)
def command_check(archive_file, output_file, text_sequence):
    sequence = []
    for s in text_sequence:
        sequence.append(json.loads(s))

    model = {"archive_file": archive_file}
    output = check.sequencer(model, sequence)

    common.logger.info(f"Writing {output_file} ...")
    with open(output_file, "w") as f:
        f.write(json.dumps(output, indent=4))


cli.add_command(command_check)
