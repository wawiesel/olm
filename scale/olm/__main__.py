import click
import sys
import scale.olm.internal as internal
import scale.olm.internal as internal
import scale.olm.core as core
import json
import structlog
import os
from pathlib import Path
import re


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
# OLM CREATE
# ---------------------------------------------------------------------------------------
#
# This is the entry point to run OLM CREATE command.
#
@click.command(name="create")
@click.argument("config_file", metavar="config.olm.json", type=click.Path(exists=True))
@click.option(
    "--generate/--nogenerate",
    default=None,
    is_flag=True,
    help="whether to perform input generation",
)
@click.option(
    "--run/--norun",
    default=None,
    is_flag=True,
    help="whether to perform runs",
)
@click.option(
    "--assemble/--noassemble",
    default=None,
    is_flag=True,
    help="whether to assemble the ORIGEN library",
)
@click.option(
    "--check/--nocheck",
    default=None,
    is_flag=True,
    help="whether to check the generated library",
)
@click.option(
    "--report/--noreport",
    default=None,
    is_flag=True,
    help="whether to create the report documentation",
)
@click.option(
    "--nprocs",
    type=int,
    default=3,
    help="how many processes to use",
)
@internal.copy_doc(internal.create)
def command_create(**kwargs):
    try:
        internal.create(**kwargs)
    except ValueError as ve:
        internal.logger.error(str(ve))
        return str(ve)


cli.add_command(command_create)


# ---------------------------------------------------------------------------------------
# OLM LINK
# ---------------------------------------------------------------------------------------
#
# This is the entry point to run OLM LINK command.
#
import scale.olm.link as link


@click.command(name="link")
@click.argument("names", type=str, nargs=-1, metavar="LIB1 LIB2 ...")
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
    "--show",
    default=False,
    is_flag=True,
    help="show all the known libraries",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="just emit commands without running them",
)
@internal.copy_doc(link.link)
def command_link(**kwargs):
    try:
        link.link(**kwargs)
    except ValueError as ve:
        internal.logger.error(str(ve))
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
@click.argument("archive_file", type=str)
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
    metavar="'{\"_type\": NAME, <OPTIONS>}'",
    multiple=True,
    help=methods_help(check.GridGradient),
)
@click.option(
    "--nprocs",
    type=int,
    default=3,
    help="how many processes to use",
)
def command_check(archive_file, output_file, text_sequence, nprocs):
    sequence = []
    for s in text_sequence:
        sequence.append(json.loads(s))

    # Back out what the sequencer expects.
    x = archive_file.rsplit(":")
    if len(x) > 1 and x[0].endswith("arpdata.txt"):
        name = x[1]
        work_dir = Path(x[0]).parent
    else:
        if not archive_file.endswith(".arc.h5"):
            internal.logger.error(
                "Libraries in HDF5 archive format must end in .arc.h5 but found",
                archive_file=archive_file,
            )
            return
        name = re.sub("\.arc\.h5$", "", archive_file)
        work_dir = Path(archive_file).parent

    output = check.sequencer(
        sequence, _model={"name": name}, _env={"work_dir": work_dir, "nprocs": nprocs}
    )

    internal.logger.info(f"Writing {output_file} ...")
    with open(output_file, "w") as f:
        f.write(json.dumps(output, indent=4))


cli.add_command(command_check)


@click.command(name="init")
@click.argument("config_dir", type=str, required=False)
@click.option(
    "--variant", "-v", "variant", type=str, default=None, help="Name of model variant."
)
@click.option(
    "--list",
    "-l",
    "list_variants",
    is_flag=True,
    default=False,
    help="List all known variants and exit.",
)
@internal.copy_doc(internal.init)
def command_init(**kwargs):
    try:
        internal.init(**kwargs)
    except ValueError as ve:
        logger.error(str(ve))
        return str(ve)


cli.add_command(command_init)
