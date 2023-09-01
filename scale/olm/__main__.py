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
    is_flag=True,
    help="whether to perform input generation",
)
@click.option(
    "--run/--norun",
    default=False,
    is_flag=True,
    help="whether to perform runs",
)
@click.option(
    "--build/--nobuild",
    default=False,
    is_flag=True,
    help="whether to build the reactor library",
)
@click.option(
    "--check/--nocheck",
    default=False,
    is_flag=True,
    help="whether to check the generated library",
)
@click.option(
    "--report/--noreport",
    default=False,
    is_flag=True,
    help="whether to create the report documentation",
)
@click.option(
    "--all",
    "do_all",
    default=False,
    is_flag=True,
    help="do all modes",
)
@click.option(
    "--nprocs",
    type=int,
    required=False,
    help="how many processes to use",
)
def command_do(config_file, generate, run, build, check, report, do_all, nprocs):
    # Set whether we do the modes.
    do = dict()
    do["generate"] = generate
    do["run"] = run
    do["build"] = build
    do["check"] = check
    do["report"] = report
    all_modes = ["generate", "run", "build", "check", "report"]
    if do_all:
        for mode in all_modes:
            do[mode] = True

    try:
        # Load the input data.
        with open(config_file, "r") as f:
            data = json.load(f)
        data["model"]["config_file"] = config_file

        # If nprocs is present, override.
        if nprocs:
            data["run"]["nprocs"] = nprocs
            data["check"]["nprocs"] = nprocs

        # Update paths in the model block.
        model = common.update_model(data["model"])

        # Run each enabled mode in sequence.
        for mode in all_modes:
            if do[mode]:
                output = common.fn_redirect(**data[mode], model=model)
                output_file = str(Path(model["work_dir"]) / mode) + ".json"
                common.logger.info(f"Writing {output_file} ...")
                with open(output_file, "w") as f:
                    f.write(json.dumps(output, indent=4))

    except ValueError as ve:
        common.logger.error(str(ve))
        return str(ve)


cli.add_command(command_do)


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
def command_link(names, paths, env, dest, show, dry_run):
    try:
        # Read all available libraries.
        registry0 = common.create_registry(paths, env)
        if show:
            print("{:40s} {}".format("name", "path"))
            for name in registry0:
                print("{:40s} {}".format(name, registry0[name].path))

        # Copy into reduced registry of only things we want.
        registry = dict()
        for name in names:
            if not name in registry0:
                raise ValueError("name={} not found in provided paths!".format(name))

            registry[name] = registry0[name]

        if not show:
            link.make_mini_arpdatatxt(dry_run, registry, dest)

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
    metavar="'{\"_type\": NAME, <OPTIONS>}'",
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
