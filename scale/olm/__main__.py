import click
import sys
import scale.olm.common as common
import json
import structlog
import os


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
        #     	if libinfo.format==format:
        #     		link.copy(dry_run, libinfo, dest)
        #     	else:
        #     		link.convert(dry_run, libinfo, format, dest)
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
    desc = "run checking method NAME with options OPTS (in JSON format) with the following supported checks: \n\n"
    for m in methods:
        desc += m.__name__ + " '" + json.dumps(m.default_params().__dict__) + "' where "
        params = m.describe_params()
        for p in params:
            desc += p + " is " + params[p] + ", "
        desc = desc[:-2]  # remove last comma
        desc += "\n\n"
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
    help="results output file",
)
@click.option(
    "--method",
    "-m",
    nargs=2,
    type=str,
    metavar="NAME '{OPTS}'",
    multiple=True,
    help=methods_help(check.GridGradient, check.Continuity),
)
def command_check(archive_file, output_file, method):
    try:
        # Process all the input.
        run_list = []
        this_module = sys.modules["scale.olm.check"]
        seq = 0
        for m in method:
            name, json_str = m
            common.logger.info(
                "Checking method options for method={}, sequence={}".format(name, seq)
            )
            seq += 1
            this_class = getattr(this_module, name)
            params = json.loads(json_str)

            run_list.append(this_class(params))

        # Read the archive.
        archive = common.Archive(archive_file)

        # Execute in sequence.
        p = 0
        seq = 0
        output = dict()
        for r in run_list:
            common.logger.info("Running checking sequence={}".format(seq))
            info = r.run(archive)
            output[str(seq)] = {"name": name, "info": info.__dict__}
            seq += 1
            if not info.test_pass:
                p = 1
            with open(output_file, "w") as fp:
                fp.write(json.dumps(output, indent=4))
            common.logger.info("Updated output file={}".format(output_file))
        common.logger.info("Finished without exception and status code={}".format(p))

        return p

    except ValueError as ve:
        common.logger.error(str(ve))
        return str(ve)


cli.add_command(command_check)
