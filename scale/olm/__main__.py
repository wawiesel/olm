import click
import sys
import scale.olm.internal as internal
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
@click.group(no_args_is_help=True)
def olm():
    pass


# ---------------------------------------------------------------------------------------
# OLM CREATE
# ---------------------------------------------------------------------------------------
#
# This is the entry point to run OLM CREATE command.
#
@click.command(
    name="create",
    no_args_is_help=True,
    epilog="""

**Usage**

Create a reactor library locally at :code:`uox_quick/_work/arpdata.txt`.

.. code:: console

  \b
  $ olm init --variant uox_quick
  $ olm create -j6 uox_quick/config.olm.json

""",
)
@click.argument("config_file", metavar="config.olm.json", type=click.Path(exists=True))
@click.option(
    "--generate/--nogenerate",
    default=None,
    is_flag=True,
    help="Whether to perform input generation.",
)
@click.option(
    "--run/--norun",
    default=None,
    is_flag=True,
    help="Whether to perform runs.",
)
@click.option(
    "--assemble/--noassemble",
    default=None,
    is_flag=True,
    help="Whether to assemble the ORIGEN library.",
)
@click.option(
    "--check/--nocheck",
    default=None,
    is_flag=True,
    help="Whether to check the generated library.",
)
@click.option(
    "--report/--noreport",
    default=None,
    is_flag=True,
    help="Whether to create the report documentation.",
)
@click.option(
    "--nprocs",
    "-j",
    type=int,
    default=3,
    help="How many processes to use.",
)
@internal.copy_doc(internal.create)
def olm_create(**kwargs):
    try:
        internal.create(**kwargs)
    except ValueError as ve:
        internal.logger.error(str(ve))
        return str(ve)


olm.add_command(olm_create)


# ---------------------------------------------------------------------------------------
# OLM INIT
# ---------------------------------------------------------------------------------------
#
# This is the entry point to run OLM INIT command.
#
@click.command(
    name="init",
    no_args_is_help=True,
    epilog="""

**Usage**

Choose from one of the variants when you pass the --list option.

.. code:: console

  \b
  $ olm init --list

By default creates a directory called `mox_quick` with the files.

.. code:: console

  \b
  $ olm init --variant mox_quick

""",
)
@click.argument("config_dir", type=str, required=False)
@click.option("--variant", "-a", type=str, default=None, help="Name of model variant.")
@click.option(
    "--list",
    "-l",
    "list_",
    is_flag=True,
    default=False,
    help="List all known variants and exit.",
)
@internal.copy_doc(internal.init)
def olm_init(**kwargs):
    try:
        internal.init(**kwargs)
    except ValueError as ve:
        internal.logger.error(str(ve))
        return str(ve)


olm.add_command(olm_init)


# ---------------------------------------------------------------------------------------
# OLM LINK
# ---------------------------------------------------------------------------------------
#
# This is the entry point to run OLM LINK command.
#
@click.command(
    name="link",
    no_args_is_help=True,
    epilog="""

**Usage**

First create and install a reactor library to $HOME/.olm

.. code:: console

  \b
  $ olm init --variant uox_quick
  $ olm create -j6 uox_quick/config.olm.json
  $ olm install --overwrite uox_quick/_work
  $ export SCALE_OLM_PATH=$HOME/.olm

In a SCALE input file, use a shell to link the library before the origami input. 
This will work with SCALE 6.2 and later:

.. code::

  \b
  =shell
  olm link uox_quick
  end
  =origami
  lib=[ uox_quick ]
  ...
  end

With SCALE 6.3.2 and later, the SCALE_OLM_PATH is searched directly by ORIGAMI and
the link is unnecessary.

""",
)
@click.argument("names", type=str, nargs=-1, metavar="NAME1 NAME2 ...")
@click.option(
    "--path",
    "-p",
    "paths",
    multiple=True,
    type=click.Path(exists=True),
    help="Path to prepend to SCALE_OLM_PATH.",
)
@click.option(
    "--env/--noenv",
    default=True,
    help="Whether to allow using the environment variable SCALE_OLM_PATH.",
)
@click.option(
    "--dest",
    "-d",
    default=os.getcwd(),
    type=click.Path(exists=True),
    help="Destination directory (default: current).",
)
@click.option(
    "--show",
    default=False,
    is_flag=True,
    help="Show all the known libraries.",
)
@click.option(
    "-o",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Allow overwriting of destination files.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Just emit commands without running them.",
)
@internal.copy_doc(internal.link)
def olm_link(**kwargs):
    try:
        internal.link(**kwargs)
    except ValueError as ve:
        internal.logger.error(str(ve))
        return str(ve)


olm.add_command(olm_link)


# ---------------------------------------------------------------------------------------
# OLM INSTALL
# ---------------------------------------------------------------------------------------
#
# This is the entry point to run OLM INSTALL command.
#
@click.command(
    name="install",
    no_args_is_help=True,
    epilog="""

**Usage**

Install a reactor library to $HOME/.olm after using :code:`olm create`.

.. code:: console

  \b
  $ olm install --overwrite uox_quick/_work
  $ export SCALE_OLM_PATH=$HOME/.olm

""",
)
@click.argument("work_dir", type=str, required=False)
@click.option(
    "--dest",
    "-d",
    type=str,
    default=None,
    help="Destination for the installation (defaults to $HOME/.olm).",
)
@click.option(
    "-o",
    "--overwrite",
    is_flag=True,
    default=False,
    help="Allow overwriting of destination files.",
)
@internal.copy_doc(internal.install)
def olm_install(**kwargs):
    try:
        internal.install(**kwargs)
    except ValueError as ve:
        internal.logger.error(str(ve))
        return str(ve)


olm.add_command(olm_install)


# ---------------------------------------------------------------------------------------
# OLM CHECK
# ---------------------------------------------------------------------------------------
#
# This is the entry point to run OLM CHECK command.
#
def methods_help(*methods):
    desc = "Run checking method NAME with options OPTS in JSON string format. The following checks are supported.\n\n"
    for m in methods:
        t = m.default_params()
        desc += "\n\b\nNAME={}, with <OPTS>\n".format(m.__name__)
        params = m.describe_params()
        for k in t:
            desc += "  {} - {} ({})\n".format(k, params[k], t[k])
        desc += "\n\n"
    return desc


import scale.olm.check as check


@click.command(
    name="check",
    no_args_is_help=True,
    epilog="""

**Usage**

Check a reactor library for quality.

.. code:: console

  \b
  $ olm init --variant uox_quick
  $ olm check -j6 -s '{"_type": "GridGradient", "eps0": 1e-6}' data/w17x17.arc.h5
  $ cat check.json    # Default output file.

""",
)
@click.argument("archive_file", type=str)
@click.option(
    "--output",
    "-o",
    "output_file",
    default="check.json",
    type=str,
    metavar="my.arc.h5",
    help="""Output file where results are written.""",
)
@click.option(
    "--sequence",
    "-s",
    "text_sequence",
    type=str,
    metavar='\'{"_type": "NAME", <OPTS>}\'',
    multiple=True,
    help=methods_help(check.GridGradient),
)
@click.option(
    "--nprocs",
    "-j",
    type=int,
    default=3,
    metavar="INT",
    help="How many processes to use.",
)
@internal.copy_doc(internal.check)
def olm_check(**kwargs):
    try:
        internal.check(**kwargs)
    except ValueError as ve:
        internal.logger.error(str(ve))
        return str(ve)


olm.add_command(olm_check)


# ---------------------------------------------------------------------------------------
# OLM SCHEMA
# ---------------------------------------------------------------------------------------
#
# This is the entry point to run OLM SCHEMA command.
#
@click.command(
    name="schema",
    no_args_is_help=True,
    epilog="""

**Usage**

Generate the schema for a specific :code:`_type` that could appear in a config.olm.json file.

.. code:: console

  \b
  $ olm schema 'scale.olm.generate.comp:uo2_simple'

""",
)
@click.argument("_type", type=str)
@click.option(
    "--color/--nocolor",
    is_flag=True,
    default=True,
    help="Output in color.",
)
@click.option(
    "--description/--nodescription",
    is_flag=True,
    default=False,
    help="Output description (mainly intended for sphinx docs).",
)
@click.option(
    "--infer",
    is_flag=True,
    default=False,
    help="Infer the schema.",
)
@click.option(
    "--state",
    is_flag=True,
    default=False,
    help="Infer the schema.",
)
@internal.copy_doc(internal.schema)
def olm_schema(**kwargs):
    try:
        from pygments import highlight, lexers, formatters
        import json

        d = internal.schema(**kwargs)
        formatted_json = json.dumps(d, indent=4)
        if kwargs["color"]:
            colorful_json = highlight(
                formatted_json, lexers.JsonLexer(), formatters.TerminalFormatter()
            )
            click.echo(colorful_json)
        else:
            print(formatted_json)

    except ValueError as ve:
        internal.logger.error(str(ve))
        return str(ve)


olm.add_command(olm_schema)
