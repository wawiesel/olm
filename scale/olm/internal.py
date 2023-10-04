"""
This package directs results to other parts of the system. The command line
interface should call functions from here and not implement any of its own
logic. No click dependence should be here. This is mainly for unit testing.
Nothing should depend on this package. It is a top level package.
"""

import pathlib
from pathlib import Path
from tqdm import tqdm, tqdm_notebook
import copy
import glob
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scale.olm.core as core
import shutil
import structlog
import subprocess
import sys
import time

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        int(os.environ.get("SCALE_LOG_LEVEL", logging.INFO))
    ),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)

logger = structlog.get_logger(__name__)


# This is special for the docstring copier below.
from functools import wraps
from typing import Callable, TypeVar, Any
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def copy_doc(
    copy_func: Callable[..., Any]
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Prepends the docstring from another function. This function is intended to
    be used as a decorator.

    Args:
        copy_func (Callable[..., Any]): The function whose docstring should be copied.

    Returns:
        Callable[[Callable[P, T]], Callable[P, T]]: The decorator function.

    Examples:

        Define a function with a docstring.

        >>> def some():
        ...    '''This is a some doc string'''
        ...    return None

        Copy some docstring to pig function.

        >>> @copy_doc(some)
        ... def pig():
        ...    return None

        >>> pig.__doc__
        'This is a some doc string'

    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        wrapper.__doc__ = copy_func.__doc__
        return wrapper

    return decorator


def create(config_file, generate, run, assemble, check, report, nprocs):
    # Note this docstring should appear in the command line interface help message so
    # do not add additional information that would not be valid for the CLI.
    """Create ORIGEN reactor libraries!

    The create command manages an entire ORIGEN reactor library creation sequence
    which goes through these stages.

        1. `generate` the various inputs that cover the desired input space

        2. `run` the inputs with SCALE and generate individual libraries for each point in space

        3. `assemble` the individual libraries into an ORIGEN reactor library that can interpolate throughout the space

        4. `check` the quality of the ORIGEN reactor library

        5. `report` the entire creation process in HTML or PDF

    The linchpin of the process is a configuration file which must always be passed
    to the command. This file defines the input for each stage in JSON format.
    """
    logger.debug("Entering create with args", **locals())

    # Set whether we do each stage.
    do = dict()
    if generate != None:
        do["generate"] = generate
    if run != None:
        do["run"] = run
    if assemble != None:
        do["assemble"] = assemble
    if check != None:
        do["check"] = check
    if report != None:
        do["report"] = report
    logger.debug("User input for stages", create=do)

    # Check for any positive/negative explicitly specified by user so we can
    # handle doing/not doing the other stages appropriately.
    pos = 0
    neg = 0
    for k, v in do.items():
        if v:
            pos += 1
        else:
            neg += 1
    if pos > 0:
        do_other = False  # If any activations, assume the others should be off.
    elif neg > 0:
        do_other = True  # If any negations, assume the others should be on.
    else:
        do_other = True  # Default do everything.
    all_stages = ["generate", "run", "assemble", "check", "report"]
    for stage in all_stages:
        if not stage in do:
            do[stage] = do_other
    logger.debug("Final stages", create=do)

    # Create the environment.
    _env, config = _load_env(config_file, nprocs)

    # Run each enabled stage in sequence.
    for stage in all_stages:
        if do[stage]:
            logger.debug("Configuring", stage=stage, **config[stage])
            output = _fn_redirect(**config[stage], _model=config["model"], _env=_env)
            output_file = str(Path(_env["work_dir"]) / stage) + ".olm.json"
            logger.info(f"Writing {output_file}")
            with open(output_file, "w") as f:
                f.write(json.dumps(output, indent=4))


def _get_init_variants():
    init_path = Path(__file__).parent / "init"
    logger.debug("Initialization variants located", init_path=init_path)
    variants = [
        str(Path(v).relative_to(init_path)) for v in glob.glob(str(init_path / "*"))
    ]
    return init_path, variants


def _write_init_variant(variant, config_dir):
    logger.info("Creating init dir", config_dir=config_dir)
    config_path = Path(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)

    init_path, variants = _get_init_variants()
    if not variant in variants:
        logger.error(f"Requested variant={variant} unknown!", known_variants=variants)
        return
    variant_path = init_path / variant
    c0 = variant_path / "config.olm.json"
    d0 = config_path / "config.olm.json"
    logger.info("Copying config from", source=str(c0), destination=str(d0))
    shutil.copyfile(c0, d0)

    tm = core.TemplateManager()
    variant_files_json = variant_path / "files.json"
    if variant_files_json.exists():
        with open(variant_files_json, "r") as f:
            files = json.load(f)
            for file, path in files.items():
                resolved_path = tm.path(path)
                logger.info(
                    f"Copying files for variant={variant}",
                    source=str(resolved_path),
                    destination=str(config_path / file),
                )
                shutil.copyfile(resolved_path, config_path / file)


def init(config_dir, variant, list_variants):
    # Note this docstring should appear in the command line interface help message so
    # do not add additional information that would not be valid for the CLI.
    """
    Initialize a new ORIGEN reactor library configuration.

    Choose from one of the variants when you pass the --list option.

        olm init --variant mox_quick

    By default creates a directory called `mox_quick` with the files.
    """
    if list_variants or variant == None:
        init_path, variants = _get_init_variants()
        i = 0
        for v in variants:
            i += 1
            print("{:d}. {}".format(i, v))
        return

    if not config_dir:
        logger.debug("Assumed local directory", config_dir=variant)
        config_dir = variant

    _write_init_variant(variant, config_dir)


def _make_mini_arpdatatxt(dry_run, registry, dest, replace=False):
    """Create a local arpdata.txt and arplibs"""

    logger.debug(f"Setting up at destination dir={dest}")

    # Concatenate the blocks from each name.
    mini_arpdata = ""
    files_to_copy = []
    for name in registry:
        arpinfo = registry[name]
        path = arpinfo.path
        logger.debug(f"Appending {name} from {path}")
        mini_arpdata += f"!{name}\n" + arpinfo.block + "\n"
        for i in range(arpinfo.num_libs()):
            files_to_copy.append(
                Path(path.parent) / "arplibs" / arpinfo.get_lib_by_index(i)
            )

    # Create an arpdata.txt file with the concatenated content.
    a = Path(dest) / "arpdata.txt"
    if a.exists() and (not replace):
        logger.error(
            f"arpdata.txt already exists at path={path} and will not be overwritten"
        )
    else:
        if dry_run:
            logger.debug("Not writing {a} because --dry-run")
        else:
            if a.exists():
                os.remove(a)
            with open(a, "w") as f:
                f.write(mini_arpdata)

    # Create the arplibs directory by copying data files.
    d = Path(dest) / "arplibs"
    if d.exists() and (not replace):
        logger.error(
            f"arplibs directory already exists at path={path} and will not be overwritten"
        )
    else:
        if dry_run:
            logger.debug(f"Not writing {d} because --dry-run")
        else:
            d.mkdir(exist_ok=True)
        for file in files_to_copy:
            if dry_run:
                logger.debug(f"Not copying {file} to {d} because --dry-run")
            else:
                logger.debug(f"Copying {file} to {d}")
                shutil.copy(file, d)


def _update_registry(registry, path):
    """Update a registry of library names using all the paths"""

    p = Path(path)
    logger.debug("Searching path={}".format(p))

    # Look for arpdata.txt version.
    q1 = p / "arpdata.txt"
    q1.resolve()
    if q1.exists():
        r = p / "arplibs"
        r.resolve()
        if not r.exists():
            logger.warning(
                "{} exists but the paired arplibs/ directory at {} does not--ignoring libraries listed".format(
                    q1, r
                )
            )
        else:
            logger.debug("Found arpdata.txt!")
            blocks = core.ArpInfo.parse_arpdata(q1)
            for n in blocks:
                if n in registry:
                    logger.warning(
                        "library name {} has already been registered at path={} ignoring same name found at {}".format(
                            n, registry[n].path, p
                        )
                    )
                else:
                    logger.debug("Found library name {} in {}!".format(n, q1))
                    arpinfo = core.ArpInfo()
                    arpinfo.init_block(n, blocks[n])
                    arpinfo.path = q1
                    arpinfo.arplibs_dir = r
                    registry[n] = arpinfo


def _create_registry(paths, env):
    """Search for a library 'name', at every path in 'paths', optionally using
    environment variable SCALE_OLM_PATH"""
    registry = dict()

    logger.debug("Searching provided paths ({})...".format(len(paths)))
    for path in paths:
        _update_registry(registry, path)

    if env and "SCALE_OLM_PATH" in os.environ:
        env_paths = os.environ["SCALE_OLM_PATH"].split(":")
        logger.debug("Searching SCALE_OLM_PATH paths ({})...".format(len(env_paths)))
        for path in env_paths:
            _update_registry(registry, path)

    return registry


def install(work_dir, destination, overwrite=False, dry_run=False):
    # Note this docstring should appear in the command line interface help message so
    # do not add additional information that would not be valid for the CLI.
    """Install ORIGEN reactor libraries!

    After creating a new ORIGEN reactor library, this command installs it to
    a destination on the file system. After this, the work directory can be
    deleted.

    """
    logger.debug("Entering install with args", **locals())

    # Get useful paths.
    work_path = Path(work_dir)
    arpdata_txt = work_path / "arpdata.txt"
    arplibs = work_path / "arplibs"

    # Check the destination.
    if destination == None:
        for home in ["HOME", "HOMEDIR"]:
            if home in os.environ:
                destination = Path(os.environ[home]) / ".olm"
                break
    else:
        destination = Path(destination)

    # Determine if there is anything existing.
    registry0 = _create_registry(paths=[destination], env=False)

    # Checks for the potential install.
    blocks = core.ArpInfo.parse_arpdata(arpdata_txt)
    if len(blocks) > 1:
        lib_str = ",".join(blocks.keys())
        raise ValueError(
            f"Only one library expected in {arpdata_txt}. Found {lib_str}."
        )
    name = list(blocks.keys())[0]
    if name in registry0:
        install_arpdata_txt = destination / "arpdata.txt"
        if overwrite:
            logger.debug(f"Overwriting {name} in {install_arpdata_txt}!")
            for lib in registry0[name].lib_list:
                file = destination / "arplibs" / lib
                if not dry_run:
                    if file.exists():
                        os.remove(file)
                logger.debug(f"Deleting file {file} because overwrite of {name}!")
        else:
            raise ValueError(
                f"New library name={name} already exists in destination {install_arpdata_txt}! Will not overwrite!"
            )
    logger.info("Installing ", model=name, destination=str(destination))

    # Add new/updated one to the registry.
    arp_info = core.ArpInfo()
    arp_info.init_block(name, blocks[name])
    arp_info.path = arpdata_txt.resolve()
    arp_info.arplibs_dir = arplibs.resolve()
    registry0[name] = arp_info

    # Create a new "mini" arpdata.txt in the destination that has all the previous libraries
    # plus this new libraries.
    destination.mkdir(parents=True, exist_ok=True)
    _make_mini_arpdatatxt(dry_run, registry0, destination, replace=True)


def _raise_scalerte_error():
    raise ValueError(
        """
    The scalerte executable was not found. Do one of the following and rerun.
    1. Set environment variable SCALE_DIR to a valid location of a SCALE install with
       scalerte at ${SCALE_DIR}/bin/scalerte. Do this by writing a command in the
       command line, for example: 
       "export SCALE_DIR=/Applications/SCALE-6.3.0.app/Contents/Resources"
    2. Set environment variable OLM_SCALERTE directly to scalerte by writing a command 
       in the command line, for example:
       "export OLM_SCALERTE=/Applications/SCALE-6.3.0.app/Contents/Resources/bin/scalerte"
        """
    )


def _raise_obiwan_error():
    raise ValueError(
        """
    The obiwan executable was not found. Do one of the following and rerun.
    1. Set environment variable SCALE_DIR to a valid location of a SCALE install with
       obiwan at ${SCALE_DIR}/bin/obiwan.
    2. Set environment variable OLM_OBIWAN directly to obiwan."""
    )


def _load_env(config_file: str, nprocs: int = 0):
    """Update the environment with paths."""

    # Load the input data.
    config_path = Path(config_file).resolve()
    logger.info("Loading configuration file", config_file=str(config_path))
    with open(config_path, "r") as f:
        config = json.load(f)
    model = config["model"]
    name = model["name"]
    sources = model.get("sources", [])
    revision = model.get("revision", ["UNKNOWN"])[-1]
    logger.info("Creating new", model=name, sources=sources, revision=revision)

    # Determine where the work will take place.
    work_path = Path(os.environ.get("OLM_WORK_DIR", config_path.parent / "_work"))

    # Create the working directory.
    if not work_path.exists():
        logger.info("Creating OLM_WORK_DIR", work_dir=str(work_path))
        work_path.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("Using existing OLM_WORK_DIR", work_dir=str(work_path))

    # Attempt to load an existing environment.
    env_path = work_path / "env.olm.json"
    if env_path.exists():
        logger.info("Loading existing environment", env_file=str(env_path))
        try:
            with open(env_path, "r") as f:
                env = json.load(f)
        except:
            logger.info("Problem loading, re-initializing ...")
            env = {}
    else:
        env = {}

    # Overwrite with any new data.
    env["config_file"] = str(config_path)
    env["work_dir"] = str(work_path)

    if "SCALE_DIR" in os.environ:
        scale_path = Path(os.environ["SCALE_DIR"])
        env["scalerte"] = str(scale_path / "bin" / "scalerte")
        logger.info("From SCALE_DIR environment variable", scalerte=env["scalerte"])
        env["obiwan"] = str(scale_path / "bin" / "obiwan")
        logger.info("From SCALE_DIR environment variable", obiwan=env["obiwan"])
    if "OLM_SCALERTE" in os.environ:
        env["scalerte"] = str(Path(os.environ["OLM_SCALERTE"]))
        logger.info("From OLM_SCALERTE environment variable", scalerte=env["scalerte"])
    if "OLM_OBIWAN" in os.environ:
        env["obiwan"] = str(Path(os.environ["OLM_OBIWAN"]))
        logger.info("From OLM_OBIWAN environment variable", obiwan=env["obiwan"])
    if nprocs > 0:
        env["nprocs"] = nprocs
        logger.info("Setting default number of workers", nprocs=nprocs)

    # Write out new environment.
    with open(env_path, "w") as f:
        json.dump(env, f, indent=4)

    return env, config


def _get_function_handle(mod_fn):
    """Takes module:function like uvw:xyz and returns the function handle to the
    function 'xyz' within the module 'uvw'."""
    mod, fn = mod_fn.split(":")
    this_module = sys.modules[mod]
    fn_handle = getattr(this_module, fn)
    return fn_handle


def _fn_redirect(_type, **x):
    """Uses the _type input to find a function handle of that name, then executes with all the
    data except the _type."""
    fn_x = _get_function_handle(_type)
    return fn_x(**x)


def run_command(command_line, check_return_code=True, echo=True, error_match="Error"):
    """Run a command as a subprocess. Throw on bad error code or finding 'Error' in the output."""
    logger.info("Running external ", command_line=command_line)
    p = subprocess.Popen(
        command_line,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        encoding="utf-8",
    )

    text = ""
    while True:
        line = p.stdout.readline()
        text += line
        if error_match in line:
            logger.debug("{error_match} in {line}")
            raise ValueError(line.strip())
        elif echo:
            logger.info(line.rstrip())
        else:
            logger.debug(line.rstrip())
        if not line:
            break

    if not p.returncode:
        retcode = 1
    else:
        retcode = p.returncode

    if check_return_code:
        if retcode != 0:
            if text.strip() == "":
                raise ValueError(
                    f"command line='{command_line}' failed to run in the shell. Check this is a valid path or recognized executable."
                )
            else:
                msg = p.stderr.read().strip()
                if retcode < 0:
                    logger.info(
                        f"Negative return code {retcode} on last command:\n{command_line}\n"
                    )
                    raise ValueError(str(msg))

    return text


def _runtime_in_hours(runtime):
    """Convert runtime in seconds to well-formatted runtime in hours."""
    return "{:.2g}".format(runtime / 3600.0)


def _execute_makefile(dry_run, _env, base_path, input_list):
    """Kernel to execute makefile-based runs

    The main thing this provides is some parallelism opportunity and graceful job
    cancelling. We should move to using the pure python executors in the future,
    like the wrapper in core.ThreadPoolExecutor.

    """
    import scale.olm.core as core

    if not "scalerte" in _env:
        _raise_scalerte_error()

    scalerte = _env["scalerte"]

    input_listing = " ".join(input_list)

    contents = f"""
outputs = $(patsubst %.inp, %.out, {input_listing})

.PHONY: all

all: $(outputs)

%.out: %.inp
\t@rm -f $@.FAILED
\t{scalerte} $<
\t@grep 'Error' $@ && mv -f $@ $@.FAILED && echo "^^^^^^^^^^^^^^^^ errors from $<" || true

clean:
\trm -f $(outputs)
"""

    make_file = base_path / "Makefile"
    with open(make_file, "w") as f:
        f.write(contents)

    version = core.ScaleRunner(scalerte).version
    logger.info("Running SCALE", version=version)

    nprocs = _env["nprocs"]
    command_line = f"cd {base_path} && make -j {nprocs}"
    if dry_run:
        logger.warning("No SCALE runs will be performed because dry_run=True!")
    else:
        run_command(command_line)

    # Get file listing.
    runs = list()
    total_runtime = 0
    for input0 in input_list:
        input = base_path / input0
        output = Path(input).with_suffix(".out")
        success = output.exists()
        runtime = core.ScaleOutfile.get_runtime(output) if success else 3.6e6
        total_runtime += runtime
        runs.append(
            {
                "input_file": str(input),
                "output_file": str(output),
                "success": success,
                "runtime_hrs": _runtime_in_hours(runtime),
            }
        )

    return {
        "scalerte": str(scalerte),
        "base_dir": str(base_path.relative_to(_env["work_dir"])),
        "dry_run": dry_run,
        "nprocs": nprocs,
        "command_line": command_line,
        "version": version,
        "runs": runs,
        "total_runtime_hrs": _runtime_in_hours(total_runtime),
    }
