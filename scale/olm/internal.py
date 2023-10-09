"""
This :obj:`scale.olm.internal` module contains functions called from the click-based
command line interface. This is so __main__ contains little logic. There should be no
click dependence here--it should all be in __main__.

------------------------------------------------------------------------------------------
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
from typing import Callable, TypeVar, Any, Set
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

        # Here we make sure we remove anything after \f when we copy the docstring
        # as per click rules. This is so the command line docs and the HTML docs
        # for the CLI are the same. The API docs will use the local docstrings
        # and have everything, including what follows \f.
        s = copy_func.__doc__
        if not s:
            raise ValueError(
                "@copy_doc({}) will not work because it has an empty docstring!".format(
                    copy_func.__name__
                )
            )
        i = s.find("\f")
        if i >= 0:
            s = s[0:i]
        wrapper.__doc__ = s
        return wrapper

    return decorator


def create(
    config_file: str,
    generate: bool,
    run: bool,
    assemble: bool,
    check: bool,
    report: bool,
    nprocs: int,
):
    """Create ORIGEN reactor libraries!

    The create command manages an entire ORIGEN reactor library creation sequence
    which goes through these stages.

        1. **generate** the various inputs that cover the desired input space

        2. **run** the inputs with SCALE and generate individual libraries for each point in space

        3. **assemble** the individual libraries into an ORIGEN reactor library that can interpolate throughout the space

        4. **check** the quality of the ORIGEN reactor library

        5. **report** the entire creation process in HTML or PDF

    The linchpin of the process is a configuration file which must always be passed
    to the command. This file defines the input for each stage in JSON format.

    \f
    See :ref:`config.olm.json` for details.

    Args:
        config_dir: Directory where configuration files should be written.

        generate: Whether to generate.

        run: Whether to run.

        assemble: Whether to assemble.

        check: Whether to check.

        report: Whether to report.

        nprocs: Number of processess to use.

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
    init_path = Path(__file__).parent / "variants"
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


def init(config_dir: str, variant: str, list_: bool):
    """Initialize a new ORIGEN reactor library configuration.
    \f
    Args:
        config_dir: Directory where configuration files should be written.

        variant: Initialization variant to write.

        list\_: Just list available variants.

    """
    if list_ or variant == None:
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
                try:
                    shutil.copy(file, d)
                except shutil.SameFileError:
                    pass


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


def install(work_dir: str, dest: str, overwrite: bool = False, dry_run: bool = False):
    """Install ORIGEN reactor libraries!

    After creating a new ORIGEN reactor library, this command installs it to
    a dest on the file system. After this, the work directory can be
    deleted.

    \f
    Args:
        work_dir: Directory where ORIGEN library files are.

        dest: Destination directory to install the reactor libraries.

        overwrite: Overwrite at the destination.

        dry_run: Do not move any files.

    """
    logger.debug("Entering install with args", **locals())

    # Get useful paths.
    work_path = Path(work_dir)
    arpdata_txt = work_path / "arpdata.txt"
    arplibs = work_path / "arplibs"

    # Check the dest.
    if dest == None:
        for home in ["HOME", "HOMEDIR"]:
            if home in os.environ:
                dest = Path(os.environ[home]) / ".olm"
                break
    else:
        dest = Path(dest)

    # Determine if there is anything existing.
    registry0 = _create_registry(paths=[dest], env=False)

    # Checks for the potential install.
    blocks = core.ArpInfo.parse_arpdata(arpdata_txt)
    if len(blocks) > 1:
        lib_str = ",".join(blocks.keys())
        raise ValueError(
            f"Only one library expected in {arpdata_txt}. Found {lib_str}."
        )
    name = list(blocks.keys())[0]
    if name in registry0:
        install_arpdata_txt = dest / "arpdata.txt"
        if overwrite:
            logger.debug(f"Overwriting {name} in {install_arpdata_txt}!")
            for lib in registry0[name].lib_list:
                file = dest / "arplibs" / lib
                if not dry_run:
                    if file.exists():
                        os.remove(file)
                logger.debug(f"Deleting file {file} because overwrite of {name}!")
        else:
            raise ValueError(
                f"New library name={name} already exists in dest {install_arpdata_txt}! Will not overwrite!"
            )
    logger.info("Installing ", model=name, dest=str(dest))

    # Add new/updated one to the registry.
    arp_info = core.ArpInfo()
    arp_info.init_block(name, blocks[name])
    arp_info.path = arpdata_txt.resolve()
    arp_info.arplibs_dir = arplibs.resolve()
    registry0[name] = arp_info

    # Create a new "mini" arpdata.txt in the dest that has all the previous libraries
    # plus this new libraries.
    dest.mkdir(parents=True, exist_ok=True)
    _make_mini_arpdatatxt(dry_run, registry0, dest, replace=True)


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
    mod_fn = mod_fn.split(":")
    if len(mod_fn) != 2:
        raise ValueError(
            f"The expected form for {mod_fn} is the module_name:function_name, separated by a single colon."
        )

    this_module = sys.modules[mod_fn[0]]
    try:
        fn_handle = getattr(this_module, mod_fn[1])
        return fn_handle
    except:
        return None


def _indent(text, i):
    space = " " * i
    return space.join(("\n" + text.lstrip()).splitlines(True))


def _get_schema(_type: str, with_state: bool = False):
    """Get the schema for the type."""

    # Search for a hard-coded schema.
    mod, schema_fn = _type.split(":")
    _schema = mod + ":_schema_" + schema_fn
    fn = _get_function_handle(_schema)
    if fn == None:
        raise ValueError(
            f"No schema function associated with {_type}! Should be called {_schema}. You can try to infer with `olm schema --infer '{_type}'."
        )
    return fn(with_state=with_state)


def _infer_schema(_type: str, _exclude: Set[str] = set(), with_state: bool = False):
    """Infer a schema for the type."""
    import typing
    import pydantic
    import inspect

    # Shortcut to this common exclusion.
    if not with_state:
        _exclude.add("state")

    # Use inspect to get required arguments.
    required = {}
    fn = _get_function_handle(_type)
    for k, v in inspect.signature(fn).parameters.items():
        required[k] = v.default is v.empty

    # Iterate through type hints to collect types.
    z = typing.get_type_hints(fn)
    # If not a function try a class.
    if not z:
        z = typing.get_type_hints(fn.__init__)
    t = {}
    for k, v in z.items():
        # Skip hidden arguments that start with _ by convention.
        if k == "_type":
            # v.__args__[0] is the inner Literal[] because we allow the functions
            # to be called without _type so it is technically optional, but in
            # terms of JSON schema we want it required.
            t["olm_redirect_type"] = (v.__args__[0], ...)
            continue
        elif k.startswith("_") or (k in _exclude):
            continue
        if required[k]:
            t[k] = (v, ...)
        else:
            t[k] = (v, None)

    # Create a model using the arguments to get a JSON schema.
    Model = pydantic.create_model(
        _type.split(":")[1],
        **t,
    )

    e = Model.model_json_schema()

    # Fix up.
    try:
        i = e["required"].index("olm_redirect_type")
        e["required"][i] = "_type"
        e["properties"]["_type"] = e["properties"].pop("olm_redirect_type")
    except:
        pass

    return e


def _collapsible_json(title: str, json_str: str):
    # Indent 4 for code + 4 for collapse = 8.
    json_str = _indent(json_str, 8)

    return f"""
.. collapse:: {title}

    .. code:: JSON

{json_str}

.. only::latex

    END {title}

"""


def _get_schema_description(_type: str):
    """Return a description for the schema for the documentation."""

    # Populate this string.
    description = ""

    # Get example arguments, format as string and indent properly.
    fn_path = _type.replace(":", ".")
    mod, fn_name = _type.split(":")
    test_fn = _get_function_handle(mod + ":_test_args_" + fn_name)
    args = test_fn()
    section = mod.replace("scale.olm.", "").replace(".", "/")
    description += (
        f'Specified with :code:`"_type": "{_type}"` in **config.olm.json**.\n'
    )
    description += _collapsible_json(
        "Example input in config.olm.json/" + section, json.dumps(args, indent=4)
    )
    description += "\n"

    # Get intermediate input and output only for generate.
    if mod.startswith("scale.olm.generate"):
        args = test_fn(with_state=True)
        description += _collapsible_json(
            f"Args passed to Python function: {fn_path}", json.dumps(args, indent=4)
        )
        description += "\n"

        # Get example output
        fn = _get_function_handle(_type)
        out = fn(**args)
        if isinstance(out, dict):
            if "_input" in out:
                del out["_input"]
            if "work_dir" in out:
                out["work_dir"] = "/path/to/_work"
        var = mod.replace("scale.olm.generate.", "")
        description += _collapsible_json(
            f"Data available in template: {var}", json.dumps(out, indent=4)
        )
        description += "\n"

    # Add see also.
    description += f"See also: :obj:`{fn_path}`"

    return description


def schema(
    _type: str,
    color: bool,
    description: bool = False,
    infer: bool = False,
    state: bool = False,
):
    """Emit the JSON schema corresponding to a particular _type."""

    logger.debug(f"Trying to determine schema for {_type}")
    if infer:
        e = _infer_schema(_type, with_state=state)
    else:
        e = _get_schema(_type, with_state=state)

    if description:
        logger.debug(f"Adding description to schema for {_type}")
        e["$$description"] = _get_schema_description(_type).split("\n")

    return e


def _fn_redirect(_type, **x):
    """Uses the _type input to find a function handle of that name, then executes with all the
    data except the _type."""
    fn_x = _get_function_handle(_type)
    return fn_x(**x)


def run_command(
    command_line: str,
    check_return_code: bool = True,
    echo: bool = True,
    error_match: str = "Error",
):
    """Run a command as a subprocess.

    Throw on bad error code or finding the error_match string in the output.

    \f
    Args:
        command_line: Command line to run.

        check_return_code: Whether to check the return code.

        echo: Whether to echo STDOUT and STDERR.

        error_match: String to match to decide if there are errors.

    """
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


def link(
    names: list[str],
    paths: list[str],
    env: bool,
    dest: str,
    show: bool,
    overwrite: bool,
    dry_run: bool,
):
    """Link custom ORIGEN reactor libraries to a SCALE calculation.
    \f
    Args:
        names: Names of libraries to link.

        paths: List of paths to prepend to SCALE_OLM_PATH.

        env: Whether to use the environment or not.

        dest: Destination directory to link the reactor libraries.

        show: Just show the available libraries.

        overwrite: Overwrite at the destination.

        dry_run: Do not move any files.

    """

    # Read all available libraries.
    registry0 = _create_registry(paths, env)
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
        _make_mini_arpdatatxt(dry_run, registry, dest, replace=overwrite)

    return 0


def check(archive_file: str, output_file: str, text_sequence: list[str], nprocs: int):
    """Run a sequence of checks on ORIGEN archives.

    \f
    Args:
        archive_file: Input archive file to check.

        output_file: Output file to write test results.

        text_sequence: List of strings, each one the JSON string of a test.

        nprocs: Number of simultaneous processes to use.

    """
    import scale.olm.check

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
            logger.error(
                "Libraries in HDF5 archive format must end in .arc.h5 but found",
                archive_file=archive_file,
            )
            return
        name = re.sub("\.arc\.h5$", "", archive_file)
        work_dir = Path(archive_file).parent

    output = scale.olm.check.sequencer(
        sequence, _model={"name": name}, _env={"work_dir": work_dir, "nprocs": nprocs}
    )

    logger.info(f"Writing {output_file} ...")
    with open(output_file, "w") as f:
        f.write(json.dumps(output, indent=4))
