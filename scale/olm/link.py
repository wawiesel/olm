import scale.olm.internal as internal
import scale.olm.core as core
from pathlib import Path
import shutil
import os


def link(names, paths, env, dest, show, dry_run):
    """
    Link in custom ORIGEN Reactor Libraries!
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
        _make_mini_arpdatatxt(dry_run, registry, dest)

    return 0


def _make_mini_arpdatatxt(dry_run, registry, dest):
    """Create a local arpdata.txt and arplibs"""

    internal.logger.debug(f"Setting up at destination dir={dest}")

    # Concatenate the blocks from each name.
    mini_arpdata = ""
    files_to_copy = []
    for name in registry:
        arpinfo = registry[name]
        path = arpinfo.path
        internal.logger.debug(f"Linking {name} from {path}")
        mini_arpdata += f"!{name}\n" + arpinfo.block
        for i in range(arpinfo.num_libs()):
            files_to_copy.append(
                Path(path.parent) / "arplibs" / arpinfo.get_lib_by_index(i)
            )

    # Create an arpdata.txt file with the concatenated content.
    a = Path(dest) / "arpdata.txt"
    if a.exists():
        internal.logger.error(
            f"arpdata.txt already exists at path={path} and will not be overwritten"
        )
    else:
        if dry_run:
            internal.logger.debug("Not writing {a} because --dry-run")
        else:
            with open(a, "w") as f:
                f.write(mini_arpdata)

    # Create the arplibs directory by copying data files.
    d = Path(dest) / "arplibs"
    if d.exists():
        internal.logger.error(
            f"arplibs directory already exists at path={path} and will not be overwritten"
        )
    else:
        if dry_run:
            internal.logger.debug(f"Not writing {d} because --dry-run")
        else:
            os.mkdir(d)
        for file in files_to_copy:
            if dry_run:
                internal.logger.debug(f"Not copying {file} to {d} because --dry-run")
            else:
                internal.logger.debug(f"Copying {file} to {d}")
                shutil.copy(file, d)


def _update_registry(registry, path):
    """Update a registry of library names using all the paths"""

    p = Path(path)
    internal.logger.debug("Searching path={}".format(p))

    # Look for arpdata.txt version.
    q1 = p / "arpdata.txt"
    q1.resolve()
    if q1.exists():
        r = p / "arplibs"
        r.resolve()
        if not r.exists():
            internal.logger.warning(
                "{} exists but the paired arplibs/ directory at {} does not--ignoring libraries listed".format(
                    q1, r
                )
            )
        else:
            internal.logger.debug("Found arpdata.txt!")
            blocks = core.ArpInfo.parse_arpdata(q1)
            for n in blocks:
                if n in registry:
                    internal.logger.warning(
                        "library name {} has already been registered at path={} ignoring same name found at {}".format(
                            n, registry[n].path, p
                        )
                    )
                else:
                    internal.logger.debug("Found library name {} in {}!".format(n, q1))
                    arpinfo = core.ArpInfo()
                    arpinfo.init_block(n, blocks[n])
                    arpinfo.path = q1
                    arpinfo.arplibs_dir = r
                    registry[n] = arpinfo


def _create_registry(paths, env):
    """Search for a library 'name', at every path in 'paths', optionally using
    environment variable SCALE_OLM_PATH"""
    registry = dict()

    internal.logger.debug("Searching provided paths ({})...".format(len(paths)))
    for path in paths:
        _update_registry(registry, path)

    if env and "SCALE_OLM_PATH" in os.environ:
        env_paths = os.environ["SCALE_OLM_PATH"].split(":")
        internal.logger.debug(
            "Searching SCALE_OLM_PATH paths ({})...".format(len(env_paths))
        )
        for path in env_paths:
            _update_registry(registry, path)

    return registry
