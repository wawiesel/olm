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
    registry0 = internal._create_registry(paths, env)
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
        internal._make_mini_arpdatatxt(dry_run, registry, dest)

    return 0
