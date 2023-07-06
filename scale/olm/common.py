import h5py
from tqdm import tqdm, tqdm_notebook
import numpy as np
import structlog
import json
from pathlib import Path
import os
import matplotlib.pyplot as plt
import sys
import os

logger = structlog.getLogger(__name__)


class LibInfo:
    def __init__(self):
        self.format = "arpdata.txt"


def update_model(model):
    """Update the model section with paths."""

    # Find SCALE and utils.
    scale_env_var = model["scale_env_var"]
    if not scale_env_var in os.environ:
        logger.error(
            f"Environment variable scale_env_var='{scale_env_var}' must be set!"
        )
        raise ValueError

    scale_dir = os.environ[scale_env_var]
    scalerte = Path(scale_dir) / "bin" / "scalerte"
    obiwan = Path(scale_dir) / "bin" / "obiwan"

    model["scale_dir"] = str(scale_dir)
    model["scalerte"] = str(scalerte)
    model["obiwan"] = str(obiwan)

    # Main directory is where the config file is.
    if not "config_file" in model:
        model["config_file"] = None
        model["dir"] = os.getcwd()
    else:
        config_file = Path(model["config_file"]).resolve()
        model["config_file"] = str(config_file)
        model["dir"] = os.path.dirname(config_file)
    dir = Path(model["dir"])
    dir = dir.resolve()
    model["dir"] = str(dir)

    # Working directory for calculations.
    if not "work_dir" in model:
        model["work_dir"] = "."
    work_dir = dir / model["work_dir"]
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    model["work_dir"] = str(work_dir)

    # Final archive file.
    archive_file = work_dir / (model["name"] + ".h5")
    archive_file.resolve()
    model["archive_file"] = str(archive_file)

    return model


def get_function_handle(mod_fn):
    """Takes module:function like scale.olm.common:fuelcomp_uox_simple and returns
    the function handle to the function."""
    mod, fn = mod_fn.split(":")
    this_module = sys.modules[mod]
    fn_handle = getattr(this_module, fn)
    return fn_handle


def fn_redirect(x):
    """Takes a dictionary and uses the '.type' key to find a function handle of that name,
    then executes with all the keys except that type."""
    fn_x = get_function_handle(x[".type"])
    del x[".type"]
    return fn_x(**x)


def pass_through(**x):
    """Simple pass through used with the olm.json function specification."""
    return x


def execute_repo(model, generate, run, build, check, report):
    return {
        "model": fn_redirect(model),
        "generate": fn_redirect({"model": model, **generate}),
        "run": fn_redirect({"model": model, **run}),
        "build": fn_redirect({"model": model, **build}),
        "check": fn_redirect({"model": model, **check}),
        "report": fn_redirect({"model": model, **report}),
    }


def parse_arpdata(path):
    """Get all the library names on an arpdata.txt"""
    names = list()
    with open(path, "r") as f:
        for line in f.readlines():
            if line.startswith("!"):
                names.append(line[1:].strip())
    return names


def update_registry(registry, path):
    """Update a registry of library names using all the paths"""

    p = Path(path)
    logger.info("searching path={}".format(p))

    # Look for arpdata.txt version.
    q1 = p / "arpdata.txt"
    q1.resolve()
    if q1.exists():
        r = p / "arplibs"
        r.resolve()
        if not r.exists():
            logger.warning(
                "{} exists but the paired arplibs/ directory at {} does not--disregarding libraries".format(
                    q1, r
                )
            )
        else:
            logger.info("found arpdata.txt!")
            for n in parse_arpdata(q1):
                if n in registry:
                    logger.warning(
                        "library name {} has already been registered at path={} ignoring same name found at {}".format(
                            n, registry[n].path, p
                        )
                    )
                else:
                    logger.info("found library name {} in {}!".format(n, p))
                    libinfo = LibInfo()
                    libinfo.format = "arpdata.txt"
                    libinfo.name = n
                    libinfo.path = p
                    registry[n] = libinfo

    # Look for archive version at ${path}/*/olm.json


def create_registry(paths, env):
    """Search for a library 'name', at every path in 'paths', optionally using
    environment variable SCALE_OLM_PATH"""
    registry = dict()

    logger.info("searching provided paths ({})...".format(len(paths)))
    for path in paths:
        update_registry(registry, path)

    if env and "SCALE_OLM_PATH" in os.environ:
        env_paths = os.environ["SCALE_OLM_PATH"].split(":")
        logger.info("searching SCALE_OLM_PATH paths ({})...".format(len(env_paths)))
        for path in env_paths:
            update_registry(registry, path)

    return registry


def plot_hist(x):
    """Plot histograms from relative and absolute histogram data (rhist,ahist)."""

    plt.hist2d(
        np.log10(x.rhist),
        np.log10(x.ahist),
        bins=np.linspace(-40, 20, 100),
        cmin=1,
        alpha=0.2,
    )
    ind1 = (x.rhist > x.epsr) & (x.ahist > x.epsa)
    h = plt.hist2d(
        np.log10(x.rhist[ind1]),
        np.log10(x.ahist[ind1]),
        bins=np.linspace(-40, 20, 100),
        cmin=1,
        alpha=1.0,
    )
    ind2 = x.rhist > x.epsr
    plt.hist2d(
        np.log10(x.rhist[ind2]),
        np.log10(x.ahist[ind2]),
        bins=np.linspace(-40, 20, 100),
        cmin=1,
        alpha=0.6,
    )
    plt.colorbar(h[3])
    plt.xlabel(r"$\log \tilde{h}_{ijk}$")
    plt.ylabel(r"$\log h_{ijk}$")
    plt.grid()
    plt.show()


class Archive:
    """Simple class to read an ORIGEN Archive into memory. The hierarchy of ORIGEN
    data is a Transition Matrix is the necessary computational piece. A Library is a
    time-dependent sequence of Transition Matrices. An Archive is a multi-dimensional
    interpolatable space of Libraries."""

    def __init__(self, file):
        logger.info("Loading archive file={}".format(file))

        self.file_name = file
        self.h5 = h5py.File(file, "r")
        (
            self.axes_names,
            self.axes_values,
            self.axes_shape,
            self.ncoeff,
            self.nvec,
        ) = Archive.extract_axes(self.h5)

        # populate coefficient data
        self.coeff = np.zeros((*self.axes_shape, self.ncoeff))
        data = self.h5["incident"]["neutron"]
        for i in tqdm(data.keys()):
            if i != "TransitionStructure":
                d = Archive.get_indices(
                    self.axes_names, self.axes_values, data[i]["tags"]["continuous"]
                )
                dn = (*d, slice(None), slice(None))
                self.coeff[dn] = data[i]["matrix"]

        # Add another point if the dimension only has one so that we can make it easier
        # to do operations like gradients.
        n = len(self.axes_shape)
        for i in range(n):
            if self.axes_shape[i] == 1:
                self.axes_shape[i] = 2
                x0 = self.axes_values[i][0]
                if x0 == 0.0:
                    x1 = 0.05
                else:
                    x1 = 1.05 * x0
                self.axes_values[i] = np.append(self.axes_values[i], x1)
                coeff = np.copy(self.coeff)
                self.coeff = np.repeat(self.coeff, 2, axis=i)

    @staticmethod
    def get_indices(axes_names, axes_values, point_data):
        y = [0] * len(point_data)
        for name in point_data:
            i = np.flatnonzero(axes_names == name)[0]
            iaxis = axes_values[i]
            value = point_data[name]
            diff = np.absolute(axes_values[i] - value)
            j = np.argmin(diff)
            # print(name,i,j)
            y[i] = j
        return tuple(y)

    @staticmethod
    def extract_axes(h5):
        data = h5["incident"]["neutron"]
        dim_names = list()
        libs = list()
        ncoeff = 0
        nvec = 0
        for i in data.keys():
            if i != "TransitionStructure":
                libs.append(data[i])
                ncoeff = np.shape(data[i]["matrix"])[1]
                nvec = np.shape(data[i]["loss_xs"])[1]
                labels = data[i]["tags"]["continuous"]
                for x in labels:
                    dim_names.append(x)
        dim_names = list(np.unique(np.asarray(dim_names)))
        # print(dim_names)
        # print(libs)

        # create 1d dimensions array
        n = len(libs)
        dims = dict()
        for name in dim_names:
            dims[name] = np.zeros(n)
        times = list()
        D = 6
        for i in range(n):
            for name in libs[i]["tags"]["continuous"]:
                value = libs[i]["tags"]["continuous"][name]
                dims[name][i] = value[()].round(decimals=D)
            times.append(np.asarray(libs[i]["burnups"]).round(decimals=D))
        # print(dims)
        # print(times)

        # determine values in each dimension and add time
        axes_names = list(dim_names)
        ndims = len(dim_names) + 1
        axes_values = [0] * ndims
        i = 0
        for name in dims:
            axes_values[i] = np.unique(dims[name].round(decimals=D))
            i += 1
        axes_names.append("times")
        axes_values[i] = times[0]

        # print('axes:',axes)
        # print('axes_names',axes_names)

        # determine the shape/size of each dimension
        axes_shape = list(axes_values)
        for i in range(ndims):
            axes_shape[i] = len(axes_values[i])

        # convert names and shapes to np array before leaving
        axes_names = np.asarray(axes_names)
        axes_shape = np.asarray(axes_shape)
        return axes_names, axes_values, axes_shape, ncoeff, nvec
