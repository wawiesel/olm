import h5py
from tqdm import tqdm, tqdm_notebook
import numpy as np
import structlog
import json
from pathlib import Path
import os
import matplotlib.pyplot as plt
import sys
import copy
import subprocess
import shutil
from jinja2 import Template, StrictUndefined, exceptions
import re

logger = structlog.getLogger(__name__)


def get_runtime(output):
    with open(output, "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(" ")
            # t-depl finished. used 35.2481 seconds.
            if len(tokens) >= 5:
                if (
                    tokens[1] == "finished."
                    and tokens[2] == "used"
                    and tokens[4] == "seconds."
                ):
                    return float(tokens[3])
    return 0


def run_summary(build):
    rows = list()
    rows.append(["output", "runtime (s)"])
    work_dir = Path(build["work_dir"])
    for p in build["perms"]:
        output = p["files"]["output"]
        rows.append([output, get_runtime(work_dir / output)])
    return rows


def static_summary(params):
    rows = list()
    input_desc = params["input_desc"]
    for p in params:
        if p != "input_desc":
            v = params[p]
            d = ""
            if p in input_desc:
                d = input_desc[p]
            rows.append([p, v, d])
    return rows


def run_command(command_line, check_return_code=True):
    logger.info(f"running command:\n{command_line}")
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
        if "Error" in line:
            raise ValueError(line.strip())
        else:
            logger.info(line.rstrip())
        if not line:
            break
    if check_return_code and p.returncode != 0:
        msg = p.stderr.read().strip()
        if not msg == "":
            raise ValueError(str(msg))
    return text


def get_history_from_f71(obiwan, f71, caseid0):
    """
     Parse the history of the form as follows:
    pos         time        power         flux      fluence       energy    initialhm libpos   case   step DCGNAB
    (-)          (s)         (MW)    (n/cm2-s)      (n/cm2)        (MWd)      (MTIHM)    (-)    (-)    (-)    (-)
      1  0.00000e+00  4.00000e+01  8.11143e+14  0.00000e+00  0.00000e+00  1.00000e+00      1      1      0 DC----
      2  2.16000e+06  4.00000e+01  6.22529e+14  1.53582e+21  1.00000e+03  1.00000e+00      1      1     10 DC----
      3  2.16000e+07  4.00000e+01  4.26681e+14  8.78948e+21  1.00000e+04  1.00000e+00      2      1     10 DC----
      4  5.40000e+07  4.00000e+01  4.26566e+14  1.34274e+22  2.50000e+04  1.00000e+00      3      1     10 DC----
      5  1.08000e+08  4.00000e+01  4.31263e+14  2.30677e+22  5.00000e+04  1.00000e+00      4      1     10 DC----
      6  1.51200e+08  4.00000e+01  4.32303e+14  1.86058e+22  7.00000e+04  1.00000e+00      5      1     10 DC----
      7  1.94400e+08  4.00000e+01  4.33742e+14  1.86669e+22  9.00000e+04  1.00000e+00      6      1     10 DC----
      8  2.37600e+08  4.00000e+01  4.35733e+14  1.87415e+22  1.10000e+05  1.00000e+00      7      1     10 DC----
    """
    logger.info(f"extracting history from {f71}")
    text = run_command(f"{obiwan} view -format=info {f71}")
    burndata = list()
    initialhm0 = None
    last_days = 0.0
    for line in text.split("\n")[2:]:
        if len(line) < 60:
            break
        tokens = line.rstrip().split()
        caseid = tokens[8]
        if caseid0 == int(caseid):
            days = float(tokens[1]) / 86400.0
            if days == 0.0:
                initialhm0 = float(tokens[6])
            else:
                burndata.append({"power": float(tokens[2]), "burn": (days - last_days)})
            last_days = days
    return {"burndata": burndata, "initialhm": initialhm0}


class ArpInfo:
    def __init__(self):
        self.name = ""
        self.lib_map = None
        self.fuel_type = ""
        self.block = ""

    def clear_lib_map(self):
        self.lib_map = list()

        if self.fuel_type == "UOX":
            for ie in range(len(self.enrichment_list)):
                self.lib_map.append(list())
                for im in range(len(self.mod_dens_list)):
                    self.lib_map[ie].append("")

        elif self.fuel_type == "MOX":
            raise ValueError("mox not implemented")
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unkonwn (UOX/MOX)".format(self.fuel_type)
            )

    def init_block(self, name, block):
        """Initialize data from a single block of arpdata WITHOUT the ! line"""

        self.name = name
        self.block = block
        if self.name.startswith("mox_"):
            self.fuel_type = "MOX"
        elif self.name.startswith("act_"):
            self.fuel_type = "ACT"
        else:
            self.fuel_type = "UOX"

        tokens = self.block.split()
        if self.fuel_type == "UOX":
            ne = int(tokens[0])
            nm = int(tokens[1])
            nb = int(tokens[2])
            s = 3
            self.enrichment_list = [float(x) for x in tokens[s : s + ne]]
            s += ne
            self.mod_dens_list = [float(x) for x in tokens[s : s + nm]]
            s += nm
            self.clear_lib_map()
            for ie in range(len(self.enrichment_list)):
                for im in range(len(self.mod_dens_list)):
                    filename = tokens[s].replace("'", "").replace('"', "")
                    self.lib_map[ie][im] = filename
                    s += 1
            self.burnup_list = [float(x) for x in tokens[s : s + nb]]

        elif self.fuel_type == "MOX":
            np = int(tokens[0])
            nf = int(tokens[1])
            nd = int(tokens[2])
            nm = int(tokens[3])
            nb = int(tokens[4])
            s = 5
            self.pu_frac_list = [float(x) for x in tokens[s : s + np]]
            s += np
            self.fissile_pu_frac_list = [float(x) for x in tokens[s : s + nf]]
            s += nf
            s += 1  # Skip dummy entry
            self.mod_dens_list = [float(x) for x in tokens[s : s + nm]]
            s += nm
            nfile = np * nf * nm
            #### READ FILES

            self.burnup_list = [float(x) for x in tokens[s : s + nb]]
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unkonwn (UOX/MOX)".format(self.fuel_type)
            )

    @staticmethod
    def parse_arpdata(file):
        """Simple function to parse the blocks of arpdata.txt"""
        logger.debug(f"reading {file} ...")
        blocks = dict()
        with open(file, "r") as f:
            for line in f.readlines():
                if line.startswith("!"):
                    name = line.strip()[1:]
                    logger.debug(f"reading {name} ...")
                    blocks[name] = ""
                else:
                    blocks[name] += line
        return blocks

    def init_uox(self, name, file_list, enrichment_list, mod_dens_list):
        # Convert to interpolation space, assuming correct set up.
        self.name = name
        self.fuel_type = "UOX"
        self.enrichment_list = sorted(set(enrichment_list))
        self.mod_dens_list = sorted(set(mod_dens_list))
        self.burnup_list = []
        self.block = ""
        self.clear_lib_map()

        # Map flat list of file_list.
        for i in range(len(file_list)):
            e = enrichment_list[i]
            m = mod_dens_list[i]
            ie = self.enrichment_list.index(e)
            im = self.mod_dens_list.index(m)
            self.lib_map[ie][im] = file_list[i]

    def set_canonical_filenames(self, ext):
        # Keep track of filename counts so we are sure we don't create a duplicate.
        counts = set()
        if self.fuel_type == "UOX":
            # Fill with data.
            self.clear_lib_map()
            (nm, ne) = self.get_ndims()
            for ie in range(ne):
                e = self.enrichment_list[ie]
                for im in range(nm):
                    m = self.mod_dens_list[im]
                    filename = "{}_e{:04d}w{:04d}{}".format(
                        self.name, int(1000 * e), int(1000 * m), ext
                    )
                    self.lib_map[ie][im] = filename
                    if filename in counts:
                        raise ValueError(
                            f"canonical filename={filename} has already been used--most likely due to too small grid spacing!"
                        )
                    counts.add(filename)

        elif self.fuel_type == "MOX":
            raise ValueError("mox not implemented")
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unkonwn (UOX/MOX)".format(self.fuel_type)
            )

    def get_lib_by_index(self, i):
        if self.fuel_type == "UOX":
            (im, ie) = self.get_dim_by_index(i)
            return self.lib_map[ie][im]
        elif self.fuel_type == "MOX":
            raise ValueError("mox not implemented")
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unkonwn (UOX/MOX)".format(self.fuel_type)
            )

    def get_ndims(self):
        if self.fuel_type == "UOX":
            ne = len(self.enrichment_list)
            nm = len(self.mod_dens_list)
            return (nm, ne)
        elif self.fuel_type == "MOX":
            raise ValueError("mox not implemented")
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unkonwn (UOX/MOX)".format(self.fuel_type)
            )

    def num_libs(self):
        return np.prod(self.get_ndims())

    def get_dim_by_index(self, i):
        return np.unravel_index(i, self.get_ndims())

    def interptags_by_index(self, i):
        if self.fuel_type == "UOX":
            d = self.interpvars_by_index(i)
            y = ["{}={}".format(x, d[x]) for x in d]
            return ",".join(y)
        elif self.fuel_type == "MOX":
            raise ValueError("mox not implemented")
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unkonwn (UOX/MOX)".format(self.fuel_type)
            )

    def interpvars_by_index(self, i):
        if self.fuel_type == "UOX":
            (im, ie) = self.get_dim_by_index(i)
            return {
                "enrichment": self.enrichment_list[ie],
                "mod_dens": self.mod_dens_list[im],
            }
        elif self.fuel_type == "MOX":
            raise ValueError("mox not implemented")
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unkonwn (UOX/MOX)".format(self.fuel_type)
            )

    def get_arpdata(self):
        entry = ""
        if self.fuel_type == "UOX":
            ne = len(self.enrichment_list)
            nm = len(self.mod_dens_list)
            nb = len(self.burnup_list)
            entry += "{} {} {}\n".format(ne, nm, nb)
            entry += "\n".join([str(x) for x in self.enrichment_list]) + "\n"
            entry += "\n".join([str(x) for x in self.mod_dens_list]) + "\n"
            for ie in range(ne):
                for im in range(nm):
                    entry += "'{}'\n".format(self.lib_map[ie][im])
            entry += "\n".join([str(x) for x in self.burnup_list])
        elif self.fuel_type == "MOX":
            raise ValueError("mox not implemented")
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unkonwn (UOX/MOX)".format(self.fuel_type)
            )

        self.block = entry
        return "!{}\n{}".format(self.name, self.block)

    def create_temp_archive(self, arpdata_txt, temp_arc):
        h5arc = None
        n = 1
        arpdir = arpdata_txt.parent / "arplibs"
        if self.fuel_type == "UOX":
            (nm, ne) = self.get_ndims()
            for ie in range(ne):
                for im in range(nm):
                    lib = Path(self.lib_map[ie][im])
                    if not h5arc:
                        logger.info(f"initializing temporary archive {temp_arc}")
                        shutil.copyfile(arpdir / lib, temp_arc)
                        h5arc = h5py.File(temp_arc, "a")
                    else:
                        n += 1
                        logger.info(
                            f"adding library {lib} to temporary archive {temp_arc}"
                        )
                        h5arc["incident"]["neutron"][f"lib{n}"] = h5py.ExternalLink(
                            arpdir / lib, "/incident/neutron/lib1"
                        )

        elif arpinfo.fuel_type == "MOX":
            raise ValueError("mox not implemented")
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unkonwn (UOX/MOX)".format(self.fuel_type)
            )

        return h5arc


def expand_template(template_text, data):
    # Instance template.
    j2t = Template(template_text, undefined=StrictUndefined)

    # Catch specific types of error.
    try:
        return j2t.render(data)
    except exceptions.UndefinedError as ve:
        raise ValueError(
            "Undefined variable reported (most likely template has a variable that is undefined in the configuration file). Error from template expansion: "
            + str(ve)
        )


def parse_burnups_from_triton_output(output):
    """Parse the table that looks like this:

    Sub-Interval   Depletion   Sub-interval    Specific      Burn Length  Decay Length   Library Burnup
         No.       Interval     in interval  Power(MW/MTIHM)     (d)          (d)           (MWd/MTIHM)
    ----------------------------------------------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------
            0     ****Initial Bootstrap Calculation****                                      0.00000E+00
            1          1                1          40.000      25.000         0.000          5.00000e+02
            2          1                2          40.000     300.000         0.000          7.00000e+03
            3          1                3          40.000     300.000         0.000          1.90000e+04
            4          1                4          40.000     312.500         0.000          3.12500e+04
            5          1                5          40.000     312.500         0.000          4.37500e+04
            6          1                6          40.000     333.333         0.000          5.66667e+04
            7          1                7          40.000     333.333         0.000          7.00000e+04
            8          1                8          40.000     333.333         0.000          8.33333e+04
    ----------------------------------------------------------------------------------------------------

    """
    burnup_list = []
    with open(output, "r") as f:
        n = 0
        found = False
        for line in f.readlines():
            words = line.split()
            if words == [
                "Sub-Interval",
                "Depletion",
                "Sub-interval",
                "Specific",
                "Burn",
                "Length",
                "Decay",
                "Length",
                "Library",
                "Burnup",
            ]:
                found = True
            if found:
                n += 1
                if n > 4 and line.strip().startswith("-----"):
                    found = False
                elif n > 4:
                    bu = float(line.split()[-1])
                    burnup_list.append(bu)
    logger.info(
        "found burnup_list=[{}]".format(",".join([str(x) for x in burnup_list]))
    )
    return burnup_list


def update_model(model):
    """Update the model section with paths."""

    # Find SCALE and utils.
    scale_env_var = model["scale_env_var"]
    if not scale_env_var in os.environ:
        raise ValueError(
            f"Environment variable scale_env_var='{scale_env_var}' must be set!"
        )

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
            blocks = ArpInfo.parse_arpdata(q1)
            for n in blocks:
                if n in registry:
                    logger.warning(
                        "library name {} has already been registered at path={} ignoring same name found at {}".format(
                            n, registry[n].path, p
                        )
                    )
                else:
                    logger.info("found library name {} in {}!".format(n, q1))
                    arpinfo = ArpInfo()
                    arpinfo.init_block(n, blocks[n])
                    arpinfo.path = q1
                    arpinfo.arplibs_dir = r
                    registry[n] = arpinfo


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

    def __init__(self, file, name=""):
        logger.info("Loading archive file={}".format(file))

        self.file_name = file

        # Initialize in-memory data structure.
        if file.name == "arpdata.txt":
            blocks = ArpInfo.parse_arpdata(file)
            arpinfo = ArpInfo()
            arpinfo.init_block(name, blocks[name])
            temp_arc = file.with_suffix(".arc.h5")
            self.name = name
            self.h5 = arpinfo.create_temp_archive(file, temp_arc)
        else:
            self.h5 = h5py.File(file, "r")

        # Get important axis variables.
        (
            self.axes_names,
            self.axes_values,
            self.axes_shape,
            self.ncoeff,
            self.nvec,
        ) = Archive.extract_axes(self.h5)

        # Populate coefficient data.
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
