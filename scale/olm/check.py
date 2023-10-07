"""
Module for checking classes.
"""
__all__ = ["sequencer", "GridGradient", "LowOrderConsistency"]

import numpy as np
from tqdm import tqdm, tqdm_notebook
import scale.olm.core as core
import scale.olm.run as run
import json
from pathlib import Path
import copy
import os
import scale.olm.internal as internal
from typing import List, Union, Dict, Literal


class CheckInfo:
    def __init__(self):
        self.test_pass = False


Model = Dict[str, any]
Env = Dict[str, any]

# -----------------------------------------------------------------------------------------

_TYPE_SEQUENCER = "scale.olm.check:sequencer"


def _schema_sequencer(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_SEQUENCER, with_state=with_state)
    return _schema


def _test_args_sequencer(with_state: bool = False):
    return {
        "_type": _TYPE_SEQUENCER,
        "sequence": [
            {"eps0": 0.0001, "_type": "scale.olm.check:GridGradient"},
            {
                "_type": "scale.olm.check:LowOrderConsistency",
                "name": "loc",
                "template": "model/origami/system-uox.jt.inp",
                "target_q1": 0.70,
                "target_q2": 0.95,
                "eps0": 1e-12,
                "epsa": 1e-6,
                "epsr": 1e-3,
                "nuclide_compare": ["0092235", "0094239"],
            },
        ],
    }


def sequencer(
    sequence: List[dict],
    _model: Model,
    _env: Env,
    dry_run: bool = False,
    _type: Literal[_TYPE_SEQUENCER] = None,
):
    """Run a sequence of checks.

    Args:
        sequence: List of checks to run by name.

        _model: Reference model data

        _env: Environment data.

    """
    output = []
    if dry_run:
        return {"test_pass": False, "output": output}

    test_pass = True
    try:
        # Process all the input.
        run_list = []
        i = 0
        for s in sequence:
            # Set the full name.
            t = s["_type"]
            if t.find(":") == -1:
                t = "scale.olm.check:" + t
            s["_type"] = t

            internal.logger.info("Checking options for", type=t, index=i)
            i += 1

            # Initialize the class.
            this_class = internal._fn_redirect(**s, _env=_env, _model=_model)
            run_list.append(this_class)

        # Read the archive.
        work_dir = Path(_env["work_dir"])
        arpdata_txt = work_dir / "arpdata.txt"
        name = _model["name"]
        if arpdata_txt.exists():
            archive = core.ReactorLibrary(arpdata_txt, name)
        else:
            archive = core.ReactorLibrary(Path(f"{name}.arc.h5"))

        # Execute in sequence.
        i = 0
        for r in run_list:
            internal.logger.info("Running checking sequence={}".format(i))

            info = r.run(archive)
            output.append(info.__dict__)
            i += 1

            if not info.test_pass:
                test_pass = False

        internal.logger.info(
            "Finished without exception test_pass={}".format(test_pass)
        )

    except ValueError as ve:
        internal.logger.error(str(ve))

    return {"test_pass": test_pass, "sequence": output}


# -----------------------------------------------------------------------------------------

_TYPE_GRIDGRADIENT = "scale.olm.check:GridGradient"


def _schema_GridGradient(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_GRIDGRADIENT, with_state=with_state)
    return _schema


def _test_args_GridGradient(with_state: bool = False):
    args = {"_type": _TYPE_GRIDGRADIENT}
    args.update(GridGradient.default_params())
    return args


class GridGradient:
    """Compute the grid gradients

    Computes the absolute and relative gradients of the reaction coefficient data
    in each dimension at each point and collects them into a data structure.

    The fraction of relative gradients which fall below the specified limit :code:`epsr`
    is the first quality score, :code:`q1=1-fr` where :code:`fr` is the failed fraction.
    The test passes quality check 1 if the :code:`q1<=target_q1`.

    Most often, we care less about relative differences when the absolute values are
    very small, e.g. a 10% difference in a 1e-12 barn cross section is not as big
    a deal as a 1% difference in a 100 barn cross section. Quality score :code:`q2`
    takes this into account by considering the fraction of points which fail the
    pure relative test, :code:`q1`, and those that fail a combined test where the
    relative gradient must exceed :code:`epsr` and the absolute gradient must exceed
    :code:`epsa`. The failed fraction is :code:`fa` and the combined score for
    :code:`q2=1-0.9*fa-0.1*fr`. In this way, one cannot get a perfect 1.0 for either
    score if there are any failures in a relative sense, but the second score penalizes
    them less. The second test passes if :code:`q2<=target_q2`.

    Args:
        eprs: The limit for the relative gradient.
        epsa: The limit for the absolute gradient.
        target_q1: The target for the q1 (relative only) score.
        target_g2: The target for the q2 (weighted relative and absolute) score.
        eps0: The minimum gradient to care about.

    """

    @staticmethod
    def describe_params():
        return {
            "eps0": "minimum value",
            "epsa": "absolute epsilon",
            "epsr": "relative epsilon",
            "target_q1": "target for quality score 1",
            "target_q2": "target for quality score 2",
        }

    @staticmethod
    def default_params():
        c = GridGradient()
        return {
            "eps0": c.eps0,
            "epsa": c.epsa,
            "epsr": c.epsr,
            "target_q2": c.target_q2,
            "target_q1": c.target_q1,
        }

    def __init__(
        self,
        _model: dict = None,
        _env: dict = {},
        eps0: float = 1e-20,
        epsa: float = 1e-1,
        epsr: float = 1e-1,
        target_q1: float = 0.5,
        target_q2: float = 0.7,
        _type: Literal[_TYPE_GRIDGRADIENT] = None,
    ):
        self.eps0 = eps0
        self.epsa = epsa
        self.epsr = epsr
        self.target_q1 = target_q1
        self.target_q2 = target_q2
        self.nprocs = _env.get("nprocs", 3)

    def run(self, archive):
        """Run the calculation and return post-processed results"""

        internal.logger.info(
            "Running "
            + self.__class__.__name__
            + " check with params={}".format(json.dumps(self.__dict__))
        )
        self.__calc(archive)

        # After calc the self.ahist, rhist, khist, and rel_axes variables are ready to
        # compute metrics.
        info = self.info()
        internal.logger.info(
            "Completed "
            + self.__class__.__name__
            + " with q1={:.2f} and q2={:.2f}".format(info.q1, info.q2)
        )

        return info

    def info(self):
        """Recalculate and return the score information."""
        info = CheckInfo()
        info.name = self.__class__.__name__
        info.eps0 = self.eps0
        info.epsa = self.epsa
        info.epsr = self.epsr
        info.target_q1 = self.target_q1
        info.target_q2 = self.target_q2
        info.wa = int(
            np.logical_and((self.ahist > self.epsa), (self.rhist > self.epsr)).sum()
        )
        info.wr = int((self.rhist > self.epsr).sum())
        info.m = int(len(self.ahist))
        info.fr = float(info.wr) / info.m
        info.q1 = 1.0 - info.fr
        info.fa = float(info.wa) / info.m
        info.q2 = 1.0 - 0.9 * info.fa - 0.1 * info.fr

        info.test_pass_q1 = info.q1 >= info.target_q1
        info.test_pass_q2 = info.q2 >= info.target_q2
        info.test_pass = info.test_pass_q1 and info.test_pass_q2

        return info

    def __calc(self, archive):
        """Drives the set up for the kernel with archive as input"""

        self.rel_axes = list()
        for x_list in archive.axes_values:
            dx = x_list[-1] - x_list[0]
            x0 = x_list[0]
            z = list()
            for x in x_list:
                z.append((x - x0) / dx)
            self.rel_axes.append(z)
        internal.logger.info("Finished computing relative values on axes")

        self.yreshape = np.moveaxis(archive.coeff, [-1], [0])
        internal.logger.info("Finished reshaping coefficients")

        internal.logger.info("Computing grid gradients ...")
        self.ahist, self.rhist, self.khist = GridGradient.__kernel(
            self.rel_axes, self.yreshape, self.eps0
        )
        internal.logger.info("Finished computing grid gradients")

    @staticmethod
    def __kernel(rel_axes, yreshape, eps0):
        """Lowest level kernel for the calculation"""

        # Number of dimensions.
        n = len(rel_axes)

        # Number of coefficients.
        ncoeff = np.shape(yreshape)[0]

        # Initialize histogram variables.
        rhist = np.zeros(n * n * ncoeff)
        ahist = np.zeros(n * n * ncoeff)
        khist = np.zeros(n * n * ncoeff)

        # For each coefficient in the transition matrix.
        for k in tqdm(range(ncoeff)):
            # Get just the grid of values for this coefficient.
            y = yreshape[k, ...]

            # Compute the maximum value at any point in the grid.
            max_y = np.amax(y)
            if max_y <= 0:
                max_y = eps0

            # Compute the gradient dy/dx for all axes.
            yp = np.asarray(np.gradient(y, *rel_axes))

            # Iterate through every combination of two dimensions.
            # TODO: generalize to all dimensions
            for i in range(n):
                ypi = yp[i, ...]

                for j in range(n):
                    # Calculate the flat index.
                    iu = k * n * n + i * n + j

                    # Calculate the difference between gradients.
                    yda = np.absolute(np.diff(ypi, axis=j))
                    ahist[iu] = np.amax(yda)

                    # Calculate relative difference using max_y calculated earlier.
                    ydr = yda / max_y
                    rhist[iu] = np.amax(ydr)

                    # Remember the coefficient index of this particular gradient difference.
                    khist[iu] = k

        return ahist, rhist, khist


# -----------------------------------------------------------------------------------------

_TYPE_LOWORDERCONSISTENCY = "scale.olm.check:LowOrderConsistency"


def _schema_LowOrderConsistency(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_LOWORDERCONSISTENCY, with_state=with_state)
    return _schema


def _test_args_LowOrderConsistency(with_state: bool = False):
    args = {"_type": _TYPE_LOWORDERCONSISTENCY}
    args.update(LowOrderConsistency.default_params())
    return args


class LowOrderConsistency:
    """Check that we are consistent with the original calculation.

    The ORIGEN library approach can be viewed as a high-order/low-order methodology
    where the ORIGEN library interpolation represents a low-order method which
    should agree with the high-order method.

    This check assumes that we already have high-order (e.g. TRITON) nuclide
    inventory results available. We use each of the libraries in the interpolation
    space in a new low-order (ORIGAMI) calculation. Consistent inputs are automatically
    constructed from available data. We then compare all nuclide inventory differences
    in the same way as for the :obj:`GridGradient` method, instead of relative and
    absolute gradients, we have relative and absolute differences in nuclide inventory.

    A number of plots are produced as side effects, referenced in the dictionary
    returned from the run() method.

    Args:
        name: Name of the test.
        template: Template file to use for the low-order calculation.
        nuclide_compare: List of nuclide identifiers for the detailed error plots.
        eprs: The limit for the relative gradient.
        epsa: The limit for the absolute gradient.
        target_q1: The target for the q1 (relative only) score.
        target_g2: The target for the q2 (weighted relative and absolute) score.
        eps0: The minimum gradient to care about.

    """

    @staticmethod
    def describe_params():
        return {
            "eps0": "minimum value",
            "epsa": "absolute epsilon",
            "epsr": "relative epsilon",
            "target_q1": "target for quality score 1",
            "target_q2": "target for quality score 2",
            "nuclide_compare": "plot me",
            "template": "template file name",
            "name": "name for test",
        }

    @staticmethod
    def default_params():
        import inspect

        # Use inspect to get required arguments.
        defaults = {}
        fn = internal._get_function_handle(_TYPE_LOWORDERCONSISTENCY)
        for k, v in inspect.signature(fn).parameters.items():
            if k.startswith("_"):
                continue
            defaults[k] = v.default
        return defaults

    def __init__(
        self,
        name: str = "",
        template: str = "",
        eps0: float = 1e-12,
        epsa: float = 1e-6,
        epsr: float = 1e-3,
        target_q1: float = 0.9,
        target_q2: float = 0.95,
        nuclide_compare: List[str] = ["0092235", "0094239"],
        _model: Model = None,
        _env: Env = None,
        _type: Literal[_TYPE_LOWORDERCONSISTENCY] = None,
        _dry_run: bool = False,
    ):
        self._env = _env
        self._model = _model
        self.name = name
        self.nuclide_compare = nuclide_compare
        self.eps0 = eps0
        self.epsa = epsa
        self.epsr = epsr
        self.target_q1 = target_q1
        self.target_q2 = target_q2

        if _dry_run:
            return

        if _env == None:
            dir = Path.cwd()
        else:
            dir = Path(_env["config_file"]).parent

        tm = core.TemplateManager([dir])

        self.template_path = tm.path(template)
        internal.logger.info(
            "check " + __class__.__name__, template_file=self.template_path
        )

        self.work_path = Path(_env["work_dir"])
        self.check_path = self.work_path / "check" / name
        self.check_dir = self.check_path.relative_to(self.work_path)

    @staticmethod
    def make_diff_plot(identifier, image, time, min_diff, max_diff, max_diff0, perms):
        """Make the difference plot."""
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 18})
        plt.figure()
        color = core.NuclideInventory._nuclide_color(identifier)
        plt.fill_between(
            np.asarray(time) / 86400.0,
            100 * np.asarray(min_diff),
            100 * np.asarray(max_diff),
            alpha=0.3,
            color=color,
        )

        for perm in perms:
            plt.plot(
                np.asarray(time) / 86400.0,
                100 * np.asarray(perm["(lo-hi)/max(|hi|)"]),
                "k-",
                alpha=0.4,
            )

        plt.xlabel("time (days)")
        plt.ylabel("lo/hi-1 (%)")
        plt.legend(["{} (max error: {:.2f} %)".format(identifier, 100 * max_diff0)])
        plt.savefig(image, bbox_inches="tight")

    def info(self):
        """Recalculate test statistics."""
        import matplotlib.pyplot as plt
        import sys

        # set number of permutations, timesteps, and nuclides for error array
        info = CheckInfo()
        info.name = self.__class__.__name__

        info.eps0 = self.eps0
        info.epsa = self.epsa
        info.epsr = self.epsr
        info.target_q1 = self.target_q1
        info.target_q2 = self.target_q2
        if not self.run_success:
            info.test_pass = False
            return info

        # Create a base comparison data structure to repeat for every permutation.
        internal.logger.info("Setting up detailed comparison structures...")
        info.nuclide_compare = dict()
        ntime = len(self.time_list)
        for nuclide in self.nuclide_compare:
            i = self.names.index(nuclide)
            internal.logger.info(
                f"Found nuclide={nuclide} at index {i} for detailed comparison"
            )
            info.nuclide_compare[nuclide] = {
                "nuclide_index": i,
                "nuclide": nuclide,
                "time": self.time_list,
                "max_diff": [-sys.float_info.max] * ntime,
                "min_diff": [sys.float_info.max] * ntime,
                "perms": [],
                "image": "",
            }

        self.ahist = np.array(self.lo_list)
        self.rhist = np.array(self.lo_list)
        self.hi = np.array(self.hi_list)
        self.lo = np.array(self.lo_list)

        # For each permutation.
        internal.logger.info("Calculating all comparison histogram data...")
        for k in range(len(self.lo_list)):
            # For each time.
            for j in range(len(self.lo_list[k])):
                osum = self.lo_list[k][j].sum()
                tsum = self.hi_list[k][j].sum()
                oden = self.lo_list[k][j] / osum
                tden = self.hi_list[k][j] / tsum
                self.lo[k, j, :] = oden
                self.hi[k, j, :] = tden
                self.ahist[k, j, :] = np.absolute(oden - tden)
                self.rhist[k, j, :] = np.absolute(
                    (oden + self.eps0) / (tden + self.eps0) - 1.0
                )

        # Extract each nuclide time series.
        internal.logger.info("Calculating nuclide-wise comparisons...")

        for n in info.nuclide_compare:
            i_nuclide = info.nuclide_compare[n]["nuclide_index"]
            for k in range(len(self.lo_list)):
                lo = self.lo[k, :, i_nuclide]
                hi = self.hi[k, :, i_nuclide]
                err = (lo - hi) / (self.eps0 + np.amax(np.absolute(hi)))
                info.nuclide_compare[n]["perms"].append(
                    {
                        "hi_ii_json": str(
                            self.ii_json_list[k][0].relative_to(self.work_path)
                        ),
                        "lo_ii_json": str(
                            self.ii_json_list[k][1].relative_to(self.work_path)
                        ),
                        "point_index": k,
                        "lo": list(lo),
                        "hi": list(hi),
                        "(lo-hi)/max(|hi|)": list(err),
                    }
                )

        # Get maximum and min error across all permutations.
        internal.logger.info("Calculating max/min across permutations...")
        for n, d in info.nuclide_compare.items():
            i_nuclide = d["nuclide_index"]
            for k in range(len(self.lo_list)):
                err = d["perms"][k]["(lo-hi)/max(|hi|)"]
                for j in range(len(self.time_list)):
                    d["max_diff"][j] = np.amax([err[j], d["max_diff"][j]])
                    d["min_diff"][j] = np.amin([err[j], d["min_diff"][j]])

            d["max_diff0"] = np.amax(
                [np.absolute(d["max_diff"]), np.absolute(d["min_diff"])]
            )
            image = self.check_path / (n + "-diff.png")
            internal.logger.info(
                "creating nuclide diff", image=str(image.relative_to(self.work_path))
            )
            info.nuclide_compare[n]["image"] = str(image)

            label = core.NuclideInventory._nice_label0(self.composition_manager, n)
            LowOrderConsistency.make_diff_plot(
                label,
                image,
                d["time"],
                d["min_diff"],
                d["max_diff"],
                d["max_diff0"],
                d["perms"],
            )

        self.ahist = np.ndarray.flatten(self.ahist)
        self.rhist = np.ndarray.flatten(self.rhist)
        hist_image = self.check_path / "hist.png"
        internal.logger.info(
            "creating histogram ", image=str(hist_image.relative_to(self.work_path))
        )
        core.RelAbsHistogram.plot_hist(
            self,
            hist_image,
            xlabel=r"$\log_{10} |hi/lo-1|$",
            ylabel=r"$\log_{10} |hi-lo|$",
        )
        info.hist_image = str(hist_image)

        info.wa = int(
            np.logical_and((self.ahist > self.epsa), (self.rhist > self.epsr)).sum()
        )
        info.wr = int((self.rhist > self.epsr).sum())
        info.m = int(len(self.ahist))
        info.fr = float(info.wr) / info.m
        info.q1 = 1.0 - info.fr
        info.fa = float(info.wa) / info.m
        info.q2 = 1.0 - 0.9 * info.fa - 0.1 * info.fr
        info.test_pass_q1 = info.q1 >= info.target_q1
        info.test_pass_q2 = info.q2 >= info.target_q2
        info.test_pass = info.test_pass_q1 and info.test_pass_q2
        # Other stats.
        info.mean_abs_diff = np.mean(self.ahist)
        info.mean_rel_diff = np.mean(self.rhist)
        info.std_abs_diff = np.std(self.ahist)
        info.std_rel_diff = np.std(self.rhist)

        return info

    def __run_lo_fidelity(self, do_run):
        """Run the LOWER fidelity calculation which should be consistent as possible with
        the already-complete higher order calculation."""

        # Load the template file.
        with open(self.template_path, "r") as f:
            template_text = f.read()

        # Load the assemble data.
        assemble_json = self.work_path / "assemble.olm.json"
        with open(assemble_json, "r") as f:
            assemble_d = json.load(f)

        # For each point in space.
        ii_json_list = list()
        f71_list = list()
        input_list = list()
        for point in assemble_d["points"]:
            # Create the check input path.
            lib = Path(point["files"]["lib"])
            base = lib.stem
            check_input = self.check_path / base / (base + ".inp")

            # Save the list.
            hi_ii_json = self.work_path / point["files"]["ii_json"]
            lo_ii_json = check_input.with_suffix(".ii.json")
            f71_list.append(check_input.with_suffix(".f71"))
            ii_json_list.append((hi_ii_json, lo_ii_json))

            # Create the directory.
            check_input.parent.mkdir(parents=True, exist_ok=True)

            # Populate data.
            check_data = {
                **point,
                "name": self.name,
                "_": {"env": self._env, "model": self._model},
            }

            # Write out data file.
            check_data_file = check_input.parent / "data.olm.json"
            with open(check_data_file, "w") as f:
                f.write(json.dumps(check_data, indent=4))
            internal.logger.debug(
                "Writing LowOrderConsistency check", data_file=check_data_file
            )

            # Fill the template.
            filled_text = core.TemplateManager.expand_text(template_text, check_data)

            # Write the check input file.
            internal.logger.debug(
                "Writing LowOrderConsistency check", input_file=check_input
            )
            input_list.append(str(check_input.relative_to(self.check_path)))
            with open(check_input, "w") as f:
                f.write(filled_text)

        # Use the makefile execution strategy for now.
        runinfo = internal._execute_makefile(
            dry_run=not do_run,
            _env=self._env,
            base_path=self.check_path,
            input_list=input_list,
        )

        # Actually generate the ii.json for the low fidelity calcs we just ran.
        if do_run:
            for f71 in f71_list:
                lo = internal.run_command(
                    f"{self._env['obiwan']} view -format=ii.json {f71} -cases='[{self.lo_case}]'",
                    echo=False,
                )
                lo_ii_json = f71.with_suffix(".ii.json")
                with open(lo_ii_json, "w") as f:
                    f.write(lo)

        return ii_json_list

    def __load_ii_json(self, ii_json_list):
        """Load the ii.json data that exists on disk for HIGH and LOWER fidelity into memory."""
        # We want nuclide data from one of the ii.json files.
        self.composition_manager = None

        # Convert the f71 to ii.json and extract the relevant information into memory.
        self.hi_list = list()
        self.lo_list = list()
        for hi_ii_json, lo_ii_json in ii_json_list:
            internal.logger.debug(f"loading HI {hi_ii_json}")
            # Load the json data into HIGH fidelity and LOWER fidelity data structures.
            # Note there's a little duplicate code here, but probably not worth refactoring.
            with open(hi_ii_json, "r") as f:
                jt = json.load(f)
                case = jt["responses"]["system"]

                # Just load once for the first available.
                if self.composition_manager == None:
                    self.composition_manager = core.CompositionManager(
                        jt["data"]["nuclides"]
                    )

                hi = np.array(case["amount"])
                hi_vector = case["nuclideVectorHash"]
                self.hi_list.append(hi)
                self.names = jt["definitions"]["nuclideVectors"][hi_vector]
                self.time_list = case["time"]

            internal.logger.debug(f"loading LO {lo_ii_json}")
            with open(lo_ii_json, "r") as f:
                jo = json.load(f)
                case = jo["responses"][f"case({self.lo_case})"]
                lo = np.array(case["amount"])
                lo_time = case["time"]
                self.lo_list.append(lo)

                # Check consistency.
                if not np.array_equal(lo_time, self.time_list):
                    raise ValueError(
                        f"HIGH fidelity list of times={self.time_list} is inconsistent with LOWER fidelity list of times {lo_time}"
                    )
                lo_vector = case["nuclideVectorHash"]
                if not lo_vector == hi_vector:
                    raise ValueError(
                        f"HIGH fidelity nuclide vector hash {hi_vector} is not the same as LOWER fidelity vector hash {lo_vector}, meaning the two nuclide sets are somehow inconsistent, which should not be possible."
                    )

    def run(self, archive):
        """Run a consistent set of LOWER fidelity calculations which also produce an
        f71--typically ORIGAMI."""

        # TODO: Allow input to change this or other smart way to determine if the data
        # does not need to be regenerated. Here, this is just for development iterations
        # to disable long SCALE runs while trying to debug checking.
        do_run = os.environ.get("SCALE_OLM_DO_RUN", "True") in ["True"]
        if not do_run:
            internal.logger.warning(
                "Runs suppressed by environment variable SCALE_OLM_DO_RUN!"
            )

        # Set the case identifiers for the high and low problems.
        self.hi_case = -2
        self.lo_case = 1

        try:
            self.ii_json_list = self.__run_lo_fidelity(do_run)
            self.__load_ii_json(self.ii_json_list)
            self.run_success = True

        except ValueError as ve:
            self.run_success = False
            internal.logger.error(str(ve))

        return self.info()
