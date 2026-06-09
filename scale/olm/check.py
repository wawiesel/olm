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
                "metric": "grams_per_initial_hm",
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
        failed_checks = []
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

        # Read the reactor_library.
        work_dir = Path(_env["work_dir"])
        arpdata_txt = work_dir / "arpdata.txt"
        name = _model["name"]
        if arpdata_txt.exists():
            reactor_library = core.ReactorLibrary(arpdata_txt, name)
        else:
            reactor_library = core.ReactorLibrary(Path(f"{name}.arc.h5"))

        # Execute in sequence.
        i = 0
        for r in run_list:
            internal.logger.info("Running checking sequence={}".format(i))

            info = r.run(reactor_library)
            output.append(info.__dict__)
            i += 1

            if not info.test_pass:
                test_pass = False
                failed_checks.append(getattr(info, "name", r.__class__.__name__))

        if test_pass:
            internal.logger.info(
                "Finished check sequence", test_pass=True, checks=len(output)
            )
        else:
            internal.logger.warning(
                "Finished check sequence with failing checks",
                test_pass=False,
                checks=len(output),
                failed_checks=failed_checks,
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

    def run(self, reactor_library):
        """Run the calculation and return post-processed results"""

        internal.logger.info(
            "Running "
            + self.__class__.__name__
            + " check with params={}".format(json.dumps(self.__dict__))
        )
        self.__calc(reactor_library)

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

    def __calc(self, reactor_library):
        """Drives the set up for the kernel with reactor_library as input"""

        self.rel_axes = list()
        for x_list in reactor_library.axes_values:
            dx = x_list[-1] - x_list[0]
            x0 = x_list[0]
            z = list()
            for x in x_list:
                z.append((x - x0) / dx)
            self.rel_axes.append(z)
        internal.logger.info("Finished computing relative values on axes")

        self.yreshape = np.moveaxis(reactor_library.coeff, [-1], [0])
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
        nd = np.sum([len(a)-1 for a in rel_axes])
        rhist = np.zeros(n * nd * ncoeff)
        ahist = np.zeros(n * nd * ncoeff)
        khist = np.zeros(n * nd * ncoeff)

        # For each coefficient in the transition matrix.
        for k in tqdm(range(ncoeff)):
            # Get just the grid of values for this coefficient.
            y = yreshape[k, ...]

            # Compute the min/max magnitude at any point in the grid.
            max_y = np.amax(np.absolute(y))
            if max_y <= 0:
                max_y = eps0
            min_y = np.amin(np.absolute(y))
            if min_y <= 0:
                min_y = eps0
            mid_y = 0.5*(min_y+max_y)

            for i in range(n):
                # First and second derivatives.
                yp = np.asarray( np.gradient(y, rel_axes[i], axis=i) )
                ypp = np.asarray( np.gradient(yp, rel_axes[i], axis=i) )

                # Move the axis `i` to the last dimension so we can use ... below.
                # Go ahead and take absolute value to simplify too.
                ypp_abs = np.absolute(np.moveaxis(ypp, i, -1))

                # Evaluate Rolle's theorem for each interval along axis=i
                # March through each interval explicitly
                dx_list = np.diff(rel_axes[i])
                for j in range(len(dx_list)):
                    dx = dx_list[j]

                    # Take max of left and right ypp for this interval.
                    max_ypp = max(np.amax(ypp_abs[..., j]), np.amax(ypp_abs[..., j+1]) )

                    # Compute error for this interval
                    error = (dx ** 2) * max_ypp / 8.0
                    #print(i,j,dx,max_ypp,error)

                    # Flat index.
                    iu = k * n * nd + i * nd + j

                    # Update absolute and relative versions.
                    ahist[iu] = error
                    rhist[iu] = error/(mid_y*mid_y)

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
        metric: Primary inventory metric to use for quality scores.
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
            "metric": "primary inventory metric",
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
        metric: Literal[
            "grams_per_initial_hm", "atom_fraction"
        ] = "grams_per_initial_hm",
        eps0: float = 1e-12,
        epsa: float = 1e-6,
        epsr: float = 1e-3,
        target_q1: float = 0.9,
        target_q2: float = 0.95,
        nuclide_compare: List[str] = ["u235", "pu239"],
        _model: Model = None,
        _env: Env = None,
        _type: Literal[_TYPE_LOWORDERCONSISTENCY] = None,
        _dry_run: bool = False,
    ):
        self._env = _env
        self._model = _model
        self.name = name
        self.nuclide_compare = nuclide_compare
        self.metric = metric
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
    def make_scaled_difference_plot(
        identifier,
        image,
        time,
        min_scaled_difference,
        max_scaled_difference,
        max_abs_scaled_difference,
        perms,
    ):
        """Make the scaled-difference plot."""
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 18})
        plt.figure()
        color = core.NuclideInventory._nuclide_color(identifier)
        plt.fill_between(
            np.asarray(time) / 86400.0,
            100 * np.asarray(min_scaled_difference),
            100 * np.asarray(max_scaled_difference),
            alpha=0.3,
            color=color,
        )

        for perm in perms:
            plt.plot(
                np.asarray(time) / 86400.0,
                100 * np.asarray(perm["scaled_difference"]),
                "k-",
                alpha=0.4,
            )

        plt.xlabel("time (days)")
        plt.ylabel(LowOrderConsistency._scaled_difference_ylabel())
        plt.legend(
            [
                "{} (max |scaled_difference|: {:.2f} %)".format(
                    identifier, 100 * max_abs_scaled_difference
                )
            ]
        )
        plt.savefig(image, bbox_inches="tight")

    @staticmethod
    def _metric_units(metric):
        return {
            "grams_per_initial_hm": "g/gIHM",
            "atom_fraction": "atom fraction",
        }[metric]

    def _amounts_to_grams_per_initial_hm(self, amounts):
        """Convert inventory amounts to grams per gram initial heavy metal.

        Formula:
            g/gIHM[p, t, n] =
                amounts[p, t, n] [mol] * mass[n] [g/mol]
                / (initialhm[p] [MTIHM] * 1.0e6 [g/MT])

        The nuclide masses come from CompositionManager.mass().
        Axes are point, time, and nuclide. The initial heavy metal is per point.
        """
        cm = self.composition_manager
        masses = np.array([cm.mass(name) for name in self.names])
        initialhm_g = (
            np.asarray(self.initialhm_list, dtype=float)[:, None, None] * 1.0e6
        )
        nuclide_g = (
            np.asarray(amounts, dtype=float)
            * np.asarray(masses, dtype=float)[None, None, :]
        )
        return nuclide_g / initialhm_g

    def _amounts_to_atom_fraction(self, amounts):
        amounts = np.asarray(amounts, dtype=float)
        if amounts.ndim != 3:
            raise ValueError(
                "LowOrderConsistency inventory amounts must have shape "
                "(point, time, nuclide)."
            )
        totals = amounts.sum(axis=2)
        if np.any(totals == 0.0):
            raise ValueError("Cannot calculate atom fractions with zero total atoms.")
        return amounts / totals[:, :, None]

    @staticmethod
    def _difference_arrays(lo, hi, eps0):
        """Return absolute and relative pointwise inventory differences.

        ahist = |lo - hi|
        rhist = |(lo + eps0) / (hi + eps0) - 1|
        """
        ahist = np.absolute(lo - hi)
        rhist = np.absolute((lo + eps0) / (hi + eps0) - 1.0)
        return ahist, rhist

    @staticmethod
    def _scaled_difference(lo, hi):
        """Return scaled_difference = (lo - hi) / max(hi) for one curve."""
        return (lo - hi) / np.amax(hi)

    @staticmethod
    def _scaled_difference_ylabel():
        return "(lo - hi) / max(hi) (%)"

    @staticmethod
    def _relative_difference_xlabel():
        return r"$\log_{10} |lo/hi-1|$"

    def _absolute_difference_ylabel(self):
        label = r"$\log_{10} |hi-lo|$"
        if self.metric == "grams_per_initial_hm":
            return label + " [g/gIHM]"
        return label

    @staticmethod
    def _require_initial_time(label, times):
        times = np.asarray(times, dtype=float)
        matches = np.where(np.isclose(times, 0.0, rtol=0.0, atol=1.0e-6))[0]
        if len(matches) != 1:
            raise ValueError(
                f"{label} list of times must include exactly one time=0.0 "
                f"entry; times={list(times)}"
            )

    @staticmethod
    def _matching_time_indices(reference_time, candidate_time):
        LowOrderConsistency._require_initial_time("HIGH order", reference_time)
        LowOrderConsistency._require_initial_time("LOW order", candidate_time)

        indices = []
        candidate_time = np.asarray(candidate_time, dtype=float)
        for time in np.asarray(reference_time, dtype=float):
            matches = np.where(
                np.isclose(candidate_time, time, rtol=0.0, atol=1.0e-6)
            )[0]
            if len(matches) != 1:
                raise ValueError(
                    f"HIGH order time={time} did not match exactly one "
                    "LOW order time."
                )
            indices.append(int(matches[0]))
        return indices

    def _metric_arrays(self):
        hi_amount = np.asarray(self.hi_list, dtype=float)
        lo_amount = np.asarray(self.lo_list, dtype=float)

        if self.metric == "atom_fraction":
            return (
                self._amounts_to_atom_fraction(lo_amount),
                self._amounts_to_atom_fraction(hi_amount),
            )

        return (
            self._amounts_to_grams_per_initial_hm(lo_amount),
            self._amounts_to_grams_per_initial_hm(hi_amount),
        )

    def info(self):
        """Recalculate test statistics."""
        import sys

        # set number of permutations, timesteps, and nuclides for error array
        info = CheckInfo()
        info.name = self.__class__.__name__

        info.eps0 = self.eps0
        info.epsa = self.epsa
        info.epsr = self.epsr
        info.target_q1 = self.target_q1
        info.target_q2 = self.target_q2
        info.metric = self.metric
        info.units = self._metric_units(self.metric)
        if not self.run_success:
            info.test_pass = False
            return info

        # Create a base comparison data structure to repeat for every permutation.
        internal.logger.info("Setting up detailed comparison structures...")
        info.nuclide_compare = dict()
        ntime = len(self.time_list)
        for nuclide in self.nuclide_compare:
            eam = self.composition_manager.eam(nuclide)
            izzzaaa = self.composition_manager.izzzaaa(nuclide)
            i = self.names.index(izzzaaa)
            internal.logger.info(
                f"Found nuclide={nuclide} at index {i} for detailed comparison"
            )
            info.nuclide_compare[eam] = {
                "nuclide_index": i,
                "nuclide": eam,
                "nuclide_izzzaaa": izzzaaa,
                "time": self.time_list,
                "max_scaled_difference": [-sys.float_info.max] * ntime,
                "min_scaled_difference": [sys.float_info.max] * ntime,
                "perms": [],
                "image": "",
            }

        internal.logger.info("Calculating all comparison histogram data...")
        self.lo, self.hi = self._metric_arrays()
        self.ahist, self.rhist = self._difference_arrays(self.lo, self.hi, self.eps0)

        # Extract each nuclide time series.
        internal.logger.info("Calculating nuclide-wise comparisons...")

        for n in info.nuclide_compare:
            i_nuclide = info.nuclide_compare[n]["nuclide_index"]
            for k in range(len(self.lo_list)):
                lo = self.lo[k, :, i_nuclide]
                hi = self.hi[k, :, i_nuclide]
                err = self._scaled_difference(lo, hi)
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
                        "scaled_difference": list(err),
                    }
                )

        # Get maximum and min error across all permutations.
        internal.logger.info("Calculating max/min across permutations...")
        for n, d in info.nuclide_compare.items():
            i_nuclide = d["nuclide_index"]
            for k in range(len(self.lo_list)):
                err = d["perms"][k]["scaled_difference"]
                for j in range(len(self.time_list)):
                    d["max_scaled_difference"][j] = np.amax(
                        [err[j], d["max_scaled_difference"][j]]
                    )
                    d["min_scaled_difference"][j] = np.amin(
                        [err[j], d["min_scaled_difference"][j]]
                    )

            d["max_abs_scaled_difference"] = np.amax(
                [
                    np.absolute(d["max_scaled_difference"]),
                    np.absolute(d["min_scaled_difference"]),
                ]
            )
            image = self.check_path / (n + "-scaled-difference.png")
            internal.logger.info(
                "creating nuclide scaled difference",
                image=str(image.relative_to(self.work_path)),
            )
            info.nuclide_compare[n]["image"] = str(image)

            label = core.NuclideInventory._nice_label0(self.composition_manager, n)
            LowOrderConsistency.make_scaled_difference_plot(
                label,
                image,
                d["time"],
                d["min_scaled_difference"],
                d["max_scaled_difference"],
                d["max_abs_scaled_difference"],
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
            xlabel=self._relative_difference_xlabel(),
            ylabel=self._absolute_difference_ylabel(),
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

    def __run_lo_order(self, do_run):
        """Run the LOW order calculation which should be consistent as possible with
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
        self.initialhm_list = list()
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
            try:
                initialhm = float(point["history"]["initialhm"])
            except KeyError as exc:
                raise ValueError(
                    "LowOrderConsistency requires history.initialhm "
                    f"for point={base}"
                ) from exc
            if initialhm <= 0.0:
                raise ValueError(
                    "LowOrderConsistency requires positive initial heavy metal "
                    f"for point={base}"
                )
            self.initialhm_list.append(initialhm)

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

        # Actually generate the ii.json for the LOW order calcs we just ran.
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
        """Load HIGH order and LOW order ii.json data from disk into memory."""
        # We want nuclide data from one of the ii.json files.
        self.composition_manager = None

        # Convert the f71 to ii.json and extract the relevant information into memory.
        self.hi_list = list()
        self.lo_list = list()
        self.time_list = None
        for hi_ii_json, lo_ii_json in ii_json_list:
            internal.logger.debug(f"loading HI {hi_ii_json}")
            # Load the json data into HIGH order and LOW order data structures.
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
                self.names = jt["definitions"]["nuclideVectors"][hi_vector]
                hi_time = case["time"]

            internal.logger.debug(f"loading LO {lo_ii_json}")
            with open(lo_ii_json, "r") as f:
                jo = json.load(f)
                case = jo["responses"][f"case({self.lo_case})"]
                lo = np.array(case["amount"])
                lo_time = case["time"]

                # Check consistency and align LOW order extra points to the HIGH grid.
                indices = self._matching_time_indices(hi_time, lo_time)
                lo = lo[indices, :]
                lo_vector = case["nuclideVectorHash"]
                if not lo_vector == hi_vector:
                    raise ValueError(
                        f"HIGH order nuclide vector hash {hi_vector} is not "
                        f"the same as LOW order vector hash {lo_vector}, "
                        "meaning the two nuclide sets are somehow inconsistent, "
                        "which should not be possible."
                    )
                if self.time_list is None:
                    self.time_list = hi_time
                elif not np.array_equal(hi_time, self.time_list):
                    raise ValueError(
                        f"HIGH order list of times={hi_time} is inconsistent "
                        f"with previous HIGH order list of times {self.time_list}"
                    )
                self.hi_list.append(hi)
                self.lo_list.append(lo)

    def run(self, reactor_library):
        """Run a consistent set of LOW order calculations which also produce an
        f71--typically ORIGAMI."""

        # TODO: The reactor_library is not explicitly used because it was already expanded
        # into the Low Order/ORIGAMI input file. We may need to force some kind of
        # consistency here.

        # TODO: Allow input to change this or other smart way to determine if the data
        # does not need to be regenerated. Here, this is just for development iterations
        # to disable long SCALE runs while trying to debug checking.
        do_run = os.environ.get("SCALE_OLM_DO_RUN", "True") in ["True"]
        if not do_run:
            internal.logger.warning(
                "Runs suppressed by environment variable SCALE_OLM_DO_RUN!"
            )

        # TODO: This needs to be more adaptive; for instance,
        # Polaris' basis material is not case -2 (can vary).
        # Set the case identifiers for the high and low problems.
        self.hi_case = -2
        self.lo_case = 1

        try:
            self.ii_json_list = self.__run_lo_order(do_run)
            self.__load_ii_json(self.ii_json_list)
            self.run_success = True

        except ValueError as ve:
            self.run_success = False
            internal.logger.error(str(ve))

        return self.info()
