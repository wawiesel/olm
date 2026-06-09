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
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List, Union, Dict, Literal, Annotated, Optional


class CheckInfo:
    def __init__(self):
        self.test_pass = False


Model = Dict[str, any]
Env = Dict[str, any]


def _quality_summary_from_histograms(
    ahist,
    rhist,
    epsa,
    epsr,
    target_q1,
    target_q2,
):
    """Return q-score data for already calculated absolute/relative errors."""
    ahist = np.ndarray.flatten(np.asarray(ahist, dtype=float))
    rhist = np.ndarray.flatten(np.asarray(rhist, dtype=float))
    if len(ahist) != len(rhist):
        raise ValueError("Absolute and relative error histograms must be the same size.")
    if len(ahist) == 0:
        raise ValueError("Cannot calculate q-scores from empty error histograms.")

    wa = int(np.logical_and((ahist > epsa), (rhist > epsr)).sum())
    wr = int((rhist > epsr).sum())
    m = int(len(ahist))
    fr = float(wr) / m
    fa = float(wa) / m
    q1 = 1.0 - fr
    q2 = 1.0 - 0.9 * fa - 0.1 * fr
    return {
        "wa": wa,
        "wr": wr,
        "m": m,
        "fr": fr,
        "q1": q1,
        "target_q1": target_q1,
        "test_pass_q1": q1 >= target_q1,
        "fa": fa,
        "q2": q2,
        "target_q2": target_q2,
        "test_pass_q2": q2 >= target_q2,
        "test_pass": q1 >= target_q1 and q2 >= target_q2,
        "mean_abs_diff": float(np.mean(ahist)),
        "mean_rel_diff": float(np.mean(rhist)),
        "std_abs_diff": float(np.std(ahist)),
        "std_rel_diff": float(np.std(rhist)),
    }


class LowOrderConsistencyConvergence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nlib_start: Annotated[int, Field(gt=0)] = 1
    nlib_max: Annotated[int, Field(gt=0)] = 1
    nburn_start: Annotated[int, Field(gt=0)] = 1
    nburn_max: Annotated[int, Field(gt=0)] = 1
    q1_stop_criteria: Annotated[float, Field(ge=0.0)] = 0.0
    q2_stop_criteria: Annotated[float, Field(ge=0.0)] = 0.0

    @model_validator(mode="after")
    def _check_ranges(self):
        if self.nlib_max < self.nlib_start:
            raise ValueError(
                "LowOrderConsistency convergence.nlib_max must be greater than "
                "or equal to convergence.nlib_start."
            )
        if self.nburn_max < self.nburn_start:
            raise ValueError(
                "LowOrderConsistency convergence.nburn_max must be greater than "
                "or equal to convergence.nburn_start."
            )
        return self

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

        summary = _quality_summary_from_histograms(
            self.ahist,
            self.rhist,
            self.epsa,
            self.epsr,
            self.target_q1,
            self.target_q2,
        )
        for key, value in summary.items():
            setattr(info, key, value)

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
    if args.get("convergence") is None:
        args.pop("convergence")
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
        convergence: Optional convergence study settings. Omit this block to run
            one low-order calculation with nlib=1 and nburn=1.
        eps0: The minimum value used in the relative difference calculation.
        epsa: The limit for the absolute difference.
        epsr: The limit for the relative difference.
        target_q1: The target for the q1 (relative only) score.
        target_q2: The target for the q2 (weighted relative and absolute) score.

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
            "convergence": "optional ORIGAMI nlib/nburn convergence study",
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
        eps0: Annotated[float, Field(ge=0.0)] = 1e-12,
        epsa: Annotated[float, Field(ge=0.0)] = 1e-6,
        epsr: Annotated[float, Field(ge=0.0)] = 1e-3,
        target_q1: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9,
        target_q2: Annotated[float, Field(ge=0.0, le=1.0)] = 0.95,
        convergence: Optional[LowOrderConsistencyConvergence] = None,
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
        self.convergence_enabled = convergence is not None
        self.convergence = LowOrderConsistencyConvergence.model_validate(
            convergence or {}
        )
        self.nlib_start = self.convergence.nlib_start
        self.nlib_max = self.convergence.nlib_max
        self.nburn_start = self.convergence.nburn_start
        self.nburn_max = self.convergence.nburn_max
        self.q1_stop_criteria = self.convergence.q1_stop_criteria
        self.q2_stop_criteria = self.convergence.q2_stop_criteria

        if _dry_run:
            return

        if _env == None:
            dir = Path.cwd()
        else:
            dir = Path(_env["config_file"]).parent

        tm = core.TemplateManager([dir])

        self.template_path = tm.path(template)
        self.template_paths = tm.paths
        internal.logger.info(
            "check " + __class__.__name__, template_file=self.template_path
        )

        self.work_path = Path(_env["work_dir"])
        self.base_check_path = self.work_path / "check" / name
        self.check_path = self.base_check_path
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
    def make_time_quality_plot(
        image,
        time,
        q1,
        q2,
        target_q1,
        target_q2,
        background=None,
    ):
        """Make the q1/q2-by-time plot."""
        import matplotlib.pyplot as plt

        days = np.asarray(time, dtype=float) / 86400.0
        plt.rcParams.update({"font.size": 18})
        plt.figure()
        for run in background or []:
            time_quality = run["time_quality"]
            background_days = [float(row["time_days"]) for row in time_quality]
            plt.plot(
                background_days,
                [float(row["q1"]) for row in time_quality],
                color="C0",
                alpha=0.18,
                linewidth=1.0,
            )
            plt.plot(
                background_days,
                [float(row["q2"]) for row in time_quality],
                color="C1",
                alpha=0.18,
                linewidth=1.0,
            )
        plt.plot(days, q1, marker="o", label="q1")
        plt.plot(days, q2, marker="s", label="q2")
        plt.axhline(target_q1, color="C0", linestyle="--", label="target q1")
        plt.axhline(target_q2, color="C1", linestyle="--", label="target q2")
        plt.xlabel("time (days)")
        plt.ylabel("quality score")
        plt.ylim(0.0, 1.02)
        plt.legend()
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

    def _quality_summary(self, ahist, rhist):
        return _quality_summary_from_histograms(
            ahist,
            rhist,
            self.epsa,
            self.epsr,
            self.target_q1,
            self.target_q2,
        )

    def _time_quality(self, ahist, rhist):
        rows = []
        for index, time in enumerate(self.time_list):
            summary = self._quality_summary(
                ahist[:, index : index + 1, :],
                rhist[:, index : index + 1, :],
            )
            rows.append(
                {
                    "index": index,
                    "time": float(time),
                    "time_days": float(time) / 86400.0,
                    **summary,
                }
            )
            burnup = self._burnup_for_time_index(index)
            if burnup is not None:
                rows[-1]["burnup"] = burnup
                rows[-1]["burnup_units"] = "MWd/MTIHM"
                rows[-1]["burnup_gwd_per_mtihm"] = burnup / 1000.0
                rows[-1]["burnup_gwd_per_mtu"] = burnup / 1000.0
            self._add_time_quality_shortfall(rows[-1])
        return rows

    @staticmethod
    def _time_quality_pass(time_quality):
        return all(row["test_pass"] for row in time_quality)

    def _burnup_for_time_index(self, index):
        burnup_list = getattr(self, "burnup_list", None)
        if burnup_list is None or len(burnup_list) != len(self.time_list):
            return None
        return float(burnup_list[index])

    @staticmethod
    def _add_time_quality_shortfall(row):
        q1_shortfall = max(0.0, float(row["target_q1"]) - float(row["q1"]))
        q2_shortfall = max(0.0, float(row["target_q2"]) - float(row["q2"]))
        q1_margin = float(row["q1"]) - float(row["target_q1"])
        q2_margin = float(row["q2"]) - float(row["target_q2"])

        if q1_shortfall >= q2_shortfall and q1_shortfall > 0.0:
            limiting_score = "q1"
            value = float(row["q1"])
            target = float(row["target_q1"])
            shortfall = q1_shortfall
        elif q2_shortfall > 0.0:
            limiting_score = "q2"
            value = float(row["q2"])
            target = float(row["target_q2"])
            shortfall = q2_shortfall
        elif q1_margin <= q2_margin:
            limiting_score = "q1"
            value = float(row["q1"])
            target = float(row["target_q1"])
            shortfall = 0.0
        else:
            limiting_score = "q2"
            value = float(row["q2"])
            target = float(row["target_q2"])
            shortfall = 0.0

        row["limiting_score"] = limiting_score
        row["limiting_score_value"] = value
        row["limiting_score_target"] = target
        row["limiting_score_shortfall"] = shortfall

    @staticmethod
    def _worst_time_quality(time_quality):
        if not time_quality:
            return None
        return dict(
            max(
                time_quality,
                key=lambda row: (
                    float(row["limiting_score_shortfall"]),
                    -min(
                        float(row["q1"]) - float(row["target_q1"]),
                        float(row["q2"]) - float(row["target_q2"]),
                    ),
                ),
            )
        )

    @staticmethod
    def _first_failed_time_quality(time_quality):
        for row in time_quality:
            if not row["test_pass"]:
                return dict(row)
        return None

    @staticmethod
    def _burnup_list_from_assemble(assemble_data):
        burnup = assemble_data.get("space", {}).get("burnup", {}).get("grid")
        if burnup is not None:
            return burnup
        for point in assemble_data.get("points", []):
            burnup = point.get("_arpinfo", {}).get("burnup_list")
            if burnup is not None:
                return burnup
        return None

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

    def _use_convergence_subdirectories(self):
        return self.nlib_max > self.nlib_start or self.nburn_max > self.nburn_start

    def _set_check_path_for_convergence(self, nlib, nburn):
        check_path = self.base_check_path
        if self._use_convergence_subdirectories():
            if self.nlib_max > self.nlib_start:
                check_path = check_path / f"nlib{nlib:04d}"
            if self.nburn_max > self.nburn_start:
                check_path = check_path / f"nburn{nburn:04d}"
        self.check_path = check_path
        self.check_dir = self.check_path.relative_to(self.work_path)

    @staticmethod
    def _convergence_summary(info):
        keys = [
            "nlib",
            "nburn",
            "q1",
            "q2",
            "test_pass",
            "test_pass_q1",
            "test_pass_q2",
            "test_pass_time",
            "time_quality",
            "worst_time_quality",
            "first_failed_time_quality",
            "mean_abs_diff",
            "mean_rel_diff",
        ]
        return {key: getattr(info, key) for key in keys if hasattr(info, key)}

    def _scores_converged(self, previous_info, current_info, fixed_grid):
        if previous_info is None:
            return fixed_grid
        dq1 = abs(current_info.q1 - previous_info.q1)
        dq2 = abs(current_info.q2 - previous_info.q2)
        return dq1 <= self.q1_stop_criteria and dq2 <= self.q2_stop_criteria

    @staticmethod
    def _time_quality_plot_background(info):
        background = []
        seen = set()
        final_key = (getattr(info, "nlib", None), getattr(info, "nburn", None))
        for history_name in ("nlib_history", "nburn_history"):
            for row in getattr(info, history_name, []):
                time_quality = row.get("time_quality")
                if not time_quality:
                    continue
                key = (row.get("nlib"), row.get("nburn"))
                if key == final_key or key in seen:
                    continue
                seen.add(key)
                background.append({"label": history_name, "time_quality": time_quality})
        return background

    def _write_time_quality_plot(self, info):
        if not hasattr(info, "time_quality"):
            return
        time_quality = info.time_quality
        image = getattr(info, "time_quality_image", None)
        if not image:
            return
        LowOrderConsistency.make_time_quality_plot(
            image,
            [row["time"] for row in time_quality],
            [row["q1"] for row in time_quality],
            [row["q2"] for row in time_quality],
            self.target_q1,
            self.target_q2,
            background=self._time_quality_plot_background(info),
        )

    @staticmethod
    def _convergence_history_rows(info):
        rows = []
        seen = set()
        for history_name in ("nlib_history", "nburn_history"):
            for row in getattr(info, history_name, []):
                if "time_quality" not in row:
                    continue
                key = (row.get("nlib"), row.get("nburn"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        return rows

    @staticmethod
    def _time_quality_limits(time_quality):
        if not time_quality:
            return None

        min_q1_row = min(time_quality, key=lambda row: float(row["q1"]))
        min_q2_row = min(time_quality, key=lambda row: float(row["q2"]))
        q1 = float(min_q1_row["q1"])
        q2 = float(min_q2_row["q2"])
        target_q1 = float(min_q1_row["target_q1"])
        target_q2 = float(min_q2_row["target_q2"])
        failed_scores = []
        if q1 < target_q1:
            failed_scores.append("q1")
        if q2 < target_q2:
            failed_scores.append("q2")

        limiting_row = LowOrderConsistency._worst_time_quality(time_quality)
        row = {
            "q1": q1,
            "q2": q2,
            "target_q1": target_q1,
            "target_q2": target_q2,
            "test_pass": not failed_scores,
            "pass": "pass" if not failed_scores else "fail " + "/".join(failed_scores),
            "failed_scores": failed_scores,
            "limiting_score": limiting_row["limiting_score"],
            "limiting_score_value": limiting_row["limiting_score_value"],
            "limiting_score_target": limiting_row["limiting_score_target"],
            "limiting_score_shortfall": limiting_row["limiting_score_shortfall"],
            "time": limiting_row["time"],
            "time_days": limiting_row["time_days"],
        }
        for key in (
            "burnup",
            "burnup_units",
            "burnup_gwd_per_mtihm",
            "burnup_gwd_per_mtu",
        ):
            if key in limiting_row:
                row[key] = limiting_row[key]
        return row

    @staticmethod
    def _convergence_time_quality(info):
        rows = []
        for history in LowOrderConsistency._convergence_history_rows(info):
            limits = LowOrderConsistency._time_quality_limits(history["time_quality"])
            if limits is None:
                continue
            rows.append(
                {
                    "nlib": history["nlib"],
                    "nburn": history["nburn"],
                    **limits,
                }
            )
        return rows

    @staticmethod
    def make_convergence_time_quality_plot(
        image,
        convergence_time_quality,
        target_q1,
        target_q2,
    ):
        """Make the convergence plot using worst-over-time q1/q2 scores."""
        import matplotlib.pyplot as plt

        labels = [
            f"{row['nlib']}/{row['nburn']}" for row in convergence_time_quality
        ]
        x = np.arange(len(labels))
        plt.rcParams.update({"font.size": 18})
        plt.figure()
        plt.plot(
            x,
            [float(row["q1"]) for row in convergence_time_quality],
            marker="o",
            label="min q1 over time",
        )
        plt.plot(
            x,
            [float(row["q2"]) for row in convergence_time_quality],
            marker="s",
            label="min q2 over time",
        )
        plt.axhline(target_q1, color="C0", linestyle="--", label="target q1")
        plt.axhline(target_q2, color="C1", linestyle="--", label="target q2")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.xlabel("nlib/nburn")
        plt.ylabel("worst quality score")
        plt.ylim(0.0, 1.02)
        plt.legend()
        plt.savefig(image, bbox_inches="tight")

    def _write_convergence_time_quality(self, info):
        if not self.convergence_enabled:
            return
        convergence_time_quality = self._convergence_time_quality(info)
        if not convergence_time_quality:
            return
        image = self.base_check_path / "q1-q2-by-convergence.png"
        internal.logger.info(
            "creating q1/q2 by convergence",
            image=str(image.relative_to(self.work_path)),
        )
        LowOrderConsistency.make_convergence_time_quality_plot(
            image,
            convergence_time_quality,
            self.target_q1,
            self.target_q2,
        )
        info.convergence_time_quality = convergence_time_quality
        info.convergence_time_quality_image = str(image)

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

        time_quality = self._time_quality(self.ahist, self.rhist)
        time_quality_image = self.check_path / "q1-q2-by-time.png"
        internal.logger.info(
            "creating q1/q2 by time",
            image=str(time_quality_image.relative_to(self.work_path)),
        )
        LowOrderConsistency.make_time_quality_plot(
            time_quality_image,
            self.time_list,
            [row["q1"] for row in time_quality],
            [row["q2"] for row in time_quality],
            self.target_q1,
            self.target_q2,
        )
        info.time_quality = time_quality
        info.time_quality_image = str(time_quality_image)
        info.test_pass_time = self._time_quality_pass(time_quality)
        info.worst_time_quality = self._worst_time_quality(time_quality)
        info.first_failed_time_quality = self._first_failed_time_quality(time_quality)

        summary = self._quality_summary(self.ahist, self.rhist)
        for key, value in summary.items():
            setattr(info, key, value)
        info.test_pass = info.test_pass and info.test_pass_time

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

        return info

    def __run_lo_order(self, do_run, nlib, nburn):
        """Run the LOW order calculation which should be consistent as possible with
        the already-complete higher order calculation."""

        # Load the template file.
        with open(self.template_path, "r") as f:
            template_text = f.read()

        # Load the assemble data.
        assemble_json = self.work_path / "assemble.olm.json"
        with open(assemble_json, "r") as f:
            assemble_d = json.load(f)
        self.burnup_list = self._burnup_list_from_assemble(assemble_d)

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
            check = {"name": self.name}
            if self.convergence_enabled:
                check["convergence"] = {
                    "nburn": nburn,
                    "nlib": nlib,
                }

            check_data = {
                **point,
                "name": self.name,
                "check": check,
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
            filled_text = core.TemplateManager.expand_text(
                template_text,
                check_data,
                src_path=str(self.template_path),
                search_paths=self.template_paths,
                float_format=core.TemplateManager.template_float_format(self._model),
            )

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

    def _run_once(self, do_run, nlib, nburn):
        self._set_check_path_for_convergence(nlib, nburn)
        self.ii_json_list = self.__run_lo_order(do_run, nlib, nburn)
        self.__load_ii_json(self.ii_json_list)
        self.run_success = True
        info = self.info()
        info.nlib = nlib
        info.nburn = nburn
        info.nlib_converged = self.nlib_start == self.nlib_max
        info.nburn_converged = self.nburn_start == self.nburn_max
        info.test_pass_nlib = True
        info.test_pass_nburn = True
        return info

    def _run_nlib_convergence(self, do_run, nburn):
        previous_info = None
        current_info = None
        nlib_history = []
        nlib = self.nlib_start

        while True:
            current_info = self._run_once(do_run, nlib, nburn)
            converged = self._scores_converged(
                previous_info,
                current_info,
                fixed_grid=self.nlib_start == self.nlib_max,
            )
            if previous_info is not None:
                current_info.nlib_delta_q1 = abs(current_info.q1 - previous_info.q1)
                current_info.nlib_delta_q2 = abs(current_info.q2 - previous_info.q2)
            current_info.nlib_converged = converged
            current_info.test_pass_nlib = converged
            nlib_history.append(self._convergence_summary(current_info))

            if converged or nlib >= self.nlib_max:
                break

            previous_info = current_info
            nlib = min(nlib * 2, self.nlib_max)

        current_info.nlib_history = nlib_history
        return current_info

    def _copy_nlib_convergence_status(self, target_info, nlib_info):
        target_info.nlib_history = nlib_info.nlib_history
        target_info.nlib_converged = nlib_info.nlib_converged
        target_info.test_pass_nlib = nlib_info.test_pass_nlib
        if hasattr(nlib_info, "nlib_delta_q1"):
            target_info.nlib_delta_q1 = nlib_info.nlib_delta_q1
        if hasattr(nlib_info, "nlib_delta_q2"):
            target_info.nlib_delta_q2 = nlib_info.nlib_delta_q2

    def _run_nburn_convergence(self, do_run, nlib_info):
        previous_info = None
        current_info = nlib_info
        nburn_history = []
        nburn = self.nburn_start
        nlib = nlib_info.nlib

        while True:
            if previous_info is None:
                current_info = nlib_info
            else:
                current_info = self._run_once(do_run, nlib, nburn)
                self._copy_nlib_convergence_status(current_info, nlib_info)

            converged = self._scores_converged(
                previous_info,
                current_info,
                fixed_grid=self.nburn_start == self.nburn_max,
            )
            if previous_info is not None:
                current_info.nburn_delta_q1 = abs(current_info.q1 - previous_info.q1)
                current_info.nburn_delta_q2 = abs(current_info.q2 - previous_info.q2)
            current_info.nburn_converged = converged
            current_info.test_pass_nburn = converged
            nburn_history.append(self._convergence_summary(current_info))

            if converged or nburn >= self.nburn_max:
                break

            previous_info = current_info
            nburn = min(nburn * 2, self.nburn_max)

        current_info.nburn_history = nburn_history
        return current_info

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
        current_info = None

        try:
            if self.convergence_enabled:
                current_info = self._run_nlib_convergence(do_run, self.nburn_start)
                if current_info.nlib_converged:
                    current_info = self._run_nburn_convergence(do_run, current_info)
                else:
                    current_info.nburn_history = []
            else:
                current_info = self._run_once(do_run, 1, 1)

        except ValueError as ve:
            self.run_success = False
            internal.logger.error(str(ve))
            current_info = self.info()

        if self.convergence_enabled:
            current_info.nlib = getattr(current_info, "nlib", self.nlib_start)
            current_info.nburn = getattr(current_info, "nburn", self.nburn_start)
            current_info.nlib_start = self.nlib_start
            current_info.nlib_max = self.nlib_max
            current_info.nburn_start = self.nburn_start
            current_info.nburn_max = self.nburn_max
            current_info.q1_stop_criteria = self.q1_stop_criteria
            current_info.q2_stop_criteria = self.q2_stop_criteria
            current_info.nlib_history = getattr(current_info, "nlib_history", [])
            current_info.nburn_history = getattr(current_info, "nburn_history", [])
            current_info.nlib_converged = getattr(
                current_info, "nlib_converged", self.nlib_start == self.nlib_max
            )
            current_info.nburn_converged = getattr(
                current_info, "nburn_converged", self.nburn_start == self.nburn_max
            )
            current_info.test_pass_nlib = getattr(
                current_info, "test_pass_nlib", current_info.nlib_converged
            )
            current_info.test_pass_nburn = getattr(
                current_info, "test_pass_nburn", current_info.nburn_converged
            )
            current_info.test_pass = (
                current_info.test_pass
                and current_info.test_pass_nlib
                and current_info.test_pass_nburn
            )
        else:
            for key in (
                "nlib",
                "nburn",
                "nlib_converged",
                "nburn_converged",
                "test_pass_nlib",
                "test_pass_nburn",
            ):
                if hasattr(current_info, key):
                    delattr(current_info, key)

        self._write_convergence_time_quality(current_info)
        self._write_time_quality_plot(current_info)
        return current_info
