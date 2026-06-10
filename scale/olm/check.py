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
import shutil
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
    eps0,
    epsa,
    epsr,
    target_q_r,
    target_q_ar,
):
    """Return q-score data for already calculated absolute/relative errors."""
    ahist = np.ndarray.flatten(np.asarray(ahist, dtype=float))
    rhist = np.ndarray.flatten(np.asarray(rhist, dtype=float))
    if len(ahist) != len(rhist):
        raise ValueError("Absolute and relative error histograms must be the same size.")
    if len(ahist) == 0:
        raise ValueError("Cannot calculate q-scores from empty error histograms.")
    if eps0 > 0.0:
        keep = np.logical_or(ahist >= eps0, rhist >= eps0)
        ahist = ahist[keep]
        rhist = rhist[keep]

    w_ar = int(np.logical_and((ahist > epsa), (rhist > epsr)).sum())
    w_r = int((rhist > epsr).sum())
    m = int(len(ahist))
    if m == 0:
        f_r = 0.0
        f_ar = 0.0
        mean_abs_diff = 0.0
        mean_rel_diff = 0.0
        std_abs_diff = 0.0
        std_rel_diff = 0.0
    else:
        f_r = float(w_r) / m
        f_ar = float(w_ar) / m
        mean_abs_diff = float(np.mean(ahist))
        mean_rel_diff = float(np.mean(rhist))
        std_abs_diff = float(np.std(ahist))
        std_rel_diff = float(np.std(rhist))
    q_r = float(np.clip(1.0 - f_r, 0.0, 1.0))
    q_ar = float(np.clip(1.0 - 0.9 * f_ar - 0.1 * f_r, 0.0, 1.0))
    return {
        "w_ar": w_ar,
        "w_r": w_r,
        "m": m,
        "f_r": f_r,
        "q_r": q_r,
        "target_q_r": target_q_r,
        "test_pass_q_r": q_r >= target_q_r,
        "f_ar": f_ar,
        "q_ar": q_ar,
        "target_q_ar": target_q_ar,
        "test_pass_q_ar": q_ar >= target_q_ar,
        "test_pass": q_r >= target_q_r and q_ar >= target_q_ar,
        "mean_abs_diff": mean_abs_diff,
        "mean_rel_diff": mean_rel_diff,
        "std_abs_diff": std_abs_diff,
        "std_rel_diff": std_rel_diff,
    }


class LowOrderConsistencyConvergence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nlib_start: Annotated[int, Field(gt=0)] = 1
    nlib_max: Annotated[int, Field(gt=0)] = 1
    nburn_start: Annotated[int, Field(gt=0)] = 1
    nburn_max: Annotated[int, Field(gt=0)] = 1
    q_r_stop_criteria: Annotated[float, Field(ge=0.0)] = 0.0
    q_ar_stop_criteria: Annotated[float, Field(ge=0.0)] = 0.0

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
                "template": "model/origami/lumped0d-uox.jt.inp",
                "metric": "grams_per_initial_hm",
                "target_q_r": 0.70,
                "target_q_ar": 0.95,
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
    is the relative quality score, :code:`q_r=1-f_r` where :code:`f_r` is the failed
    fraction. The relative score passes if :code:`q_r >= target_q_r`.

    Most often, we care less about relative differences when the absolute values are
    very small, e.g. a 10% difference in a 1e-12 barn cross section is not as big
    a deal as a 1% difference in a 100 barn cross section. Quality score :code:`q_ar`
    takes this into account by considering the fraction of points which fail the
    pure relative test, :code:`q_r`, and those that fail a combined test where the
    relative gradient must exceed :code:`epsr` and the absolute gradient must exceed
    :code:`epsa`. The failed fraction is :code:`f_ar` and the combined score for
    :code:`q_ar=1-0.9*f_ar-0.1*f_r`. In this way, one cannot get a perfect 1.0 for either
    score if there are any failures in a relative sense, but the second score penalizes
    them less. The absolute-relative score passes if :code:`q_ar >= target_q_ar`.

    Args:
        eprs: The limit for the relative gradient.
        epsa: The limit for the absolute gradient.
        target_q_r: The target for the q_r (relative only) score.
        target_q_ar: The target for the q_ar (weighted relative and absolute) score.
        eps0: The minimum gradient to care about.

    """

    @staticmethod
    def describe_params():
        return {
            "eps0": "minimum value",
            "epsa": "absolute epsilon",
            "epsr": "relative epsilon",
            "target_q_r": "target for relative quality score",
            "target_q_ar": "target for absolute-relative quality score",
        }

    @staticmethod
    def default_params():
        c = GridGradient()
        return {
            "eps0": c.eps0,
            "epsa": c.epsa,
            "epsr": c.epsr,
            "target_q_ar": c.target_q_ar,
            "target_q_r": c.target_q_r,
        }

    def __init__(
        self,
        _model: dict = None,
        _env: dict = {},
        eps0: float = 1e-20,
        epsa: float = 1e-1,
        epsr: float = 1e-1,
        target_q_r: float = 0.5,
        target_q_ar: float = 0.7,
        _type: Literal[_TYPE_GRIDGRADIENT] = None,
    ):
        self.eps0 = eps0
        self.epsa = epsa
        self.epsr = epsr
        self.target_q_r = target_q_r
        self.target_q_ar = target_q_ar
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
            + " with q_r={:.2f} and q_ar={:.2f}".format(info.q_r, info.q_ar)
        )

        return info

    def info(self):
        """Recalculate and return the score information."""
        info = CheckInfo()
        info.name = self.__class__.__name__
        info.eps0 = self.eps0
        info.epsa = self.epsa
        info.epsr = self.epsr
        info.target_q_r = self.target_q_r
        info.target_q_ar = self.target_q_ar

        summary = _quality_summary_from_histograms(
            self.ahist,
            self.rhist,
            self.eps0,
            self.epsa,
            self.epsr,
            self.target_q_r,
            self.target_q_ar,
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
        target_q_r: The target for the q_r (relative only) score.
        target_q_ar: The target for the q_ar (weighted relative and absolute) score.
        nuclide_scaled_difference_min_abs_ylim: Minimum absolute y-axis limit for
            nuclide scaled-difference plots, as a fraction. Omit this to use epsr.

    """

    _PLOT_FIGSIZE = (4.0, 3.0)
    _PLOT_STYLE = {
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }

    @staticmethod
    def describe_params():
        return {
            "metric": "primary inventory metric",
            "eps0": "minimum value",
            "epsa": "absolute epsilon",
            "epsr": "relative epsilon",
            "target_q_r": "target for relative quality score",
            "target_q_ar": "target for absolute-relative quality score",
            "convergence": "optional ORIGAMI nlib/nburn convergence study",
            "nuclide_compare": "plot me",
            "nuclide_scaled_difference_min_abs_ylim": (
                "minimum absolute y-axis limit for nuclide scaled-difference plots; "
                "omit to use epsr"
            ),
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
        target_q_r: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9,
        target_q_ar: Annotated[float, Field(ge=0.0, le=1.0)] = 0.95,
        convergence: Optional[LowOrderConsistencyConvergence] = None,
        nuclide_compare: List[str] = ["u235", "pu239"],
        nuclide_scaled_difference_min_abs_ylim: Optional[
            Annotated[float, Field(ge=0.0)]
        ] = None,
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
        self.target_q_r = target_q_r
        self.target_q_ar = target_q_ar
        self.convergence_enabled = convergence is not None
        self.nuclide_scaled_difference_min_abs_ylim = (
            epsr
            if nuclide_scaled_difference_min_abs_ylim is None
            else nuclide_scaled_difference_min_abs_ylim
        )
        self.convergence = LowOrderConsistencyConvergence.model_validate(
            convergence or {}
        )
        self.nlib_start = self.convergence.nlib_start
        self.nlib_max = self.convergence.nlib_max
        self.nburn_start = self.convergence.nburn_start
        self.nburn_max = self.convergence.nburn_max
        self.q_r_stop_criteria = self.convergence.q_r_stop_criteria
        self.q_ar_stop_criteria = self.convergence.q_ar_stop_criteria

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
        self.convergence_check_path = self.base_check_path / "_convergence"
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
        min_abs_ylim=0.0,
    ):
        """Make the scaled-difference plot."""
        import matplotlib.pyplot as plt

        with plt.rc_context(LowOrderConsistency._PLOT_STYLE):
            plt.figure(figsize=LowOrderConsistency._PLOT_FIGSIZE)
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

            max_abs_ylim = max(float(max_abs_scaled_difference), float(min_abs_ylim))
            plt.ylim(-100.0 * max_abs_ylim, 100.0 * max_abs_ylim)
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
            plt.close()

    @staticmethod
    def make_time_quality_plot(
        image,
        time,
        q_r,
        q_ar,
        target_q_r,
        target_q_ar,
        convergence_history=None,
    ):
        """Make the q_r/q_ar-by-time plot."""
        import matplotlib.pyplot as plt

        days = np.asarray(time, dtype=float) / 86400.0
        with plt.rc_context(LowOrderConsistency._PLOT_STYLE):
            plt.figure(figsize=LowOrderConsistency._PLOT_FIGSIZE)
            for row in convergence_history or []:
                time_quality = row.get("time_quality")
                if not time_quality:
                    continue
                history_days = [point["time_days"] for point in time_quality]
                plt.plot(
                    history_days,
                    [point["q_r"] for point in time_quality],
                    color="C0",
                    alpha=0.15,
                    linewidth=0.8,
                )
                plt.plot(
                    history_days,
                    [point["q_ar"] for point in time_quality],
                    color="C1",
                    alpha=0.15,
                    linewidth=0.8,
                )
            plt.plot(days, q_r, marker="o", label="q_r")
            plt.plot(days, q_ar, marker="s", label="q_ar")
            plt.axhline(target_q_r, color="C0", linestyle="--", label="target q_r")
            plt.axhline(target_q_ar, color="C1", linestyle="--", label="target q_ar")
            plt.xlabel("time (days)")
            plt.ylabel("quality score")
            plt.ylim(0.0, 1.02)
            plt.legend()
            plt.savefig(image, bbox_inches="tight")
            plt.close()

    @staticmethod
    def make_convergence_quality_plot(
        image,
        convergence_history,
        target_q_r,
        target_q_ar,
    ):
        """Make the minimum q_r/q_ar convergence plot."""
        import matplotlib.pyplot as plt

        x = np.arange(len(convergence_history))
        labels = [
            "{}/{}".format(row["nlib"], row["nburn"])
            for row in convergence_history
        ]
        with plt.rc_context(LowOrderConsistency._PLOT_STYLE):
            plt.figure(figsize=LowOrderConsistency._PLOT_FIGSIZE)
            plt.plot(
                x,
                [row["q_r"] for row in convergence_history],
                marker="o",
                label="q_r",
            )
            plt.plot(
                x,
                [row["q_ar"] for row in convergence_history],
                marker="s",
                label="q_ar",
            )
            plt.axhline(target_q_r, color="C0", linestyle="--", label="target q_r")
            plt.axhline(target_q_ar, color="C1", linestyle="--", label="target q_ar")
            plt.xticks(x, labels)
            plt.xlabel("nlib/nburn")
            plt.ylabel("minimum quality score")
            plt.ylim(0.0, 1.02)
            plt.legend()
            plt.savefig(image, bbox_inches="tight")
            plt.close()

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
            self.eps0,
            self.epsa,
            self.epsr,
            self.target_q_r,
            self.target_q_ar,
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
            q_r_shortfall = max(0.0, rows[-1]["target_q_r"] - rows[-1]["q_r"])
            q_ar_shortfall = max(0.0, rows[-1]["target_q_ar"] - rows[-1]["q_ar"])
            if q_r_shortfall or q_ar_shortfall:
                rows[-1]["limiting_score"] = (
                    "q_r" if q_r_shortfall >= q_ar_shortfall else "q_ar"
                )
            else:
                q_r_margin = rows[-1]["q_r"] - rows[-1]["target_q_r"]
                q_ar_margin = rows[-1]["q_ar"] - rows[-1]["target_q_ar"]
                rows[-1]["limiting_score"] = (
                    "q_r" if q_r_margin <= q_ar_margin else "q_ar"
                )
            rows[-1]["limiting_score_shortfall"] = max(
                q_r_shortfall, q_ar_shortfall
            )
        return rows

    @staticmethod
    def _test_pass_initial(time_quality):
        if not time_quality:
            return False
        initial = time_quality[0]
        return (
            LowOrderConsistency._test_pass_initial_score(time_quality, "q_r")
            and LowOrderConsistency._test_pass_initial_score(time_quality, "q_ar")
        )

    @staticmethod
    def _test_pass_initial_score(time_quality, score_key):
        if not time_quality:
            return False
        initial = time_quality[0]
        return (
            np.isclose(float(initial["time"]), 0.0, rtol=0.0, atol=1.0e-6)
            and float(initial[score_key]) == 1.0
        )

    @staticmethod
    def _minimum_score_time_quality(time_quality, score_key, target_key):
        if not time_quality:
            return None
        row = dict(min(time_quality, key=lambda row: float(row[score_key])))
        value = float(row[score_key])
        target = float(row[target_key])
        row["score"] = value
        row["target"] = target
        row["test_pass_score"] = value >= target
        return row

    @staticmethod
    def _minimum_time_quality(time_quality):
        if not time_quality:
            return None
        q_r_row = LowOrderConsistency._minimum_score_time_quality(
            time_quality, "q_r", "target_q_r"
        )
        q_ar_row = LowOrderConsistency._minimum_score_time_quality(
            time_quality, "q_ar", "target_q_ar"
        )
        return {
            "q_r": q_r_row["score"],
            "q_ar": q_ar_row["score"],
            "target_q_r": q_r_row["target"],
            "target_q_ar": q_ar_row["target"],
            "test_pass_q_r": q_r_row["test_pass_score"],
            "test_pass_q_ar": q_ar_row["test_pass_score"],
            "test_pass": q_r_row["test_pass_score"] and q_ar_row["test_pass_score"],
        }

    def _burnup_for_time_index(self, index):
        burnup_list = getattr(self, "burnup_list", None)
        if burnup_list is None or len(burnup_list) != len(self.time_list):
            return None
        return float(burnup_list[index])

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
                        float(row["q_r"]) - float(row["target_q_r"]),
                        float(row["q_ar"]) - float(row["target_q_ar"]),
                    ),
                ),
            )
        )

    @staticmethod
    def _convergence_time_summary(time_quality):
        if not time_quality:
            return {}

        minimum = LowOrderConsistency._minimum_time_quality(time_quality)
        failing_rows = [row for row in time_quality if not row["test_pass"]]
        location = LowOrderConsistency._worst_time_quality(
            failing_rows or time_quality
        )
        failures = []
        if not minimum["test_pass_q_r"]:
            failures.append("q_r")
        if not minimum["test_pass_q_ar"]:
            failures.append("q_ar")

        summary = {
            "q_r": minimum["q_r"],
            "q_ar": minimum["q_ar"],
            "test_pass_time": not failures,
            "result": "pass" if not failures else "fail " + "/".join(failures),
            "time": float(location["time"]),
            "time_days": float(location["time_days"]),
            "limiting_score": location["limiting_score"],
        }
        if "burnup" in location:
            summary["burnup"] = float(location["burnup"])
        if "burnup_gwd_per_mtihm" in location:
            summary["burnup_gwd_per_mtihm"] = float(
                location["burnup_gwd_per_mtihm"]
            )
            summary["burnup_gwd_per_mtu"] = float(
                location["burnup_gwd_per_mtihm"]
            )
        return summary

    @staticmethod
    def _burnup_list_from_assemble(assemble_data):
        history_burnups = []
        for point in assemble_data.get("points", []):
            history = point.get("history", {})
            burndata = history.get("burndata")
            if burndata is None:
                continue
            history_burnups.append(
                np.asarray(
                    LowOrderConsistency._history_burnup_points(history),
                    dtype=float,
                )
            )

        if history_burnups:
            reference_shape = history_burnups[0].shape
            if any(burnups.shape != reference_shape for burnups in history_burnups):
                raise ValueError(
                    "LowOrderConsistency cannot render burnup by time because "
                    "assembled point histories have different burnup counts."
                )
            return np.mean(np.stack(history_burnups), axis=0).tolist()

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
        if self.convergence_enabled:
            check_path = self.convergence_check_path
        if self.convergence_enabled and self._use_convergence_subdirectories():
            if self.nlib_max > self.nlib_start:
                check_path = check_path / f"nlib{nlib:04d}"
            if self.nburn_max > self.nburn_start:
                check_path = check_path / f"nburn{nburn:04d}"
        self.check_path = check_path
        self.check_dir = self.check_path.relative_to(self.work_path)

    @staticmethod
    def _replace_path_prefix(value, old_prefix, new_prefix):
        if isinstance(value, str) and value.startswith(old_prefix):
            return new_prefix + value[len(old_prefix):]
        if isinstance(value, list):
            return [
                LowOrderConsistency._replace_path_prefix(v, old_prefix, new_prefix)
                for v in value
            ]
        if isinstance(value, dict):
            return {
                k: LowOrderConsistency._replace_path_prefix(
                    v, old_prefix, new_prefix
                )
                for k, v in value.items()
            }
        return value

    @staticmethod
    def _rewrite_info_path_prefix(info, old_prefix, new_prefix):
        for key, value in list(info.__dict__.items()):
            setattr(
                info,
                key,
                LowOrderConsistency._replace_path_prefix(
                    value, old_prefix, new_prefix
                ),
            )

    def _clear_nominal_check_path(self):
        self.base_check_path.mkdir(parents=True, exist_ok=True)
        for child in self.base_check_path.iterdir():
            if child.name == "_convergence":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def _copy_final_check_outputs_to_nominal(self, info):
        source = self.check_path
        target = self.base_check_path
        if source == target:
            return
        if not source.exists():
            return

        self._clear_nominal_check_path()
        for child in source.iterdir():
            destination = target / child.name
            if child.is_dir():
                shutil.copytree(child, destination)
            else:
                shutil.copy2(child, destination)

        self._rewrite_info_path_prefix(info, str(source), str(target))
        self.check_path = target
        self.check_dir = self.check_path.relative_to(self.work_path)

    @staticmethod
    def _convergence_summary(info):
        keys = [
            "nlib",
            "nburn",
            "q_r",
            "q_ar",
            "test_pass",
            "test_pass_q_r",
            "test_pass_q_ar",
            "test_pass_time",
            "mean_abs_diff",
            "mean_rel_diff",
        ]
        summary = {key: getattr(info, key) for key in keys if hasattr(info, key)}
        if hasattr(info, "time_quality"):
            summary["time_quality"] = info.time_quality
            summary.update(
                LowOrderConsistency._convergence_time_summary(info.time_quality)
            )
        elif "test_pass" in summary:
            summary["result"] = "pass" if summary["test_pass"] else "fail"
        return summary

    @staticmethod
    def _combined_convergence_history(*histories):
        rows = []
        seen = set()
        for history in histories:
            for row in history or []:
                key = (row.get("nlib"), row.get("nburn"))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
        return rows

    @staticmethod
    def _nonfinal_convergence_history(convergence_history, final_info):
        final_key = (
            getattr(final_info, "nlib", None),
            getattr(final_info, "nburn", None),
        )
        return [
            row
            for row in convergence_history
            if (row.get("nlib"), row.get("nburn")) != final_key
        ]

    def _write_convergence_diagnostics(self, info):
        convergence_history = self._combined_convergence_history(
            getattr(info, "nlib_history", []),
            getattr(info, "nburn_history", []),
        )
        info.convergence_history = convergence_history
        if not convergence_history:
            return

        nonfinal_history = self._nonfinal_convergence_history(
            convergence_history, info
        )
        if hasattr(info, "time_quality"):
            time_quality_image = Path(info.time_quality_image)
            LowOrderConsistency.make_time_quality_plot(
                time_quality_image,
                [row["time"] for row in info.time_quality],
                [row["q_r"] for row in info.time_quality],
                [row["q_ar"] for row in info.time_quality],
                self.target_q_r,
                self.target_q_ar,
                convergence_history=nonfinal_history,
            )

        if len(convergence_history) > 1:
            image = self.check_path / "q_r-q_ar-convergence.png"
            LowOrderConsistency.make_convergence_quality_plot(
                image,
                convergence_history,
                self.target_q_r,
                self.target_q_ar,
            )
            info.convergence_quality_image = str(image)

    @staticmethod
    def _format_convergence_delta(delta, criterion):
        if delta is None:
            return ""
        return "{:.3e} / {:.3e}".format(float(delta), float(criterion))

    @staticmethod
    def _convergence_scores(info):
        if hasattr(info, "time_quality"):
            summary = LowOrderConsistency._convergence_time_summary(
                info.time_quality
            )
            if "q_r" in summary and "q_ar" in summary:
                return float(summary["q_r"]), float(summary["q_ar"])
        return float(info.q_r), float(info.q_ar)

    def _convergence_stop_reason(self, previous_info, current_info, fixed_grid):
        if previous_info is None:
            return "fixed" if fixed_grid else ""

        previous_q_r, previous_q_ar = self._convergence_scores(previous_info)
        current_q_r, current_q_ar = self._convergence_scores(current_info)
        delta_q_r = abs(current_q_r - previous_q_r)
        delta_q_ar = abs(current_q_ar - previous_q_ar)
        if (
            delta_q_r <= self.q_r_stop_criteria
            and delta_q_ar <= self.q_ar_stop_criteria
        ):
            return "delta"

        crossed_targets = (
            current_q_r >= self.target_q_r
            and current_q_ar >= self.target_q_ar
            and (previous_q_r < self.target_q_r or previous_q_ar < self.target_q_ar)
            and current_q_r >= previous_q_r
            and current_q_ar >= previous_q_ar
        )
        if crossed_targets:
            return "target"

        return ""

    def _set_convergence_deltas(self, current_info, previous_info, prefix):
        if previous_info is None:
            return
        current_q_r, current_q_ar = self._convergence_scores(current_info)
        previous_q_r, previous_q_ar = self._convergence_scores(previous_info)
        setattr(current_info, f"{prefix}_delta_q_r", abs(current_q_r - previous_q_r))
        setattr(
            current_info,
            f"{prefix}_delta_q_ar",
            abs(current_q_ar - previous_q_ar),
        )

    def _convergence_status_row(self, info, name, default_value, max_value):
        delta_q_r = getattr(info, f"{name}_delta_q_r", None)
        delta_q_ar = getattr(info, f"{name}_delta_q_ar", None)
        stop_reason = getattr(info, f"{name}_convergence_stop", "")
        if getattr(info, "run_error", ""):
            result = "error"
            reason = "check did not complete"
        elif stop_reason == "delta":
            result = "selected"
            reason = "q_r/q_ar change is within the stop criteria"
        elif stop_reason == "target":
            result = "selected"
            reason = "q_r/q_ar increased and crossed the pass targets"
        elif stop_reason == "fixed":
            result = "selected"
            reason = f"single configured {name} value"
        elif stop_reason == "max":
            result = "max"
            reason = f"reached {name}_max before another stop criterion"
        else:
            result = "selected"
            reason = ""
        return {
            "result": result,
            "value": int(getattr(info, name, default_value)),
            "max": int(max_value),
            "delta_q_r": delta_q_r,
            "delta_q_ar": delta_q_ar,
            "delta_scope": "minimum time q-score",
            "delta_q_r_text": self._format_convergence_delta(
                delta_q_r, self.q_r_stop_criteria
            ),
            "delta_q_ar_text": self._format_convergence_delta(
                delta_q_ar, self.q_ar_stop_criteria
            ),
            "reason": reason,
        }

    def _convergence_status(self, info):
        status = {}
        if self.nlib_max > self.nlib_start:
            status["nlib"] = self._convergence_status_row(
                info, "nlib", self.nlib_start, self.nlib_max
            )

        if self.nburn_max > self.nburn_start:
            if status.get("nlib", {}).get("result") == "error":
                status["nburn"] = {
                    "result": "not run",
                    "value": int(getattr(info, "nburn", self.nburn_start)),
                    "max": int(self.nburn_max),
                    "delta_q_r_text": "",
                    "delta_q_ar_text": "",
                    "reason": "skipped because nlib search did not complete",
                }
            else:
                status["nburn"] = self._convergence_status_row(
                    info, "nburn", self.nburn_start, self.nburn_max
                )
        return status

    def _failure_reasons(self, info):
        reasons = []
        if getattr(info, "run_error", ""):
            reasons.append("check calculation failed: {}".format(info.run_error))
            return reasons
        if (
            hasattr(info, "test_pass_initial_q_r")
            and not info.test_pass_initial_q_r
        ):
            reasons.append(
                "test 1.1 failed: q_r at t=0 must be 1.000"
            )
        if (
            hasattr(info, "test_pass_initial_q_ar")
            and not info.test_pass_initial_q_ar
        ):
            reasons.append(
                "test 1.2 failed: q_ar at t=0 must be 1.000"
            )
        if hasattr(info, "test_pass_time_q_r") and not info.test_pass_time_q_r:
            reasons.append(
                "test 2.1 failed: final q_r did not meet its target at one "
                "or more high-order time points"
            )
        if hasattr(info, "test_pass_time_q_ar") and not info.test_pass_time_q_ar:
            reasons.append(
                "test 2.2 failed: final q_ar did not meet its target at one "
                "or more high-order time points"
            )
        return reasons

    def _scores_converged(self, previous_info, current_info, fixed_grid):
        return bool(
            self._convergence_stop_reason(previous_info, current_info, fixed_grid)
        )

    @staticmethod
    def _mox_lumped0d_for_origami_data(comp, interpvars):
        target_pu239 = float(interpvars["pu239_frac"])
        pu_vector_scale = (100.0 - target_pu239) / (
            100.0 - float(comp["puo2"]["iso"]["pu239"])
        )
        pu_component = float(interpvars["pu_frac"]) / float(
            comp["info"]["puo2_hm_frac"]
        )
        am_component = (
            100.0
            * float(comp["amo2"]["dens_frac"])
            * float(comp["info"]["amo2_hm_frac"])
        )
        am_oxygen_component = 100.0 * float(comp["amo2"]["dens_frac"]) * (
            1.0 - float(comp["info"]["amo2_hm_frac"])
        )
        uo2_component = 100.0 - pu_component - am_component - am_oxygen_component
        puo2_iso = {
            "pu238": float(comp["puo2"]["iso"]["pu238"]) * pu_vector_scale,
            "pu239": target_pu239,
            "pu240": float(comp["puo2"]["iso"]["pu240"]) * pu_vector_scale,
            "pu241": float(comp["puo2"]["iso"]["pu241"]) * pu_vector_scale,
            "pu242": float(comp["puo2"]["iso"]["pu242"]) * pu_vector_scale,
        }
        return {
            "density": comp["density"],
            "uo2_component": uo2_component,
            "puo2_component": pu_component,
            "am_component": am_component,
            "oxygen_component": am_oxygen_component,
            "puo2_iso": puo2_iso,
        }

    def _lumped0d_for_origami_data(self, point):
        comp = point["comp"]["system"]
        interpvars = point["_arpinfo"]["interpvars"]
        data = {}
        if "pu_frac" in interpvars and "pu239_frac" in interpvars:
            data["mox"] = self._mox_lumped0d_for_origami_data(comp, interpvars)
        return data

    def _point_with_arpinfo_name(self, point):
        point_data = copy.deepcopy(point)
        arpinfo = point_data.setdefault("_arpinfo", {})
        if "name" not in arpinfo:
            arpinfo["name"] = self._model["name"]
        return point_data

    @staticmethod
    def _history_burnup_points(history):
        burnup = 0.0
        burnups = [burnup]
        for row in history.get("burndata", []):
            burnup += float(row["power"]) * float(row["burn"])
            burnups.append(burnup)
        return burnups

    @staticmethod
    def _max_required_library_burnup(history, nlib):
        if nlib <= 0:
            raise ValueError(f"LowOrderConsistency nlib must be > 0; got {nlib}")

        burnups = LowOrderConsistency._history_burnup_points(history)
        if len(burnups) < 2:
            return 0.0

        interpolation_fraction = (2.0 * float(nlib) - 1.0) / (2.0 * float(nlib))
        required = []
        for previous_burnup, current_burnup in zip(burnups[:-1], burnups[1:]):
            required.append(
                previous_burnup
                + interpolation_fraction * (current_burnup - previous_burnup)
            )
        return max(required)

    @staticmethod
    def _require_history_within_library_burnup(
        point, point_name, nlib, burnup_rtol=2.0e-2
    ):
        burnup_list = point.get("_arpinfo", {}).get("burnup_list")
        if not burnup_list:
            return

        library_max = float(burnup_list[-1])
        required_burnup = LowOrderConsistency._max_required_library_burnup(
            point["history"], nlib
        )
        tolerance = max(
            1.0e-6,
            float(burnup_rtol) * max(abs(library_max), abs(required_burnup)),
        )
        if required_burnup > library_max + tolerance:
            raise ValueError(
                "LowOrderConsistency history for point="
                f"{point_name} with nlib={nlib} requires low-order library "
                f"interpolation through {required_burnup:.6g} MWd/MTIHM, "
                "but the assembled low-order library burnup grid ends at "
                f"{library_max:.6g} MWd/MTIHM. Add a later high-order burnup "
                "point for library generation, reduce nlib, or limit the "
                "consistency check history to burnups covered by the library."
            )

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
        info.target_q_r = self.target_q_r
        info.target_q_ar = self.target_q_ar
        info.metric = self.metric
        info.units = self._metric_units(self.metric)
        info.nuclide_scaled_difference_min_abs_ylim = (
            self.nuclide_scaled_difference_min_abs_ylim
        )
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
                min_abs_ylim=self.nuclide_scaled_difference_min_abs_ylim,
            )

        time_quality = self._time_quality(self.ahist, self.rhist)
        time_quality_image = self.check_path / "q_r-q_ar-by-time.png"
        internal.logger.info(
            "creating q_r/q_ar by time",
            image=str(time_quality_image.relative_to(self.work_path)),
        )
        LowOrderConsistency.make_time_quality_plot(
            time_quality_image,
            self.time_list,
            [row["q_r"] for row in time_quality],
            [row["q_ar"] for row in time_quality],
            self.target_q_r,
            self.target_q_ar,
        )
        info.time_quality = time_quality
        info.time_quality_image = str(time_quality_image)
        info.initial_time_quality = dict(time_quality[0]) if time_quality else None
        info.test_pass_initial_q_r = self._test_pass_initial_score(
            time_quality, "q_r"
        )
        info.test_pass_initial_q_ar = self._test_pass_initial_score(
            time_quality, "q_ar"
        )
        info.minimum_q_r_time_quality = self._minimum_score_time_quality(
            time_quality, "q_r", "target_q_r"
        )
        info.minimum_q_ar_time_quality = self._minimum_score_time_quality(
            time_quality, "q_ar", "target_q_ar"
        )
        info.minimum_time_quality = self._minimum_time_quality(time_quality)
        info.test_pass_initial = self._test_pass_initial(time_quality)
        info.test_pass_time_q_r = all(row["test_pass_q_r"] for row in time_quality)
        info.test_pass_time_q_ar = all(row["test_pass_q_ar"] for row in time_quality)
        info.test_pass_time = info.test_pass_time_q_r and info.test_pass_time_q_ar
        info.worst_time_quality = self._worst_time_quality(time_quality)
        info.first_failed_time_quality = next(
            (dict(row) for row in time_quality if not row["test_pass"]),
            None,
        )

        summary = self._quality_summary(self.ahist, self.rhist)
        for key, value in summary.items():
            setattr(info, key, value)
        info.test_pass = info.test_pass_initial and info.test_pass_time

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
            eps0=self.eps0,
            epsr=self.epsr,
            epsa=self.epsa,
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
        burnup_rtol = float(assemble_d.get("burnup_rtol", 2.0e-2))

        # For each point in space.
        ii_json_list = list()
        f71_list = list()
        input_list = list()
        self.initialhm_list = list()
        for point in assemble_d["points"]:
            point_data = self._point_with_arpinfo_name(point)
            # Create the check input path.
            lib = Path(point_data["files"]["lib"])
            base = lib.stem
            check_input = self.check_path / base / (base + ".inp")

            # Save the list.
            hi_ii_json = self.work_path / point_data["files"]["ii_json"]
            lo_ii_json = check_input.with_suffix(".ii.json")
            f71_list.append(check_input.with_suffix(".f71"))
            ii_json_list.append((hi_ii_json, lo_ii_json))
            try:
                initialhm = float(point_data["history"]["initialhm"])
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
            self._require_history_within_library_burnup(
                point_data, base, nlib, burnup_rtol
            )

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
                **point_data,
                "name": self.name,
                "check": check,
                "lumped0d": self._lumped0d_for_origami_data(point_data),
                "_": {"env": self._env, "model": self._model},
            }
            if self.convergence_enabled:
                check_data["convergence_control"] = {
                    "nburn": nburn,
                    "nlib": nlib,
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
            stop_reason = self._convergence_stop_reason(
                previous_info,
                current_info,
                fixed_grid=self.nlib_start == self.nlib_max,
            )
            converged = bool(stop_reason)
            if previous_info is not None:
                self._set_convergence_deltas(
                    current_info, previous_info, "nlib"
                )
            current_info.nlib_converged = converged
            current_info.test_pass_nlib = converged
            current_info.nlib_convergence_stop = stop_reason or (
                "max" if nlib >= self.nlib_max else ""
            )
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
        if hasattr(nlib_info, "nlib_convergence_stop"):
            target_info.nlib_convergence_stop = nlib_info.nlib_convergence_stop
        if hasattr(nlib_info, "nlib_delta_q_r"):
            target_info.nlib_delta_q_r = nlib_info.nlib_delta_q_r
        if hasattr(nlib_info, "nlib_delta_q_ar"):
            target_info.nlib_delta_q_ar = nlib_info.nlib_delta_q_ar

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

            stop_reason = self._convergence_stop_reason(
                previous_info,
                current_info,
                fixed_grid=self.nburn_start == self.nburn_max,
            )
            converged = bool(stop_reason)
            if previous_info is not None:
                self._set_convergence_deltas(
                    current_info, previous_info, "nburn"
                )
            current_info.nburn_converged = converged
            current_info.test_pass_nburn = converged
            current_info.nburn_convergence_stop = stop_reason or (
                "max" if nburn >= self.nburn_max else ""
            )
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
            current_info.run_error = str(ve)

        if self.convergence_enabled:
            current_info.nlib = getattr(current_info, "nlib", self.nlib_start)
            current_info.nburn = getattr(current_info, "nburn", self.nburn_start)
            current_info.nlib_start = self.nlib_start
            current_info.nlib_max = self.nlib_max
            current_info.nburn_start = self.nburn_start
            current_info.nburn_max = self.nburn_max
            current_info.q_r_stop_criteria = self.q_r_stop_criteria
            current_info.q_ar_stop_criteria = self.q_ar_stop_criteria
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
            self._write_convergence_diagnostics(current_info)
            self._copy_final_check_outputs_to_nominal(current_info)
            current_info.convergence_status = self._convergence_status(current_info)
            current_info.failure_reasons = self._failure_reasons(current_info)
        else:
            for key in (
                "nlib",
                "nburn",
                "nlib_converged",
                "nburn_converged",
                "test_pass_nlib",
                "test_pass_nburn",
                "convergence_status",
            ):
                if hasattr(current_info, key):
                    delattr(current_info, key)
            current_info.failure_reasons = self._failure_reasons(current_info)

        return current_info
