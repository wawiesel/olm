"""
Module for contributed code that doesn't fit in the other places.

Everything should have doctests.

"""
import scale.olm.internal as internal
import scale.olm.complib as complib
import numpy as np


def parse_sfcompo_operating_history(input):
    """Parse the operating history format in SFCOMPO.

    Args:
        input: Either text or an open file an SFCOMPO operating history file.

    Returns:
        time (list[float]): Elapsed time in days.
        burnup (list[float]): Cumulative burnup in MWd/MTIHM.
        burnup_std (list[float]): Standard deviation of cumulative burnup.

    Examples:

        Initialize with sample data. Usually this data would come from reading a file.

        >>> text='''Elapsed days;Value;Point type;Uncertainty (%);Sigma
        ... 0;0 MW*d/tUi;HISTOGRAM;0;0
        ... 6.3;188.06 MW*d/tUi;HISTOGRAM;5.0;9.403
        ... 18.33;567.33 MW*d/tUi;HISTOGRAM;7;39.713
        ... 39.87;1246.44 MW*d/tUi;HISTOGRAM;8;99.715'''
        >>> time,burnup,burnup_std = parse_sfcompo_operating_history(text)
        >>> time
        [0.0, 6.3, 18.33, 39.87]
        >>> burnup
        [0.0, 188.06, 567.33, 1246.44]
        >>> burnup_std
        [0.0, 9.403, 39.713, 99.715]

        Initialize from a file.

        >>> from scale.olm.core import TempDir
        >>> td = TempDir()
        >>> op = 'operating_history.txt'
        >>> path = td.write_file(text,op)
        >>> with open(path, 'r') as f:
        ...     time,burnup,burnup_std = parse_sfcompo_operating_history(f)
        >>> time
        [0.0, 6.3, 18.33, 39.87]
        >>> burnup
        [0.0, 188.06, 567.33, 1246.44]
        >>> burnup_std
        [0.0, 9.403, 39.713, 99.715]

    """
    import csv
    from io import StringIO

    if isinstance(input, str):
        f = StringIO(input)
    else:
        f = input
    reader = csv.DictReader(f, delimiter=";")

    time = []
    burnup = []
    burnup_std = []
    bu_last = 0.0
    for row in reader:
        time.append(float(row["Elapsed days"]))
        bu = float(row["Value"].split(" ")[0])
        if bu < bu_last:
            internal.logger.warning(
                f"The cumulative burnup decreased from {bu_last} to {bu} which is impossible. Setting to {bu_last}."
            )
            bu = bu_last
        burnup.append(bu)
        burnup_std.append(float(row["Sigma"] or 0.0))
        bu_last = bu
    return time, burnup, burnup_std


def change_plot_font_size(ax, fontsize=14):
    """Change the plot font size."""
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
        + ax.get_legend().get_texts()
    ):
        item.set_fontsize(fontsize)


def sfcompo_guess_initial_mox(
    fiss_pu_frac,
    pu_frac,
    density=10.4,
    relmin=0.7,
    u235=0.231,
    am241=0.0,
    nbins=10,
    plot=False,
    complib_fun=complib.mox_ornltm2003_2,
):
    """Generate an initial mox guess based on what SFCOMPO lists.

    SFCOMPO lists the fissile Pu fraction and the Pu fraction. SCALE builds an interpolation
    just based on the Pu-239 fraction. This function takes what SFCOMPO has and builds
    an interpolatable conversion factor from a passed in MOX composition generator
    function. Setting plot=True may be useful to understand visually.

    The target composition is passed back using the same function.

    Examples:

        Search for fissile pu of 72.3% and pu fraction of 7.0%.

        >>> x = 	(fiss_pu_frac=72.3, pu_frac=7.0)
        >>> "{:.2f}".format(x['info']['pu_frac'])
        '7.00'

        >>> "{:.2f}".format(x['info']['pu239_frac'])
        '66.08'

        >>> "{:.2f}".format(x['info']['fiss_pu_frac'])
        '72.30'

        Check error is less than 0.01%.

        >>> abs(x['info']['fiss_pu_frac']/72.3-1)<1e-4
        True

        Here is an example with plotting turned on.

        .. plot::
                :include-source: True
                :show-source-link: False

                import scale.olm.contrib as contrib
                x = contrib.sfcompo_guess_initial_mox(fiss_pu_frac=72.3, pu_frac=7.0, plot=True)

    """
    import scipy as sp

    p9_list = np.linspace(fiss_pu_frac * relmin, fiss_pu_frac, nbins)
    fp_list = []
    uo2 = {"iso": {"u235": u235, "u236": 1e-10, "u234": 1e-10, "u238": 100 - u235}}
    for pu239_frac in p9_list:
        x = complib_fun(
            state={"pu239_frac": pu239_frac, "pu_frac": pu_frac},
            density=density,
            uo2=uo2,
            am241=1e-20,
        )
        iso = x["puo2"]["iso"]
        fp_list.append(iso["pu239"] + iso["pu241"])

    conv = np.asarray(p9_list) / np.asarray(fp_list)
    # fp_to_p9 = np.interp([fiss_pu_frac],fp_list,conv)[0]
    fp_to_p9 = sp.interpolate.PchipInterpolator(fp_list, conv)(fiss_pu_frac)

    target_p9 = fiss_pu_frac * fp_to_p9

    if plot:
        import matplotlib.pyplot as plt

        ax = plt.subplot()
        plt.plot(fp_list, 100 * conv, "-", marker=".")
        plt.xlabel(r"$\frac{\mathrm{fissile\;\;Pu}}{\mathrm{total\;\;Pu}}$ (%)")
        plt.ylabel(r"$\frac{{^{239}\mathrm{Pu}}}{\mathrm{fissle\;\;Pu}}$ (%)")
        plt.grid()
        plt.plot(
            [fiss_pu_frac],
            [100 * fp_to_p9],
            marker="o",
            markersize=10,
            markeredgecolor="red",
            markerfacecolor="white",
        )
        plt.legend(["relationship", "target"])
        change_plot_font_size(ax, 14)

    return complib_fun(
        state={"pu239_frac": target_p9, "pu_frac": pu_frac},
        density=density,
        uo2=uo2,
        am241=am241,
    )
