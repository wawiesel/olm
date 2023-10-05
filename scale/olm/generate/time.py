"""
Generate functions for time-dependent power histories. The preferred
output is a simple list of power levels and time at that power level using
the conventional power in MW/MTIHM and time in days.

A time generation function shall always receive the :code:`state` along with
any other parameters inside the input :code:`time` section.

"""


def constpower_burndata(state: dict[str, float], gwd_burnups: list[float]):
    """Return a list of powers and times assuming constant burnup.

    TODO: Instead of returning burndata dictionary, just return the list. Make
    this look less like a TRITON burndata specification.

    Args:

        state: state point data--only the "specific_power" key in MW/MTIHM units is
               used

        gwd_burnups: list of cumulative burnups in GWd/MTIHM

    Returns:

        dictionary with single "burndata" key with values a list of power/burn pairs
        as TRITON would expect for its burndata block

    Examples:

        >>> import scale.olm as olm
        >>> olm.generate.time.constpower_burndata(
        ...     state={"specific_power": 40},
        ...     gwd_burnups=[0,10,20]
        ... )
        {'burndata': [{'power': 40, 'burn': 250.0}, {'power': 40, 'burn': 250.0}, {'power': 40, 'burn': 250.0}]}

    """

    specific_power = state["specific_power"]

    # Calculate cumulative time to achieve each burnup.
    burnups = [float(x) * 1e3 for x in gwd_burnups]
    days = [burnup / float(specific_power) for burnup in burnups]

    # Check warnings and errors.
    if burnups[0] > 0:
        raise ValueError("Burnup step 0.0 GWd/MTHM must be included.")

    # Create the burndata block.
    burndata = []
    if len(days) > 1:
        for i in range(len(days) - 1):
            burndata.append({"power": specific_power, "burn": (days[i + 1] - days[i])})

        # Add one final step so that we can interpolate to the final requested burnup.
        burndata.append({"power": specific_power, "burn": (days[-1] - days[-2])})
    else:
        burndata.append({"power": specific_power, "burn": 0})

    return {"burndata": burndata}
