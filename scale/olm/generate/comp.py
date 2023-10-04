"""
This module contains composition creation functions.

They should return a standard format for output so they can easily
be used interchangeably (somewhat).

"""
import scale.olm.core as core
import numpy as np
import math
from pathlib import Path
import json
import copy


def _iso_uo2(u234, u235, u236):
    """Tiny helper to pass u234,u235,u238 through to create map and recalc u238."""
    return {
        "u235": u235,
        "u238": 100.0 - u234 - u235 - u236,
        "u234": u234,
        "u236": u236,
    }


def uo2_simple(state, density=0):
    """Example of a simple enrichment formula."""
    enrichment = float(state["enrichment"])
    if enrichment > 100:
        raise ValueError(f"enrichment={enrichment} must be >=0 and <=100")
    return {
        "density": density,
        "uo2": {"iso": _iso_uo2(u234=1.0e-20, u235=enrichment, u236=1.0e-20)},
        "_input": {"state": state, "density": density},
    }


def uo2_vera(state, density=0):
    """Enrichment formula from:
    Andrew T. Godfrey. VERA core physics benchmark progression problem specifications.
    Consortium for Advanced Simulation of LWRs, 2014.
    """

    enrichment = float(state["enrichment"])
    if enrichment > 10:
        raise ValueError(f"enrichment={enrichment} must be <=10% to use uo2_vera")

    return {
        "density": density,
        "uo2": {
            "iso": _iso_uo2(
                u234=0.007731 * (enrichment**1.0837),
                u235=enrichment,
                u236=0.0046 * enrichment,
            )
        },
        "_input": {"state": state, "density": density},
    }


def uo2_nuregcr5625(state, density=0):
    """Enrichment formula from NUREG/CR-5625."""

    enrichment = float(state["enrichment"])
    if enrichment > 20:
        raise ValueError(
            f"enrichment={enrichment} must be <=20% to use uo2_nuregcr5625"
        )

    return {
        "density": density,
        "uo2": {
            "iso": _iso_uo2(
                u234=0.0089 * enrichment,
                u235=enrichment,
                u236=0.0046 * enrichment,
            )
        },
        "_input": {"state": state, "density": density},
    }


def mox_ornltm2003_2(state, density, uo2=None, am241=0):
    """MOX isotopic vector calculation from ORNL/TM-2003/2, Sect. 3.2.2.1"""

    # Set to something small to avoid unnecessary extra logic below.
    if am241 < 1e-20:
        am241 = 1e-20

    # Calculate pu vector as per formula. Note that the pu239_frac is by definition:
    # pu239/(pu+am) and the Am comes in from user input.
    pu239 = float(state["pu239_frac"])
    if not (pu239 > 0.0) and (pu239 < 100.0):
        raise ValueError(f"pu239 percentage={pu239} must be between 0 and 100.")
    pu238 = 0.0045678 * pu239**2 - 0.66370 * pu239 + 24.941
    pu240 = -0.0113290 * pu239**2 + 1.02710 * pu239 + 4.7929
    pu241 = 0.0018630 * pu239**2 - 0.42787 * pu239 + 26.355
    pu242 = 0.0048985 * pu239**2 - 0.93553 * pu239 + 43.911
    x0 = {"pu238": pu238, "pu240": pu240, "pu241": pu241, "pu242": pu242}
    x, norm_x = core.CompositionManager.renormalize_wtpt(x0, 100.0 - pu239 - am241)
    x["pu239"] = pu239
    x["am241"] = am241

    # Scale by relative weight percent of Pu+Am and U.
    pu_plus_am_pct = float(state["pu_frac"])
    for k in x:
        x[k] *= pu_plus_am_pct / 100.0

    # Get U isotopes and scale to remaining weight percent.
    if uo2:
        y = copy.deepcopy(uo2["iso"])
    else:
        y = uo2_nuregcr5625(state={"enrichment": 0.24})["uo2"]["iso"]
    u_pct = 100.0 - pu_plus_am_pct
    for k in y:
        y[k] *= u_pct / 100.0

    # At this point we can combine the vectors into one heavy metal vector.
    x.update(y)

    # First part of calculation.
    comp = core.CompositionManager.calculate_hm_oxide_breakdown(x)

    # Fill in additional information.
    comp["info"] = core.CompositionManager.approximate_hm_info(comp)

    # Pass through density.
    comp["density"] = density

    # Copy the inputs.
    comp["_input"] = {"state": state, "density": density, "uo2": uo2, "am241": am241}

    return comp


def mox_multizone_2023(
    state,
    zone_names,
    zone_pins,
    density,
    uo2=None,
    zone_pu_fracs=None,
    am241=0.0,
    gd2o3_pins=0,
    gd2o3_wtpct=0.0,
):
    """Create compositions for a zoned MOX assembly with a desired average plutonium fraction.

    This function has handling for additional non-plutonium bearing pins with
    UO2+Gd2O3.

    Built-in zone plutonium fractions for BWR2016 and PWR2016 are from this publication.

    Ugur Mertyurek, Ian C. Gauld,
    Development of ORIGEN libraries for mixed oxide (MOX) fuel assembly designs,
    Nuclear Engineering and Design,
    Volume 297,2016,Pages 220-230,ISSN 0029-5493,https://doi.org/10.1016/j.nucengdes.2015.11.027.
    (https://www.sciencedirect.com/science/article/pii/S0029549315005592)

    Args:

        state: Data about the state for which this composition should be created,
            only uses the following fields.

            - 'pu_frac' fraction of plutonium to heavy metal (as a percentage)
            - 'pu239_frac' fraction of pu239 to plutonium (as a percentage)

        zone_names: Names of zonewise compositions to output e.g. ['inner','edge']
            OR a built-in composition name. Built-ins are:

            - 'PWR2016' a four-zone distribution for a generic PWR with an 'inner',
              'iedge' (inner edge), 'edge', and 'corner' zone with the zone plutonium
              fractions varying according the 2016 Mertyurek and Gauld paper.
            - 'BWR2016' a four-zone distribution for a generic BWR an 'inner',
              'iedge' (inner edge), 'edge', and 'corner' zone with the zone plutonium
              fractions varying according the 2016 Mertyurek and Gauld paper.

        zone_pins: Number of pins in each zone (assumed to have same heavy metal fraction).
            E.g. if zone_names=['inner','edge'], zone_pins=[100,44] would indicate there
            are 100 inner pins and 44 edge pins so that the relevant heavy metal ratios
            are preserved to renormalize to the assembly-average target of state['pu_frac'].

        density: Density of the fuel. This is basically a pass-through option and not
            used in the calculation itself.

        uo2: A uranium oxide composition to use for the uo2 part of the MOX.

        zone_pu_fracs: The zone plutonium fractions to use, associated with each of the
            zone_names. For example, zone_names=['inner','edge'] could have zone_pu_fracs=[1.0,0.8]
            to indicate the the 'edge' zone should have 80% of the inner zone plutonium
            fraction, renormalized so that the assembly average is what was supplied in
            state['pu_frac'].

        am241: The weight percent in Am241 content. Typically this is not necessary because
            there will be a decay calculation performed to account for the buildup of Am241.

        gd2o3_pins: Number of pins of Gd2O3 and UO2 which are not contributing to the
            assembly-average plutonium fraction. Therefore, if the assembly had 144 MOX
            pins and 25 non-MOX pins then gd2o3_pins=25 would make sure that heavy metal
            content is taken into account when renormalizing to the target state['pu_frac'].

        gd2o3_wtpct: If the gd2o3_wtpct>0.0, then the heavy metal is reduced accordingly.

    Returns:

        dict: Dictionary of compositions by name. Internal to each composition is a
            'uo2' and 'puo2' dictionary which includes details to be used in creating
            a mixed-oxide composition with SCALE.

    Examples:

    Here is a basic example with two zones.

    >>> x=mox_multizone_2023(state={"pu239_frac": 70.0, "pu_frac": 5.0},
    ... zone_names=['inner','edge'],
    ... density=10.38,
    ... zone_pins=[100,44],
    ... zone_pu_fracs=[1.0,0.8])

    Let's define a simple printing function so that we don't see so many digits in
    the output and it is formatted consistently.

    >>> def prn(x):
    ...     import json
    ...     print( json.dumps(json.loads(json.dumps(x),
    ...         parse_float=lambda x: round(float(x), 4)),
    ...         indent=4, sort_keys=True) )

    For the inner zone we will have slightly higher density fraction that 5.0% because
    of the zone weighting of 1.0 versus 0.8 for the edge.

    >>> prn(x['inner']['puo2'])
    {
        "dens_frac": 0.0533,
        "iso": {
            "pu238": 0.8642,
            "pu239": 70.0,
            "pu240": 21.1768,
            "pu241": 5.5325,
            "pu242": 2.4264
        }
    }

    Note for the edge the isotopic distribution is identical, but the density fraction
    is different.

    >>> prn(x['edge']['puo2'])
    {
        "dens_frac": 0.0426,
        "iso": {
            "pu238": 0.8642,
            "pu239": 70.0,
            "pu240": 21.1768,
            "pu241": 5.5325,
            "pu242": 2.4264
        }
    }

    We should be very close to 5% with this simple reconstruction.

    >>> 100*(x['inner']['puo2']['dens_frac']*100 + x['edge']['puo2']['dens_frac']*44 ) / 144
    5.0

    """
    if isinstance(zone_names, str):
        if zone_names == "BWR2016":
            zone_pu_fracs = [1.0, 0.75, 0.50, 0.30]
        elif zone_names == "PWR2016":
            zone_pu_fracs = [1.0, 0.90, 0.68, 0.50]
        else:
            raise ValueError(f"zone_names={zone_names} must be BWR2016/PWR2016")
        zone_names = ["inner", "iedge", "edge", "corner"]

    assert len(zone_pu_fracs) == len(zone_names)
    assert len(zone_pu_fracs) == len(zone_pins)
    data = {}

    # Get a base MOX composition to calculate Pu/HM ratios.
    x = mox_ornltm2003_2(state, density, uo2=uo2, am241=am241)
    putotal = 0
    hmtotal = 0
    hm_frac = x["info"]["hmo2_hm_frac"] / 100.0
    for i in range(len(zone_pins)):
        hm = hm_frac * zone_pins[i]
        putotal += hm * zone_pu_fracs[i]
        hmtotal += hm

    # If we have non-Pu bearing pins, UO2+Gd2O3.
    guox = {}
    if gd2o3_pins > 0:
        # This is approximate based on the uo2 that is combined with puo2,
        # to make MOX, not the UO2 combined with the Gd2O3 that we do not
        # pass in here.
        hm_frac = x["info"]["uo2_hm_frac"]
        gd_frac = gd2o3_wtpct / 100.0
        # Note does not increase Pu total
        hmtotal += hm_frac * (1 - gd_frac) * gd2o3_pins
        guox["info"] = {"uo2_hm_frac": hm_frac}
        guox["uo2"] = x["uo2"]
        guox["uo2"]["dens_frac"] = 1.0 - gd_frac
        guox["gd2o3"] = {"dens_frac": gd_frac}

    # We want to match the Pu/HM total over the assembly which should be
    # state['pu_frac'] but it will not be.
    multiplier = state["pu_frac"] / (putotal / hmtotal)

    data = {
        "_zone": {
            "zone_pu_fracs": zone_pu_fracs,
            "zone_names": zone_names,
            "zone_pins": zone_pins,
            "hmtotal": hmtotal,
            "putotal": putotal,
            "multiplier": multiplier,
        },
        "guox": guox,
    }

    # Accumulate the data.
    for i in range(len(zone_pins)):
        zone_pu_fracs[i] *= multiplier
        state0 = copy.deepcopy(state)
        state0["pu_frac"] = zone_pu_fracs[i]
        data[zone_names[i]] = mox_ornltm2003_2(state0, density, uo2, am241)

    return data
