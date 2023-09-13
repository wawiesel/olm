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
    """Create a zoned MOX assembly which preserves a desired average pu_frac including
        allowance for UO2+Gd2O3 pins.

    Default MOX zones from Mertyurek and Gauld NED 2016

        Ugur Mertyurek, Ian C. Gauld,
        Development of ORIGEN libraries for mixed oxide (MOX) fuel assembly designs,
        Nuclear Engineering and Design,
        Volume 297,
        2016,
        Pages 220-230,
        ISSN 0029-5493,
        https://doi.org/10.1016/j.nucengdes.2015.11.027.
        (https://www.sciencedirect.com/science/article/pii/S0029549315005592)

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
    for i in range(len(zone_pins)):
        wt_hm = x["info"]["hmo2_hm_frac"] / 100.0
        hm = wt_hm * zone_pins[i]
        putotal += hm * zone_pu_fracs[i]
        hmtotal += hm

    # If we have non-Pu bearing pins, UO2+Gd2O3.
    guox = {}
    if gd2o3_pins > 0:
        # This is approximate based on the uo2 that is combined with puo2,
        # to make MOX, not the UO2 combined with the Gd2O3 that we do not
        # pass in here.
        m_u = x["info"]["m_u"]
        m_o2 = x["info"]["m_o2"]
        m_uo2 = m_u + m_o2
        m_gd2o3 = 2 * 157.25 + 1.5 * m_o2
        wt_hm = (m_u) / (m_uo2 * (1.0 - gd2o3_wtpct) + gd2o3_wtpct * m_gd2o3)
        # Note does not increase Pu total
        hmtotal += wt_hm * gd2o3_pins
        guox["info"] = {"gd2o3_plus_uo2_hm_frac": wt_hm, "m_gd2o3": m_gd2o3}
        guox["uo2"] = x["uo2"]
        guox["gd2o3"] = {"dens_frac": gd2o3_wtpct / 100.0}
        guox["uo2"]["dens_frac"] = 1.0 - gd2o3_wtpct / 100.0

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
