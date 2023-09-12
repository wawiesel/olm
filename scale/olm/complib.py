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


def mox_ornltm2003_2(state, density, uo2, am241):
    """MOX isotopic vector calculation from ORNL/TM-2003/2, Sect. 3.2.2.1"""

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
    y = copy.deepcopy(uo2["iso"])
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
