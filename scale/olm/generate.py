import scale.olm.common as common
import numpy as np
import math
from pathlib import Path
import json
import copy


def __fuelcomp_uox(u234, u235, u236):
    """Tiny helper to pass u234,u235,u238 through to create map and recalc u238."""
    return {
        "u235": u235,
        "u238": 100.0 - u234 - u235 - u236,
        "u234": u234,
        "u236": u236,
    }


def __apply_prefix(data, nuclide_prefix):
    """Rename keys to add prefix."""
    for i in data.copy():
        data[nuclide_prefix + i] = data.pop(i)
    return data


def fuelcomp_uox_simple(state, nuclide_prefix=""):
    """Example of a simple enrichment formula."""
    enrichment = float(state["enrichment"])
    if enrichment > 100:
        raise ValueError(f"enrichment={enrichment} must be >=0 and <=100")
    data = __fuelcomp_uox(u234=1.0e-20, u235=enrichment, u236=1.0e-20)
    return __apply_prefix(data, nuclide_prefix)


def fuelcomp_uox_vera(state, nuclide_prefix=""):
    """Enrichment formula from:
    Andrew T. Godfrey. VERA core physics benchmark progression problem specifications.
    Consortium for Advanced Simulation of LWRs, 2014.
    """

    enrichment = float(state["enrichment"])
    if enrichment > 10:
        raise ValueError(
            f"enrichment={enrichment} must be <=10% to use fuelcomp_uox_vera"
        )

    data = __fuelcomp_uox(
        u234=0.007731 * (enrichment**1.0837),
        u235=enrichment,
        u236=0.0046 * enrichment,
    )
    return __apply_prefix(data, nuclide_prefix)


def fuelcomp_uox_nuregcr5625(state, nuclide_prefix=""):
    """Enrichment formula from NUREG/CR-5625."""

    enrichment = float(state["enrichment"])
    if enrichment > 20:
        raise ValueError(
            f"enrichment={enrichment} must be <=20% to use fuelcomp_uox_nuregcr5625"
        )

    data = __fuelcomp_uox(
        u234=0.0089 * enrichment,
        u235=enrichment,
        u236=0.0046 * enrichment,
    )
    return __apply_prefix(data, nuclide_prefix)


def __renormalize_wtpt(wtpt0, sum0, key_filter=""):
    """Renormalize to sum0 any keys matching filter."""
    # Calculate the sum of filtered elements. Copy into return value.
    wtpt = {}
    sum = 0.0
    for k, v in wtpt0.items():
        if k.startswith(key_filter):
            sum += v
            wtpt[k] = v

    # Renormalize.
    norm = sum / sum0
    for k in wtpt:
        wtpt[k] /= norm
    return wtpt, norm


def grams_per_mol(iso_wts):
    """Calculate the grams per mole of a weight percent mixture."""
    import re

    m_data = {"am241": 241.0568}

    m = 0.0
    for iso, wt in iso_wts.items():
        m_default = re.sub("^[a-z]+", "", iso)
        m_iso = m_data.get(iso, float(m_default))
        m += (wt / 100.0) / m_iso

    return 1.0 / m


def element_mass(element):
    return {"o": 15.9994, "gd": 157.25}[element]


def comp_mox_ornltm2003_2(state, density, uo2=None, am241=0):
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
    x, norm_x = __renormalize_wtpt(x0, 100.0 - pu239 - am241)
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
        y = fuelcomp_uox_nuregcr5625(state={"enrichment": 0.24})
    u_pct = 100.0 - pu_plus_am_pct
    for k in y:
        y[k] *= u_pct / 100.0

    # At this point we can combine the vectors into one heavy metal vector.
    x.update(y)
    hm_iso, hm_norm = __renormalize_wtpt(x, 100.0)
    hm_mass = grams_per_mol(hm_iso)

    # Split into elements.
    pu_iso, pu_norm = __renormalize_wtpt(hm_iso, 100.0, "pu")
    pu_mass = grams_per_mol(pu_iso)

    am_iso, am_norm = __renormalize_wtpt(hm_iso, 100.0, "am")
    am_mass = grams_per_mol(am_iso)

    u_iso, u_norm = __renormalize_wtpt(hm_iso, 100.0, "u")
    u_mass = grams_per_mol(u_iso)

    # Calculate heavy metal fractions of oxide (approximate).
    o2_mass = 2 * element_mass("o")
    puo2_hm_frac = pu_mass / (pu_mass + o2_mass)
    amo2_hm_frac = am_mass / (am_mass + o2_mass)
    uo2_hm_frac = u_mass / (u_mass + o2_mass)
    hmo2_hm_frac = hm_mass / (hm_mass + o2_mass)

    # Assume the density fraction is proportional to the heavy metal fraction
    # which was returned in the "norm".
    uo2_dens_frac = u_norm
    puo2_dens_frac = pu_norm
    amo2_dens_frac = am_norm

    return {
        "uo2": {"iso": u_iso, "dens_frac": uo2_dens_frac},
        "puo2": {"iso": pu_iso, "dens_frac": puo2_dens_frac},
        "amo2": {"iso": am_iso, "dens_frac": amo2_dens_frac},
        "am241": am241,
        "info": {
            "o2_mass": o2_mass,
            "hm_iso": hm_iso,
            "u_mass": u_mass,
            "am_mass": am_mass,
            "pu_mass": pu_mass,
            "hm_mass": hm_mass,
            "hm_norm": hm_norm,
            "puo2_hm_frac": puo2_hm_frac,
            "amo2_hm_frac": amo2_hm_frac,
            "uo2_hm_frac": uo2_hm_frac,
            "hmo2_hm_frac": hmo2_hm_frac,
        },
        "density": density,
    }


def comp_mox_zones_2023(
    state,
    zone_names,
    zone_pins,
    density,
    uo2=None,
    zone_pu_fracs=None,
    am241=0.0,
    gd2o3_pins=0,
    gd2o3_wtpct=0.0,
    nuclide_prefix="",
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

    # Get a base MOX composition to calculate Pu/HM ratios.
    x = comp_mox_ornltm2003_2(state, density, uo2=uo2, am241=am241)
    putotal = 0
    hmtotal = 0
    for i in range(len(zone_pins)):
        wt_hm = x["info"]["hmo2_hm_frac"] / 100.0
        hm = wt_hm * zone_pins[i]
        putotal += hm * zone_pu_fracs[i]
        hmtotal += hm

    # If we have non-Pu bearing pins, UO2+Gd2O3.
    if gd2o3_pins > 0:
        # This is approximate based on the uo2 that is combined with puo2,
        # to make MOX, not the UO2 combined with the Gd2O3 that we do not
        # pass in here.
        uo2_mass = x["info"]["u_mass"] + x["info"]["o2_mass"]
        gd2o3_mass = 2 * element_mass("gd") + 3 * element_mass("o")
        wt_hm = (uo2_mass) / (uo2_mass + gd2o3_mass)
        # Note does not increase Pu total
        hmtotal += wt_hm * gd2o3_pins

    data = {}

    # We want to match the Pu/HM total over the assembly which should be
    # state['pu_frac'] but it will not be.
    multiplier = state["pu_frac"] / (putotal / hmtotal)
    for i in range(len(zone_pins)):
        zone_pu_fracs[i] *= multiplier
        state0 = copy.deepcopy(state)
        state0["pu_frac"] = zone_pu_fracs[i]
        print(state0)
        data[zone_names[i]] = comp_mox_ornltm2003_2(state0, density, uo2, am241)

    return data


def triton_constpower_burndata(state, gwd_burnups):
    """Return a list of powers and times assuming constant burnup."""

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


def all_permutations(**states):
    """Generate all the permutations assuming a dense N-dimensional space."""
    dims = []
    axes = []
    for dim in states:
        axes.append(sorted(states[dim]))
        common.logger.info(f"Processing dimension '{dim}'")
        dims.append(dim)

    permutations = []
    grid = np.array(np.meshgrid(*axes)).T.reshape(-1, len(dims))
    for x in grid:
        y = dict()
        for i in range(len(dims)):
            y[dims[i]] = x[i]
        common.logger.info(f"Generated permutation '{y}'")
        permutations.append(y)

    return permutations


def expander(model, template, params, states, comp, time):
    """First expand the state to all the individual state combinations, then calculate the
    times and the compositions which may require state. The params just pass through."""

    # Handle parameters.
    params2 = common.fn_redirect(**params)

    # Generate a list of states from the state specification.
    states2 = common.fn_redirect(**states)

    # Create a formatting statement for the files.
    n = int(1 + math.log10(len(states2)))
    work_dir = model["work_dir"]
    fmt = f"{work_dir}/perm{{0:0{n}d}}/perm{{0:0{n}d}}.inp"

    # Load the template file.
    with open(Path(model["dir"]) / template, "r") as f:
        template_text = f.read()

    # Make a copy of the base compositions and handle the single object with implicit
    # name fuel.
    if "_type" in comp:
        comp0 = {"fuel": comp}
        common.logger.info(
            "Assuming single composition named 'fuel'. Use as comp.fuel.* in template file."
        )
    else:
        comp0 = comp

    # Create all the permutation information.
    perms2 = []
    i = 0
    for state2 in states2:
        # For each state, generate the compositions.
        comp2 = {}
        for k, v in comp0.items():
            comp2[k] = common.fn_redirect(**comp0[k], state=state2)

        # For each state, generate a time list.
        time2 = common.fn_redirect(**time, state=state2)

        # Generate this file name.
        file = Path(fmt.format(i))
        i += 1

        # Generate all data.
        data = {
            "file": str(file.relative_to(work_dir)),
            "params": params2,
            "comp": comp2,
            "time": time2,
            "state": state2,
        }

        filled_text = common.expand_template(template_text, data)

        # Write the file.
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "w") as f:
            f.write(filled_text)

        # Save the data.
        perms2.append(data)

    return {"work_dir": str(work_dir), "perms": perms2, "params": params2}


# Adapted from getMoxContents in SLIG; original MOX formulae from ORNL/TM-2003/2
def getMoxContents(fuelcomp, nuclide_prefix=""):
    wtpt_pu = fuelcomp[nuclide_prefix + "pu"]
    wptt_pu239 = fuelcomp[nuclide_prefix + "pu239"]
    ratios = []

    # pu zoning calculation
    if fuelcomp["pins_gd"] > 0:
        raise ValueError("Gd pins not implemented")
        ratios = [1.0, 0.75, 0.5, 0.3]  # inner/inside edge/edge/corner
    else:
        ratios = [1.0, 0.9, 0.68, 0.5]  # inner/inside edge/edge/corner

    Au = 238.0289  # g/mol
    Ao = 15.999  # g/mol

    denom = 0
    for i in range(len(ratios)):
        denom += ratios[i] * fuelcomp["pins_zone"][i]

    Ahm = ((100 - wtpt_pu) * Au + wtpt_pu * fuelcomp["avgA_pu"]) / 100.0
    hmInOnePin = Ahm / (Ahm + 2 * Ao) * fuelcomp["fuel_density"]
    hmInPins = hmInOnePin * (
        sum(fuelcomp["pins_zone"]) + fuelcomp["wtpt_gd"] * fuelcomp["pins_gd"]
    )
    puInPins = hmInPins * wtpt_pu / 100.0
    innerPuContent = 100.0 / hmInOnePin * puInPins / denom
    puContents = [innerPuContent * x for x in ratios]

    order = ["inner", "inedge", "edge", "corner"]
    pu_regions = dict(zip(order, puContents))

    return __apply_prefix(pu_regions, nuclide_prefix)
