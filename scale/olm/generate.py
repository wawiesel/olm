import scale.olm.common as common
import numpy as np
import math
from pathlib import Path


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


def fuelcomp_mox_ornltm2003_2(
    state, pins_zone, density_fuel=10.4, pins_gd=0, pct_gd=0.0, nuclide_prefix=""
):
    """MOX isotopic vector calculation from ORNL/TM-2003/2, Sect. 3.2.2.1"""

    data = {"pu239": float(state[nuclide_prefix + "pu239"])}
    assert (data["pu239"] > 0.0) and (data["pu239"] < 100.0)

    data["pu238"] = 0.0045678 * data["pu239"] ** 2 - 0.66370 * data["pu239"] + 24.941
    data["pu240"] = -0.0113290 * data["pu239"] ** 2 + 1.02710 * data["pu239"] + 4.7929
    data["pu241"] = 0.0018630 * data["pu239"] ** 2 - 0.42787 * data["pu239"] + 26.355
    data["pu242"] = 0.0048985 * data["pu239"] ** 2 - 0.93553 * data["pu239"] + 43.911

    avgApu = (
        data["pu238"] * 238.0
        + data["pu239"] * 239.0
        + data["pu240"] * 240.0
        + data["pu241"] * 241.0
        + data["pu242"] * 242.0
    )

    data["pu"] = state[nuclide_prefix + "pu"]
    data["gd"] = pct_gd
    data = __apply_prefix(data, nuclide_prefix)

    data["avgA_pu"] = avgApu
    data["pins_zone"] = pins_zone
    data["pins_gd"] = pins_gd
    data["density_fuel"] = density_fuel
    data.update(getMoxContents(data, nuclide_prefix))
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


def expander(model, template, params, states, fuelcomp, time):
    """First expand the state to all the individual state combinations, then calculate the
    times and the compositions which may require state. The params just pass through."""

    # Handle parameters.
    params2 = common.fn_redirect(params)

    # Generate a list of states from the state specification.
    perms = common.fn_redirect(states)

    # Create a formatting statement for the files.
    n = int(1 + math.log10(len(perms)))
    work_dir = model["work_dir"]
    fmt = f"{work_dir}/perm{{0:0{n}d}}/perm{{0:0{n}d}}.inp"

    # Load the template file.
    with open(Path(model["dir"]) / template, "r") as f:
        template_text = f.read()

    # Create all the state information.
    perms2 = []
    i = 0
    for perm in perms:
        # For each state, generate a fuel composition.
        fuelcomp2 = fuelcomp.copy()
        fuelcomp2["state"] = perm
        fuelcomp2 = common.fn_redirect(fuelcomp2)

        # For each state, generate a time list.
        time2 = time.copy()
        time2["state"] = perm
        time2 = common.fn_redirect(time2)

        # Generate this file name.
        file = Path(fmt.format(i))
        i += 1

        # Generate all data.
        data = {
            "file": str(file.relative_to(work_dir)),
            "params": params,
            "fuelcomp": fuelcomp2,
            "time": time2,
            "state": perm,
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
    hmInOnePin = Ahm / (Ahm + 2 * Ao) * fuelcomp["density_fuel"]
    hmInPins = hmInOnePin * (
        sum(fuelcomp["pins_zone"]) + fuelcomp["wtpt_gd"] * fuelcomp["pins_gd"]
    )
    puInPins = hmInPins * wtpt_pu / 100.0
    innerPuContent = 100.0 / hmInOnePin * puInPins / denom
    puContents = [innerPuContent * x for x in ratios]

    order = ["inner", "inedge", "edge", "corner"]
    pu_regions = dict(zip(order, puContents))

    return __apply_prefix(pu_regions, nuclide_prefix)
