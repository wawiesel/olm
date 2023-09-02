import scale.olm.common as common
import scale.olm.core as core
import numpy as np
import math
from pathlib import Path
import json
import copy


def __iso_uo2(u234, u235, u236):
    """Tiny helper to pass u234,u235,u238 through to create map and recalc u238."""
    return {
        "u235": u235,
        "u238": 100.0 - u234 - u235 - u236,
        "u234": u234,
        "u236": u236,
    }


def comp_uo2_simple(state, density=0):
    """Example of a simple enrichment formula."""
    enrichment = float(state["enrichment"])
    if enrichment > 100:
        raise ValueError(f"enrichment={enrichment} must be >=0 and <=100")
    return {
        "density": density,
        "uo2": {"iso": __iso_uo2(u234=1.0e-20, u235=enrichment, u236=1.0e-20)},
    }


def comp_uo2_vera(state, density=0):
    """Enrichment formula from:
    Andrew T. Godfrey. VERA core physics benchmark progression problem specifications.
    Consortium for Advanced Simulation of LWRs, 2014.
    """

    enrichment = float(state["enrichment"])
    if enrichment > 10:
        raise ValueError(f"enrichment={enrichment} must be <=10% to use comp_uo2_vera")

    return {
        "density": density,
        "uo2": {
            "iso": __iso_uo2(
                u234=0.007731 * (enrichment**1.0837),
                u235=enrichment,
                u236=0.0046 * enrichment,
            )
        },
    }


def comp_uo2_nuregcr5625(state, density=0):
    """Enrichment formula from NUREG/CR-5625."""

    enrichment = float(state["enrichment"])
    if enrichment > 20:
        raise ValueError(
            f"enrichment={enrichment} must be <=20% to use comp_uo2_nuregcr5625"
        )

    return {
        "density": density,
        "uo2": {
            "iso": __iso_uo2(
                u234=0.0089 * enrichment,
                u235=enrichment,
                u236=0.0046 * enrichment,
            )
        },
    }


def comp_mox_ornltm2003_2(state, density, uo2, am241):
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
    x, norm_x = common.renormalize_wtpt(x0, 100.0 - pu239 - am241)
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
    comp = common.calculate_hm_oxide_breakdown(x)

    # Fill in additional information.
    comp["info"] = common.approximate_hm_info(comp)

    # Pass through density.
    comp["density"] = density

    return comp


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
        core.logger.debug(f"Processing dimension '{dim}'")
        dims.append(dim)

    permutations = []
    grid = np.array(np.meshgrid(*axes)).T.reshape(-1, len(dims))
    for x in grid:
        y = dict()
        for i in range(len(dims)):
            y[dims[i]] = x[i]
        core.logger.debug(f"Generated permutation '{y}'")
        permutations.append(y)

    return permutations


def expander(model, template, params, states, comp, time):
    """First expand the state to all the individual state combinations, then calculate the
    times and the compositions which may require state. The params just pass through."""

    core.logger.info(f"Generating with scale.olm.expander ...")

    # Handle parameters.
    params2 = common.fn_redirect(**params)

    # Generate a list of states from the state specification.
    states2 = common.fn_redirect(**states)

    # Create a formatting statement for the files.
    nstates = len(states2)
    core.logger.info(
        f"Initiating expansion of template file={template} into {nstates} permutations ..."
    )
    n = int(1 + math.log10(nstates))
    work_dir = model["work_dir"]
    fmt = f"{work_dir}/perm{{0:0{n}d}}/perm{{0:0{n}d}}.inp"

    # Load the template file.
    template_file = Path(model["dir"]) / template
    with open(template_file, "r") as f:
        template_text = f.read()

    # Create all the permutation information.
    perms2 = []
    i = 0
    for state2 in states2:
        # For each state, generate the compositions.
        comp2 = {}
        for k, v in comp.items():
            comp2[k] = common.fn_redirect(**comp[k], state=state2)

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

    core.logger.info(f"Finished scale.olm.expander!")

    return {"work_dir": str(work_dir), "perms": perms2, "params": params2}
