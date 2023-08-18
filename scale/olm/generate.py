import scale.olm.common as common
import numpy as np
import math
from pathlib import Path


def fuelcomp_uox_simple(state, nuclide_prefix=""):
    """Example of a simple enrichment formula."""
    enrichment = state["enrichment"]
    data = {
        "u235": enrichment,
        "u238": 100.0 - float(enrichment),
        "u234": 1.0e-20,
        "u236": 1.0e-20,
    }

    # Rename keys to add prefix.
    for i in data.copy():
        data[nuclide_prefix + i] = data.pop(i)

    return data


def triton_constpower_burndata(state, gwd_burnups):
    """Return a list of powers and times."""

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

    return {"work_dir": str(work_dir), "perms": perms2}
