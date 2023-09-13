import scale.olm.complib as complib
import scale.olm.internal as internal
import scale.olm.core as core
import numpy as np
import math
from pathlib import Path
import json
import copy
import shutil


def constpower_burndata(state, gwd_burnups):
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


def full_hypercube(**states):
    """Generate all the permutations assuming a dense N-dimensional space."""
    dims = []
    axes = []
    for dim in states:
        axes.append(sorted(states[dim]))
        internal.logger.debug(f"Processing dimension '{dim}'")
        dims.append(dim)

    permutations = []
    grid = np.array(np.meshgrid(*axes)).T.reshape(-1, len(dims))
    for x in grid:
        y = dict()
        for i in range(len(dims)):
            y[dims[i]] = x[i]
        internal.logger.debug(f"Generated permutation '{y}'")
        permutations.append(y)

    return permutations


def scipy_interp(state_var: str, data_pairs, state, method: str = "linear"):
    import scipy as sp

    x0 = state[state_var]
    x_list = []
    y_list = []
    for xy in data_pairs:
        x_list.append(xy[0])
        y_list.append(xy[1])

    y0 = None
    if method.lower() == "pchip":
        y0 = sp.interpolate.pchip_interpolate(x_list, y_list, x0)
    elif method.lower() == "linear":
        y0 = sp.interpolate.interp1d(x_list, y_list)(x0)
    else:
        raise ValueError(f"scipy_interp method={method} must be one of: PCHIP, LINEAR")

    internal.logger.debug(
        "scipy_interp for method={method}", x0=x0, y0=y0, x=x_list, y=y_list
    )

    return float(y0)


def jt_expander(template, static, states, comp, time, _model, _env, dynamic=None):
    """First expand the state to all the individual state combinations, then calculate the
    times and the compositions which may require state. The static just pass through."""

    internal.logger.info(f"Generating with scale.olm.jt_expander ...")

    # Handle parameters.
    static2 = internal._fn_redirect(**static)

    # Generate a list of states from the state specification.
    states2 = internal._fn_redirect(**states)

    # Useful paths.
    work_path = Path(_env["work_dir"])
    generate_path = work_path / "perms"

    # Load the template file.
    template_path = Path(_env["config_file"]).parent / template
    with open(template_path, "r") as f:
        template_text = f.read()

    internal.logger.info(
        "Expanding into permutations", template=str(template_path), nperms=len(states2)
    )

    # Create all the permutation information.
    perms2 = []
    i = 0
    td = core.TempDir()
    for state2 in states2:
        # For each state, generate the compositions.
        comp2 = {}
        if "_type" in comp:
            comp2 = internal._fn_redirect(**comp, state=state2)
        else:
            for k, v in comp.items():
                comp2[k] = internal._fn_redirect(**v, state=state2)

        # Handle dynamic (state-dependent) parameters.
        dynamic2 = {}
        if dynamic:
            for k, v in dynamic.items():
                dynamic2[k] = internal._fn_redirect(**v, state=state2)

        # For each state, generate a time list.
        time2 = internal._fn_redirect(**time, state=state2)

        # Generate all data.
        data = {
            "static": static2,
            "comp": comp2,
            "time": time2,
            "state": state2,
            "dynamic": dynamic2,
        }

        # Write data to a temporary file to get a hash of the contents.
        tf = td.write_file(json.dumps(data, indent=4), "temp.json")
        data_hash = core.FileHasher(tf).id

        # Save some info.
        input_path = generate_path / data_hash / ("model_" + data_hash[-6:] + ".inp")
        input_file = str(input_path.relative_to(work_path))
        data["input_file"] = input_file
        data["file"] = data["input_file"]  # deprecated alias
        data_path = input_path.parent / "data.olm.json"
        i += 1
        data_file = str(data_path.relative_to(work_path))
        data["_"] = {"model": _model, "data_hash": data_hash, "data_file": data_file}

        # Write the data file in the actual directory with input and hash added. This
        # is mainly so a user can see the data that is available for template expansion
        # beside a copy of the template.
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "w") as f:
            json.dump(data, f, indent=4)

        # Expand the template and write the input to disk.
        internal.logger.info("Writing permutation", index=i, input_file=input_file)
        filled_text = core.TemplateManager.expand_text(
            template_text, data, src_path=str(template_path)
        )
        with open(input_path, "w") as f:
            f.write(filled_text)

        # Return the final thing in a permutations list.
        perms2.append(data)

    internal.logger.info(f"Finished generating with scale.olm.jt_expander!")

    return {"work_dir": _env["work_dir"], "perms": perms2, "static": static2}
