import scale.olm.common as common
import os
import json
from pathlib import Path
import copy
import shutil
import numpy as np


def arpdata_txt(model, fuel_type, suffix, dim_map):
    """Build an ORIGEN reactor library in arpdata.txt format."""

    # Get working directory.
    work_dir = Path(model["work_dir"])

    # Get list of files by using the generate.json output and changing the
    # suffix to the expected library file.
    generate_json = work_dir / "generate.json"
    with open(generate_json, "r") as f:
        generate = json.load(f)
    lib_list = list()
    input_list = list()
    for perm in generate["perms"]:
        input_list.append(perm["file"])
        # Convert from .inp to expected suffix.
        lib = work_dir / Path(perm["file"])
        lib = lib.with_suffix(suffix)
        if not lib.exists():
            common.logger.error(f"library file={lib} does not exist!")
            raise ValueError
        lib_list.append(lib)

    # Initialize library info data structure.
    arpinfo = common.ArpInfo()
    if fuel_type == "UOX":
        key_e = dim_map["enrichment"]
        key_m = dim_map["mod_dens"]
        enrichment_list = []
        mod_dens_list = []
        burnup_list = []

        perms = generate["perms"]
        for i in range(len(perms)):
            # Get library file and other convenience variables.
            lib = lib_list[i]
            input = perm["file"]
            output = Path(input).with_suffix(".out")
            perm = perms[i]
            state = perm["state"]

            # Accumulate the three dimensions:
            e = state[key_e]
            enrichment_list.append(e)

            m = state[key_m]
            mod_dens_list.append(m)

            bu = common.parse_burnups_from_triton_output(work_dir / output)

            if len(burnup_list) > 0 and not np.array_equal(burnup_list, bu):
                raise ValueError(
                    f"library file={lib} burnups deviated from previous list!"
                )
            burnup_list = bu

        arpinfo.init_uox(model["name"], lib_list, enrichment_list, mod_dens_list)
        arpinfo.burnup_list = burnup_list

    else:
        raise ValueError("only fuel_type==UOX is supported right now")

    # Set new canonical file names.
    arpinfo.set_canonical_filenames(".h5")

    # Generate burnup string.
    bu_str = ",".join([str(bu) for bu in arpinfo.burnup_list])
    idtags = "assembly_type={:s},fuel_type={:s}".format(arpinfo.name, arpinfo.fuel_type)

    # Create the arplibs directory and create data files inside.
    d = Path(work_dir) / "arplibs"
    if d.exists():
        shutil.rmtree(d)
    os.mkdir(d)
    perms = list()
    for i in range(len(lib_list)):
        old_lib = lib_list[i]
        new_lib = d / arpinfo.get_lib_by_index(i)

        obiwan = model["obiwan"]
        common.logger.info(f"using OBIWAN to convert {old_lib.name} to {new_lib.name}")

        # Set burnups on file using obiwan (should only be necessary in earlier SCALE versions).
        common.run_command(f"{obiwan} convert -i -setbu='[{bu_str}]' {old_lib}")
        bad_local = Path(old_lib.with_suffix(".f33").name)
        if bad_local.exists():
            common.logger.warning(f"fixup: moving local={bad_local} to {new_lib}")
            shutil.move(bad_local, old_lib)

        # Set tags on file using obiwan.
        interptags = arpinfo.interptags_by_index(i)
        common.run_command(
            f"{obiwan} tag -interptags='{interptags}' -idtags='{idtags}' {old_lib}"
        )
        # Convert to HDF5 and move to arplibs.
        common.run_command(
            f"{obiwan} convert -format=hdf5 -type=f33 {old_lib} -dir={d}"
        )
        shutil.move(Path(d / old_lib.name).with_suffix(".h5"), new_lib)

        # Save relevant permutation data in a list.
        perms.append({"input": input_list[i], **arpinfo.interpvars_by_index(i)})

    # Write arpdata.txt.
    arpdata_txt = work_dir / "arpdata.txt"
    common.logger.info(f"Building arpdata.txt at {arpdata_txt} ... ")
    with open(arpdata_txt, "w") as f:
        f.write(arpinfo.get_arpdata())

    return {
        "archive_file": "arpdata.txt:" + arpinfo.name,
        "perms": perms,
        "work_dir": str(work_dir),
    }
