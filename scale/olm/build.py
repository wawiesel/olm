import scale.olm.common as common
import os
import json
from pathlib import Path
import os
import copy
import shutil
import numpy as np
import subprocess
import datetime


def archive(model):
    """Build an ORIGEN reactor archive in HDF5 format."""
    archive_file = model["archive_file"]
    config_file = model["work_dir"] + os.path.sep + "generate.json"

    # Load the permuation data
    with open(config_file, "r") as f:
        data = json.load(f)

    assem_tag = "assembly_type={:s}".format(model["name"])
    lib_paths = []

    # Tag each permutation's libraries
    for perm in data["perms"]:
        perm_dir = Path(perm["file"]).parent
        perm_name = Path(perm["file"]).stem
        statevars = perm["state"]
        lib_path = os.path.join(perm_dir, perm_name + ".system.f33")
        lib_paths.append(lib_path)
        common.logger.info(f"Now tagging {lib_path}")

        ts = ",".join(key + "=" + str(value) for key, value in statevars.items())
        try:
            subprocess.run(
                [
                    model["obiwan"],
                    "tag",
                    lib_path,
                    f"-interptags={ts}",
                    f"-idtags={assem_tag}",
                ],
                capture_output=True,
                check=True,
            )
        except subprocess.SubprocessError as error:
            print(error)
            print("OBIWAN library tagging failed; cannot build archive")

    to_consolidate = " ".join(lib for lib in lib_paths)
    common.logger.info(f"Building archive at {archive_file} ... ")
    try:
        subprocess.run(
            [
                model["obiwan"],
                "convert",
                "-format=hdf5",
                "-name={archive_file}",
                to_consolidate,
            ],
            check=True,
        )
    except subprocess.SubprocessError as error:
        print(error)
        print("OBIWAN library conversion to archive format failed")

    return {"archive_file": archive_file}


def __generate_thinned_list(keep_every, y_list, always_keep_ends=True):
    """Generate a thinned list using every point (1), every other point (2),
    every third point (3), etc."""

    if not keep_every > 0:
        raise ValueError(
            "The thinning parameter keep_every={keep_every} must be an integer >0!"
        )

    thinned_list = list()
    j = 0
    rm = 1
    for y in y_list:
        if always_keep_ends and (j == 0 or j == len(y_list) - 1):
            p = True
        elif rm >= keep_every:
            p = True
        else:
            p = False
        if p:
            thinned_list.append(y)
            rm = 0
        rm += 1
        j += 1
    return thinned_list


def __get_files(work_dir, suffix, perms):
    """Get list of files by using the generate.json output and changing the suffix to the expected library file."""

    file_list = list()
    for perm in perms:
        input = perm["file"]

        # Convert from .inp to expected suffix.
        lib = work_dir / Path(input)
        lib = lib.with_suffix(suffix)
        if not lib.exists():
            raise ValueError(f"library file={lib} does not exist!")

        output = work_dir / Path(input).with_suffix(".out")
        if not output.exists():
            raise ValueError(
                f"output file={output} does not exist! Maybe run was not complete successfully?"
            )

        file_list.append({"lib": lib, "output": output})

    return file_list


def __get_burnup_list(file_list):
    """Extract a burnup list from the output file and make sure they are all the same."""
    burnup_list = list()
    for i in range(len(file_list)):
        bu = common.parse_burnups_from_triton_output(file_list[i]["output"])

        if len(burnup_list) > 0 and not np.array_equal(burnup_list, bu):
            raise ValueError(f"library file={lib} burnups deviated from previous list!")
        burnup_list = bu

    return burnup_list


def __get_arpinfo_uox(name, perms, file_list, dim_map):
    """For UOX, get the relative ARP interpolation information."""

    # Get the names of the keys in the state.
    key_e = dim_map["enrichment"]
    key_m = dim_map["mod_dens"]

    # Build these lists for each permutation to use in init_uox below.
    enrichment_list = []
    mod_dens_list = []
    lib_list = []
    for i in range(len(perms)):
        # Get the interpolation variables from the state.
        state = perms[i]["state"]
        e = state[key_e]
        enrichment_list.append(e)
        m = state[key_m]
        mod_dens_list.append(m)

        # Get the library name.
        lib_list.append(file_list[i]["lib"])

    # Create and return arpinfo.
    arpinfo = common.ArpInfo()
    arpinfo.init_uox(name, lib_list, enrichment_list, mod_dens_list)
    return arpinfo


def __get_arpinfo_mox(name, perms, file_list, dim_map):
    """For MOX, get the relative ARP interpolation information."""

    # Get the names of the keys in the state.
    key_e = dim_map["pu239_frac"]
    key_p = dim_map["pu_frac"]
    key_m = dim_map["mod_dens"]

    # Build these lists for each permutation to use in init_uox below.
    pu239_frac_list = []
    pu_frac_list = []
    mod_dens_list = []
    lib_list = []
    for i in range(len(perms)):
        # Get the interpolation variables from the state.
        state = perms[i]["state"]
        e = state[key_e]
        pu239_frac_list.append(e)
        p = state[key_p]
        pu_frac_list.append(p)
        m = state[key_m]
        mod_dens_list.append(m)

        # Get the library name.
        lib_list.append(file_list[i]["lib"])

    # Create and return arpinfo.
    arpinfo = common.ArpInfo()
    arpinfo.init_mox(name, lib_list, pu239_frac_list, pu_frac_list, mod_dens_list)
    return arpinfo


def __get_arpinfo(name, perms, file_list, fuel_type, dim_map):
    """Populate the ArpInfo data."""

    # Initialize info based on fuel type.
    if fuel_type == "UOX":
        arpinfo = __get_arpinfo_uox(name, perms, file_list, dim_map)
    elif fuel_type == "MOX":
        arpinfo = __get_arpinfo_mox(name, perms, file_list, dim_map)
    else:
        raise ValueError(
            "Unknown fuel_type={fuel_type} (only MOX/UOX is supported right now)"
        )

    # Get the burnups.
    arpinfo.burnup_list = __get_burnup_list(file_list)

    # Set new canonical file names.
    arpinfo.set_canonical_filenames(".h5")

    return arpinfo


def __process_libraries(obiwan, work_dir, arpinfo, thinned_list, file_list):
    """Process libraries with OBIWAN, including copying, thinning, setting tags, etc."""

    # Create the arplibs directory and clear data files inside.
    d = work_dir / "arplibs"
    if d.exists():
        shutil.rmtree(d)
    os.mkdir(d)

    # Generate burnup string.
    bu_str = ",".join([str(bu) for bu in arpinfo.burnup_list])

    # Generate idtags.
    idtags = "assembly_type={:s},fuel_type={:s}".format(arpinfo.name, arpinfo.fuel_type)

    # Generate burnup string for thin list.
    thin_bu_str = ",".join([str(bu) for bu in thinned_list])
    common.logger.info(
        f"Replacing burnup list {bu_str} with thinned list {thin_bu_str}..."
    )

    # Create a temporary directory for libraries in process.
    tmp = d / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # Use obiwan to perform most of the processes.
    perms = list()
    for i in range(len(file_list)):
        old_lib = file_list[i]["lib"]
        tmp_lib = tmp / file_list[i]["lib"].name
        common.logger.info(f"copying original library {old_lib} to {tmp_lib}")
        shutil.copyfile(old_lib, tmp_lib)

        # Set burnups on file using obiwan (should only be necessary in earlier SCALE versions).
        common.run_command(f"{obiwan} convert -i -setbu='[{bu_str}]' {tmp_lib}")
        bad_local = Path(tmp_lib.with_suffix(".f33").name)
        if bad_local.exists():
            common.logger.warning(f"fixup: moving local={bad_local} to {tmp_lib}")
            shutil.move(bad_local, tmp_lib)

        # Perform burnup thinning.
        if bu_str != thin_bu_str:
            common.run_command(
                f"{obiwan} convert -i -thin=1 -tvals='[{thin_bu_str}]' {tmp_lib}",
                check_return_code=False,
            )
            if bad_local.exists():
                common.logger.warning(f"fixup: moving local={bad_local} to {tmp_lib}")
                shutil.move(bad_local, tmp_lib)

        # Set tags.
        interptags = arpinfo.interptags_by_index(i)
        common.run_command(
            f"{obiwan} tag -interptags='{interptags}' -idtags='{idtags}' {tmp_lib}"
        )

        # Convert to HDF5.
        common.run_command(
            f"{obiwan} convert -format=hdf5 -type=f33 {tmp_lib} -dir={tmp}"
        )

        # Move the local library to the new proper place.
        new_lib = d / arpinfo.get_lib_by_index(i)
        shutil.move(tmp_lib.with_suffix(".h5"), new_lib)

        # Save relevant permutation data in a list.
        perms.append(
            {
                "files": {
                    "old_lib": str(old_lib.relative_to(work_dir)),
                    "new_lib": str(new_lib.relative_to(work_dir)),
                },
                "interpvars": {**arpinfo.interpvars_by_index(i)},
            }
        )

    # Remove temporary files.
    shutil.rmtree(tmp)

    # Write arpdata.txt.
    arpinfo.burnup_list = thinned_list
    arpdata_txt = work_dir / "arpdata.txt"
    common.logger.info(f"Writing arpdata.txt at {arpdata_txt} ... ")
    with open(arpdata_txt, "w") as f:
        f.write(arpinfo.get_arpdata())
    archive_file = "arpdata.txt:" + arpinfo.name

    return archive_file, perms


def arpdata_txt(model, fuel_type, suffix, dim_map, keep_every):
    """Build an ORIGEN reactor library in arpdata.txt format."""

    # Get working directory.
    work_dir = Path(model["work_dir"])

    # Get generate data which has permutations list with file names.
    generate_json = work_dir / "generate.json"
    with open(generate_json, "r") as f:
        generate = json.load(f)
    perms = generate["perms"]

    # Get library,input,output in one place.
    file_list = __get_files(work_dir, suffix, perms)

    # Get library info data structure.
    arpinfo = __get_arpinfo(model["name"], perms, file_list, fuel_type, dim_map)

    # Generate thinned burnup list.
    thinned_burnup_list = __generate_thinned_list(keep_every, arpinfo.burnup_list)

    # Process libraries into their final places.
    archive_file, perms = __process_libraries(
        model["obiwan"], work_dir, arpinfo, thinned_burnup_list, file_list
    )

    return {
        "archive_file": archive_file,
        "perms": perms,
        "work_dir": str(work_dir),
        "date": datetime.datetime.utcnow().isoformat(" ", "minutes"),
        "space": arpinfo.get_space(),
    }
