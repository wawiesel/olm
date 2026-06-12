import scale.olm.internal as internal
import scale.olm.core as core
import os
import json
from pathlib import Path
import os
import copy
import shutil
import numpy as np
import subprocess
import datetime
from dataclasses import dataclass
from typing import Literal

__all__ = ["arpdata_txt"]

_TYPE_ARPDATA_TXT = "scale.olm.assemble:arpdata_txt"


def _schema_arpdata_txt(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_ARPDATA_TXT, with_state=with_state)
    return _schema


def _test_args_arpdata_txt(with_state: bool = False):
    return {
        "_type": _TYPE_ARPDATA_TXT,
        "dry_run": False,
        "fuel_type": "UOX",
        "dim_map": {"mod_dens": "mod_dens", "enrichment": "enrichment"},
        "burnup_rtol": 2.0e-2,
        "material_lumping": "BASIS",
    }


def arpdata_txt(
    fuel_type: str,
    dim_map: dict,
    keep_every: int,
    burnup_rtol: float = 2.0e-2,
    material_lumping: str = "BASIS",
    _model: dict = {},
    _env: dict = {},
    dry_run: bool = False,
    _type: Literal[_TYPE_ARPDATA_TXT] = None,
):
    """Build an ORIGEN reactor library in arpdata.txt format.

    Args:
        fuel_type: Which type of fuel: UOX/MOX.

        dim_map: arpdata.txt requires specially named dimensions. These may exist in the
                 state or you may need to map them from the state variables.

                 if fuel_type=='UOX', enrichment, mod_dens must be mapped to state variables
                 if fuel_type=='MOX', pu239_frac, pu_frac, mod_dens must be mapped to state variables

        material_lumping: TRITON material used to assemble libraries and history.
                         Accepts BASIS, SYSTEM, or MIX<N>. BASIS and SYSTEM use
                         TRITON's system library and case -2. MIX<N> uses
                         .mixNNNN.f33 and F71 case N.

    """

    if dry_run:
        return {}

    # Get working directory.
    work_path = Path(_env["work_dir"])
    material_lumping = _MaterialLumping.from_value(material_lumping)

    # Get library info data structure.
    arpinfo = _get_arpinfo(
        _env["obiwan"],
        work_path,
        _model["name"],
        fuel_type,
        dim_map,
        material_lumping,
        burnup_rtol,
    )

    # Generate thinned burnup list.
    thinned_burnup_list = _generate_thinned_burnup_list(keep_every, arpinfo.burnup_list)

    # Process libraries into their final places.
    archive_file, points = _process_libraries(
        _env["obiwan"], work_path, arpinfo, thinned_burnup_list, material_lumping
    )

    return {
        "archive_file": archive_file,
        "points": points,
        "work_dir": str(work_path),
        "date": datetime.datetime.utcnow().isoformat(" ", "minutes"),
        "space": arpinfo.get_space(),
        "burnup_rtol": burnup_rtol,
        "material_lumping": material_lumping.value,
    }


def archive(model):
    """Build an ORIGEN reactor library in HDF5 archive format.

    Args:
        model (dict): A dictionary containing the following keys:
            - archive_file (str): The path and filename of the reactor archive to be created.
            - work_dir (str): The path to the working directory.
            - name (str): The name of the reactor.
            - obiwan (str): The path to the OBIWAN executable.

    Returns:
        dict: relevant data on the result of creating an archive
    """
    archive_file = model["archive_file"]
    config_file = model["work_dir"] + os.path.sep + "generate.olm.json"

    # Load the permuation data
    with open(config_file, "r") as f:
        data = json.load(f)

    assem_tag = "assembly_type={:s}".format(model["name"])
    lib_paths = []

    # Tag each permutation's libraries
    for perm in data["perms"]:
        perm_dir = Path(perm["input_file"]).parent
        perm_name = Path(perm["input_file"]).stem
        statevars = perm["state"]
        lib_path = os.path.join(perm_dir, perm_name + ".system.f33")
        lib_paths.append(lib_path)
        internal.logger.debug(f"Now tagging {lib_path}")

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
            print("OBIWAN library tagging failed; cannot assemble archive")

    to_consolidate = " ".join(lib for lib in lib_paths)
    internal.logger.info(f"Building archive at {archive_file} ... ")
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


def _generate_thinned_burnup_list(keep_every, y_list, always_keep_ends=True):
    """Generate a thinned list using every point (1), every other point (2),
    every third point (3), etc."""

    if not keep_every > 0:
        raise ValueError(
            "The thinning parameter keep_every={keep_every} must be an integer >0!"
        )

    thinned_burnup_list = list()
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
            thinned_burnup_list.append(y)
            rm = 0
        rm += 1
        j += 1
    return thinned_burnup_list


def _get_artifact_contract(perm):
    input_file = perm["input_file"]
    try:
        artifact_contract = perm["_scale"]["artifact_contract"]
    except KeyError:
        raise ValueError(
            "Generated permutation is missing _scale.artifact_contract "
            f"for input_file={input_file}"
        ) from None

    return core.ScaleArtifactContract.from_value(artifact_contract)


@dataclass(frozen=True)
class _MaterialLumping:
    value: str
    caseid: int
    library_suffix: str

    @classmethod
    def from_value(cls, material_lumping):
        label = str(material_lumping).strip().upper()
        if label in ("BASIS", "SYSTEM"):
            return cls(label, -2, ".system.f33")
        if label.startswith("MIX"):
            mixid = label[3:]
            if mixid.isdigit() and int(mixid) > 0:
                mixid = int(mixid)
                return cls(f"MIX{mixid}", mixid, f".mix{mixid:04d}.f33")
        raise ValueError(
            "TRITON material_lumping must be BASIS, SYSTEM, or MIX<N>; "
            f"got {material_lumping!r}"
        )

    @property
    def is_system_lumping(self):
        return self.value in ("BASIS", "SYSTEM")


_DEFAULT_MATERIAL_LUMPING = _MaterialLumping.from_value("BASIS")


def _library_suffix_for_artifact(artifact_contract, material_lumping):
    if artifact_contract == core.ScaleArtifactContract.TRITON:
        return material_lumping.library_suffix

    if artifact_contract == core.ScaleArtifactContract.POLARIS:
        if material_lumping.is_system_lumping:
            return ".system.f33"
        raise ValueError(
            "Polaris material_lumping currently supports BASIS or SYSTEM; "
            f"got {material_lumping.value}"
        )


def _get_files(work_dir, perms, material_lumping=_DEFAULT_MATERIAL_LUMPING):
    """Get list of files by using the generate.olm.json output and changing the suffix to the
    expected library file. Note this is in permutation order, not state space order."""

    file_list = list()
    for perm in perms:
        input = perm["input_file"]
        artifact_contract = _get_artifact_contract(perm)
        lib_suffix = _library_suffix_for_artifact(artifact_contract, material_lumping)

        # Convert from .inp to expected suffix.
        lib = work_dir / Path(input)
        lib = lib.with_suffix(lib_suffix)
        if not lib.exists():
            raise ValueError(f"library file={lib} does not exist!")

        output = work_dir / Path(input).with_suffix(".out")
        if not output.exists():
            raise ValueError(
                f"output file={output} does not exist! Maybe run was not complete successfully?"
            )

        f71 = work_dir / Path(input).with_suffix(".f71")
        if not f71.exists():
            raise ValueError(
                f"f71 file={f71} does not exist! Maybe run was not complete successfully?"
            )

        file_list.append(
            {
                "lib": lib,
                "output": output,
                "f71": f71,
                "artifact_contract": artifact_contract,
            }
        )

        # Polaris burnup grids are read from T16 because its F33 burnup table
        # is not populated with the nominal library grid.
        t16 = work_dir / Path(input).with_suffix(".t16")
        if t16.exists():
            file_list[-1]["t16"] = t16

    return file_list


def _burnup_lists_match(reference, candidate, burnup_rtol):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        return False
    return np.allclose(reference, candidate, rtol=burnup_rtol, atol=1.0e-8)


def _burnup_grid_mismatch_message(reference, candidate):
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        return f"reference_shape={reference.shape} candidate_shape={candidate.shape}"

    abs_delta = np.abs(candidate - reference)
    rel_delta = abs_delta / np.maximum(np.abs(reference), 1.0e-30)
    index = int(np.argmax(abs_delta))
    return (
        f"index={index} reference={reference[index]} candidate={candidate[index]} "
        f"abs_delta={abs_delta[index]} rel_delta={rel_delta[index]}"
    )


def _burnups_from_file_info(obiwan, file_info):
    artifact_contract = file_info["artifact_contract"]

    if artifact_contract == core.ScaleArtifactContract.POLARIS:
        if "t16" not in file_info:
            raise ValueError(
                "Polaris burnup extraction requires a t16 file for "
                f"library file={file_info['lib']} from output file={file_info['output']}"
            )
        return core.ScaleOutfile.parse_burnups_from_polaris_t16(file_info["t16"])

    if artifact_contract == core.ScaleArtifactContract.TRITON:
        return core.Obiwan.get_burnups_from_f33(obiwan, file_info["lib"])


def _assembled_caseid_for_artifact(output_file, artifact_contract, material_lumping):
    if artifact_contract == core.ScaleArtifactContract.TRITON:
        return material_lumping.caseid

    if artifact_contract == core.ScaleArtifactContract.POLARIS:
        if material_lumping.value == "SYSTEM":
            return material_lumping.caseid
        caseid = core.ScaleOutfile.parse_polaris_state_table(
            output_file, material_lumping.value
        )
        if caseid == -2:
            raise ValueError(
                f"Cannot identify Polaris material_lumping={material_lumping.value} "
                f"case from output file={output_file}"
            )
        return caseid


def _require_assembled_case(ii, caseid, artifact_contract, output_file):
    response_key = f"case({caseid})"
    if response_key not in ii["responses"]:
        raise ValueError(
            "Cannot identify assembled material case; "
            f"case {caseid} not found in F71 table for "
            f"artifact_contract={artifact_contract.value} output_file={output_file}"
        )
    return response_key


def _get_burnup_list(obiwan, file_list, burnup_rtol=2.0e-2):
    """Extract the ARPDATA burnup axis from each selected high-order library."""
    if burnup_rtol <= 0.0:
        raise ValueError(f"burnup_rtol must be > 0.0; got {burnup_rtol}")

    burnup_list = []
    burnup_arrays = []
    previous_library_file = ""
    for i in range(len(file_list)):
        library_file = file_list[i]["lib"]
        output_file = file_list[i]["output"]
        bu = np.asarray(_burnups_from_file_info(obiwan, file_list[i]), dtype=float)

        if len(burnup_list) == 0:
            burnup_list = bu
        elif not _burnup_lists_match(burnup_list, bu, burnup_rtol):
            raise ValueError(
                "High-order library burnups for "
                f"library file={library_file} from output file={output_file} "
                f"deviated from previous library file={previous_library_file}; "
                "arpdata.txt requires one burnup grid for the block. "
                + _burnup_grid_mismatch_message(burnup_list, bu)
            )
        burnup_arrays.append(bu)
        previous_library_file = library_file

    if not burnup_arrays:
        return []
    return np.mean(np.stack(burnup_arrays), axis=0)


def _get_arpinfo_uox(name, perms, file_list, dim_map):
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
    arpinfo = core.ArpInfo()
    arpinfo.init_uox(name, lib_list, enrichment_list, mod_dens_list)
    return arpinfo


def _get_arpinfo_mox(name, perms, file_list, dim_map):
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
    arpinfo = core.ArpInfo()
    arpinfo.init_mox(name, lib_list, pu239_frac_list, pu_frac_list, mod_dens_list)
    return arpinfo


def _arpinfo_builder_for_fuel_type(fuel_type):
    builders = {
        "UOX": _get_arpinfo_uox,
        "MOX": _get_arpinfo_mox,
    }
    try:
        return builders[fuel_type]
    except KeyError:
        expected = ", ".join(builders)
        raise ValueError(
            f"Unknown fuel_type={fuel_type}; expected one of: {expected}"
        ) from None


def _get_arpinfo(
    obiwan,
    work_dir,
    name,
    fuel_type,
    dim_map,
    material_lumping=_DEFAULT_MATERIAL_LUMPING,
    burnup_rtol=2.0e-2,
):
    """Populate the ArpInfo data."""

    # Get generate data which has permutations list with file names.
    generate_json = work_dir / "generate.olm.json"
    with open(generate_json, "r") as f:
        generate = json.load(f)
    perms = generate["perms"]

    # Get library,input,output in one place.
    file_list = _get_files(work_dir, perms, material_lumping)

    arpinfo = _arpinfo_builder_for_fuel_type(fuel_type)(
        name, perms, file_list, dim_map
    )

    # Get the ARPDATA burnups from the F33 libraries. The matching F71 files are
    # read later only for inventory metadata and interval-average powers.
    caseid = material_lumping.caseid
    arpinfo.burnup_list = _get_burnup_list(obiwan, file_list, burnup_rtol)

    # Set new canonical file names.
    arpinfo.set_canonical_filenames(".h5")
    arpinfo.material_lumping = material_lumping.value
    arpinfo.caseid = caseid

    return arpinfo


def _get_comp_system(ii_data):
    """Extract the following information from the inventory interface (ii) data."""

    x = ii_data["responses"]["system"]
    volume = x["volume"]
    amount_list = x["amount"][0]  # Initial amount
    data_map = ii_data["data"]["nuclides"]
    vh = x["nuclideVectorHash"]
    nuclide_list = ii_data["definitions"]["nuclideVectors"][vh]

    x = dict()
    total_mass = 0.0
    for i in range(len(nuclide_list)):
        name = nuclide_list[i]
        data = data_map[name]
        amount = amount_list[i]
        molar_mass = data["mass"]
        mass = amount * molar_mass
        total_mass += mass
        z = data["atomicNumber"]
        e = data["element"]
        m = data["isomericState"]
        a = data["massNumber"]
        mstr = ""
        if m >= 1:
            mstr = "m"
        elif m >= 2:
            mstr = "m" + str(m)
        eam = "{}{}{}".format(e.lower(), int(a), mstr)
        if z >= 92:
            x[eam] = amount * molar_mass

    comp = core.CompositionManager.calculate_hm_oxide_breakdown(x)
    comp["info"] = core.CompositionManager.approximate_hm_info(comp)
    comp["density"] = total_mass / volume

    return comp


def _get_replay_burndata_count(perm):
    time = perm.get("time", {})
    padding_gwd = float(time.get("final_burnup_padding_gwd", 0.0))
    if padding_gwd < 0.0:
        raise ValueError(
            "final_burnup_padding_gwd must be >= 0.0; "
            f"got {padding_gwd}"
        )
    if padding_gwd == 0.0:
        return None

    burndata = time.get("burndata")
    if not burndata:
        raise ValueError(
            "final_burnup_padding_gwd > 0.0 requires generated time.burndata "
            "so assemble can exclude the final padding interval from replay."
        )
    return len(burndata) - 1


def _truncate_history_burndata(history, replay_burndata_count):
    if replay_burndata_count is None:
        return history

    burndata = history.get("burndata", [])
    if replay_burndata_count > len(burndata):
        raise ValueError(
            "Cannot truncate high-order history to more burn intervals than it has: "
            f"requested={replay_burndata_count} available={len(burndata)}"
        )

    truncated = copy.deepcopy(history)
    truncated["burndata"] = burndata[:replay_burndata_count]
    return truncated


def _truncate_ii_system_time(ii, time_count):
    if time_count is None:
        return ii

    system = ii["responses"]["system"]
    current_count = len(system["time"])
    if time_count > current_count:
        raise ValueError(
            "Cannot truncate high-order ii.json to more time points than it has: "
            f"requested={time_count} available={current_count}"
        )

    truncated = copy.deepcopy(ii)
    system = truncated["responses"]["system"]
    system["time"] = system["time"][:time_count]
    system["amount"] = system["amount"][:time_count]
    return truncated


def _process_libraries(
    obiwan,
    work_dir,
    arpinfo,
    thinned_burnup_list,
    material_lumping=_DEFAULT_MATERIAL_LUMPING,
):
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
    thin_bu_str = ",".join([str(bu) for bu in thinned_burnup_list])
    internal.logger.info("burnup thinning:", original_bu=bu_str, thinned_bu=thin_bu_str)
    arpinfo.burnup_list = thinned_burnup_list

    # Create a temporary directory for libraries in process.
    tmp = d / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # Get generate data which has permutations list with file names.
    generate_json = work_dir / "generate.olm.json"
    with open(generate_json, "r") as f:
        generate = json.load(f)
    perms = generate["perms"]

    # Use obiwan to perform most of the processes.
    points = list()
    for i in range(arpinfo.num_libs()):
        new_lib = Path(arpinfo.get_lib_by_index(i))
        old_lib = Path(arpinfo.origin_lib_list[i])
        tmp_lib = tmp / old_lib.name
        internal.logger.debug(f"Copying original library {old_lib} to {tmp_lib}")
        shutil.copyfile(old_lib, tmp_lib)

        bad_local = Path(tmp_lib.with_suffix(".f33").name)

        # Perform burnup thinning.
        if bu_str != thin_bu_str:
            internal.run_command(
                f"{obiwan} convert -i -thin=1 -tvals='[{thin_bu_str}]' {tmp_lib}",
                check_return_code=False,
                echo=False,
            )
            if bad_local.exists():
                internal.logger.warning("Fixup: relocating local", file=str(bad_local))
                shutil.move(bad_local, tmp_lib)

        # Set tags.
        interptags = arpinfo.interptags_by_index(i)
        internal.run_command(
            f"{obiwan} tag -interptags='{interptags}' -idtags='{idtags}' {tmp_lib}",
            echo=False,
        )

        # Convert to HDF5.
        internal.run_command(
            f"{obiwan} convert -format=hdf5 -type=f33 {tmp_lib} -dir={tmp}", echo=False
        )

        # Move the local library to the new proper place.
        new_lib = d / arpinfo.get_lib_by_index(i)
        shutil.move(tmp_lib.with_suffix(".h5"), new_lib)

        # Generate the system composition information from the system ii.json.
        k = arpinfo.get_perm_by_index(i)
        perm = perms[k]
        replay_burndata_count = _get_replay_burndata_count(perm)
        replay_time_count = (
            None if replay_burndata_count is None else replay_burndata_count + 1
        )
        f71 = (work_dir / perm["input_file"]).with_suffix(".f71")
        text = internal.run_command(
            f"{obiwan} view -format=ii.json {f71}",
            echo=False,
        )

        # Load into data structure and rename.
        ii_json = new_lib.with_suffix(".ii.json")
        internal.logger.debug(f"Converting {f71} to {ii_json}")

        ii = json.loads(text)
        output_file = (work_dir / perm["input_file"]).with_suffix(".out")
        artifact_contract = _get_artifact_contract(perm)
        caseid = _assembled_caseid_for_artifact(
            output_file, artifact_contract, material_lumping
        )
        response_key = _require_assembled_case(
            ii, caseid, artifact_contract, output_file
        )
        ii["responses"]["system"] = ii["responses"].pop(response_key)
        ii = _truncate_ii_system_time(ii, replay_time_count)
        with open(ii_json, "w") as f:
            f.write(json.dumps(ii, indent=4))

        # Get the special composition data structure.
        comp_system = _get_comp_system(ii)

        # Save relevant permutation data in a list.
        history = core.Obiwan.get_history_from_f71(obiwan, f71, caseid)
        history = _truncate_history_burndata(history, replay_burndata_count)
        points.append(
            {
                "files": {
                    "origin": {
                        "lib": str(old_lib.relative_to(work_dir)),
                        "f71": str(f71.relative_to(work_dir)),
                    },
                    "lib": str(new_lib.relative_to(work_dir)),
                    "ii_json": str(ii_json.relative_to(work_dir)),
                },
                "comp": {
                    "system": comp_system,
                },
                "material_lumping": material_lumping.value,
                "history": history,
                "_": {"perm": perm},
                "_arpinfo": {
                    "name": arpinfo.name,
                    "interpvars": {**arpinfo.interpvars_by_index(i)},
                    "burnup_list": arpinfo.burnup_list,
                },
            }
        )

    # Remove temporary files.
    shutil.rmtree(tmp)

    # Write arpdata.txt.
    arpdata_txt = work_dir / "arpdata.txt"
    internal.logger.info(f"Writing arpdata.txt at {arpdata_txt} ... ")
    with open(arpdata_txt, "w") as f:
        f.write(arpinfo.get_arpdata())
    archive_file = "arpdata.txt:" + arpinfo.name

    return archive_file, points
