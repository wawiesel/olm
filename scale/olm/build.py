import scale.olm.common as common
import os
import json
from pathlib import Path
import copy
import shutil
import numpy as np


def arpdata_txt(model, fuel_type, suffix):
    """Build an ORIGEN reactor library in arpdata.txt format."""

    # Get working directory.
    work_dir = Path(model["work_dir"])

    # Get list of files by using the generate.json output and changing the
    # suffix to the expected library file.
    generate_json = work_dir / "generate.json"
    with open(generate_json, "r") as f:
        generate = json.load(f)
    files = []
    for perm in generate["perms"]:
        # Convert from .inp to expected suffix.
        file = work_dir / Path(perm["file"])
        file = file.with_suffix(suffix)
        if not file.exists():
            common.logger.error(f"library file={file} does not exist!")
            raise ValueError
        files.append(file)

    # Initialize library info data structure.
    libinfo = common.LibInfo()
    if fuel_type == "UOX":
        enrichments = []
        coolant_densities = []
        burnups0 = []
        for perm in generate["perms"]:
            # Get tags using internal state (could use obiwan in future)
            enrichments.append(perm["state"]["enrichment"])
            coolant_densities.append(perm["state"]["coolant_density"])
            burnups = [float(0)]
            for x in perm["time"]["burndata"]:
                burnups.append(burnups[-1] + float(x["power"] * x["burn"]))

            if len(burnups0) > 0:
                if not np.array_equal(burnups0, burnups):
                    common.logger.error(
                        "library file={} burnups deviated from previous list!".format(
                            perm["file"]
                        )
                    )
                    raise ValueError
            burnups0 = burnups

        libinfo.init_uox(model["name"], files, enrichments, coolant_densities)
        libinfo.burnups = burnups0
    else:
        raise ValueError

    # Generate new canonical file names.
    libinfo.files = libinfo.get_canonical_filenames(".h5")

    # Create the arplibs directory and create data files inside.
    d = Path(work_dir) / "arplibs"
    if d.exists():
        shutil.rmtree(d)
    os.mkdir(d)
    for i in range(len(files)):
        file = files[i]
        new_file = d / libinfo.get_file_by_index(i)
        common.logger.info(f"using OBIWAN to convert {file.name} to {new_file.name}")

        # convert to HDF5 and copy
        obiwan = model["obiwan"]
        common.run_command(f"{obiwan} convert -format=hdf5 -type=f33 {file} -dir={d}")
        shutil.move(Path(d / file.name).with_suffix(".h5"), new_file)

        # TODO: Alter burnups on file using obiwan

    # Write arpdata.txt.
    arpdata_txt = work_dir / "arpdata.txt"
    common.logger.info(f"Building arpdata.txt at {arpdata_txt} ... ")
    with open(arpdata_txt, "w") as f:
        f.write(libinfo.get_arpdata())

    return {"archive_file": "arpdata.txt:" + libinfo.name, "work_dir": str(work_dir)}
