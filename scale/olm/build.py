import scale.olm.common as common
import os
import json
from pathlib import Path


def arpdata_from_tags(model, build):
    # For each directory search for build['endswith'] to get list of files.

    # Read tags from the various files using obiwan.
    fuel_type = "UOX"
    files = []
    enrichments = []
    coolant_densities = []
    burnups = []
    libinfo = LibInfo()
    libinfo.init_uox(model["name"], enrichments, coolant_densities, burnups)
    libinfo.files = libinfo.get_filenames()

    # Write arpdata.txt.
    work_dir = Path(model["work_dir"])
    arpdata_txt = work_dir / "arpdata.txt"
    common.logger.info(f"Building arpdata.txt at {arpdata_txt} ... ")
    with open(arpdata_txt, "w") as f:
        f.write(libinfo.get_arpdata())

    # Create the arplibs directory by copying data files.
    d = Path(work_dir) / "arplibs"
    if d.exists():
        print("remove contents")

    os.mkdir(d)
    new_files = []
    for i in range(len(files)):
        old_file = files[i]
        new_file = d / libinfo.files[i]
        common.logger.info(f"copying {old_file} to {new_file}")
        shutil.copy(old_file, new_file)
        new_files.append(new_file)
        # convert to HDF5 in the process
        # $SCALE/bin/obiwan convert -format=hdf5 -type=f33 examples/w17x17/_build/perm00/perm00.system.f33

    return {"arpdata_txt": arpdata_txt, "files": new_files}
