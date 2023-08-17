import scale.olm.common as common
from pathlib import Path
import shutil
import os


def make_mini_arpdatatxt(dry_run, registry, dest):
    """Create a local arpdata.txt and arplibs"""

    # Concatenate the blocks from each name.
    mini_arpdata = ""
    files_to_copy = []
    for name in registry:
        arpinfo = registry[name]
        path = arpinfo.path
        common.logger.info(f"linking {name} from {path}")
        mini_arpdata += f"!{name}\n" + arpinfo.block
        files_to_copy.extend([arpinfo.arplibs_dir / x for x in arpinfo.files])

    # Create an arpdata.txt file with the concatenated content.
    a = Path(dest) / "arpdata.txt"
    if a.exists():
        common.logger.error(
            f"arpdata.txt already exists at path={path} and will not be overwritten"
        )
    else:
        if dry_run:
            common.logger.info(f"not writing {a} because --dry-run")
        else:
            with open(a, "w") as f:
                f.write(mini_arpdata)

    # Create the arplibs directory by copying data files.
    d = Path(dest) / "arplibs"
    if d.exists():
        common.logger.error(
            f"arplibs directory already exists at path={path} and will not be overwritten"
        )
    else:
        if dry_run:
            common.logger.info(f"not writing {d} because --dry-run")
        else:
            os.mkdir(d)
        for file in files_to_copy:
            if dry_run:
                common.logger.info(f"not copying {file} to {d} because --dry-run")
            else:
                common.logger.info(f"copying {file} to {d}")
                shutil.copy(file, d)
