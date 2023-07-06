import scale.olm.common as common


def archive(model):
    archive_file = model["archive_file"]
    common.logger.info(f"Building archive at {archive_file} ... ")
    common.logger.warning(f"BUILD not implemented yet!")
    return {"archive_file": archive_file}
