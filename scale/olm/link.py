import scale.olm.common as common


def make_available(dry_run, libinfo, dest, format):
    if libinfo.format == format:
        common.logger.info(
            "Library format and destination are the same ({})".format(format)
        )
    else:
        common.logger.info(
            "Library format ({}) and destination format ({}) are not the same! Will do local conversion.".format(
                libinfo.format, format
            )
        )
