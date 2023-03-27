"""
Generate functions for static data. 

"""
from typing import Literal
import scale.olm.internal as internal

__all__ = ["pass_through"]

_TYPE_PASS_THROUGH = "scale.olm.generate.static:pass_through"


def _schema_pass_through(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_PASS_THROUGH, with_state=with_state)
    return _schema


def _test_args_pass_through(with_state: bool = False):
    return {"_type": _TYPE_PASS_THROUGH, "addnux": 2, "xslib": "v7.1"}


def pass_through(_type: Literal[_TYPE_PASS_THROUGH] = None, **x):
    """Simple pass through of static data.

    Examples:

        >>> data={'x': 'sally', 'y': 9.0}
        >>> pass_through(**data)
        {'x': 'sally', 'y': 9.0}

    """
    return x
