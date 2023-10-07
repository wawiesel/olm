"""
Module for dynamic data generators, dependent on state.
"""
from typing import List, Tuple, Dict, Literal
from enum import Enum
from pydantic import BaseModel
import scale.olm.internal as internal

__all__ = ["scipy_interp"]

_TYPE_SCIPY_INTERP = "scale.olm.generate.dynamic:scipy_interp"


class ScipyInterpMethod(str, Enum):
    LINEAR = "linear"
    PCHIP = "pchip"


def _schema_scipy_interp(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_SCIPY_INTERP, with_state=with_state)
    return _schema


def _test_args_scipy_interp(with_state: bool = False):
    args = {
        "_type": _TYPE_SCIPY_INTERP,
        "state_var": "coolant_density",
        "data_pairs": [(0.3, 0.4), (0.7, 0.5), (1.1, 0.6)],
        "state": {"coolant_density": 0.67},
        "method": "pchip",
    }
    if not with_state:
        args.pop("state")
    return args


def scipy_interp(
    state_var: str,
    data_pairs: List[Tuple[float, float]],
    state: Dict[str, float],
    method: ScipyInterpMethod = ScipyInterpMethod.LINEAR,
    _type: Literal[_TYPE_SCIPY_INTERP] = None,
):
    """
    Interpolate data pairs to the value of a state variable.

    Originally used to interpolate Dancoff factors as a function of moderator/coolant
    density to actual state conditions.

    Args:
        state_var: State variable (x) to access in the 'state' dictionary.

        data_pairs: List of tuples, each an x,y data point.

        state: State dictionary.

        method: Interpolation method.

    Returns:
        float: Interpolated value.

    """
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
