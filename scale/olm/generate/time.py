"""
Generate functions for time-dependent power histories. The preferred
output is a simple list of power levels and time at that power level using
the conventional power in MW/MTIHM and time in days.

A time generation function shall always receive the :code:`state` along with
any other parameters inside the input :code:`time` section.

"""

from pydantic import BaseModel, Field, validate_call
from typing import List, Literal, Annotated
import scale.olm.internal as internal

__all__ = ["constpower_burndata"]


# Data model definitions.
class StateWithSpecificPower(BaseModel):
    specific_power: Annotated[float, Field(gt=0)]


_TYPE_CONSTPOWER_BURNDATA = "scale.olm.generate.time:constpower_burndata"


def _schema_constpower_burndata(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_CONSTPOWER_BURNDATA, with_state=with_state)
    return _schema


def _test_args_constpower_burndata(with_state: bool = False):
    args = {
        "_type": _TYPE_CONSTPOWER_BURNDATA,
        "state": {"specific_power": 40.0},
        "gwd_burnups": [0, 10, 30, 60],
    }
    if not with_state:
        args.pop("state")
    return args


@validate_call
def constpower_burndata(
    state: StateWithSpecificPower,
    gwd_burnups: Annotated[List[float], Field(min_length=1)],
    final_burnup_padding_gwd: Annotated[float, Field(ge=0.0)] = 0.0,
    _type: Literal[_TYPE_CONSTPOWER_BURNDATA] = None,
):
    """Return a list of powers and times assuming constant burnup.

    TODO: Instead of returning burndata dictionary, just return the list. Make
    this look less like a TRITON burndata specification.

    Args:

        state: state point data--only the "specific_power" key in MW/MTIHM units is
               used

        gwd_burnups: list of cumulative burnups in GWd/MTIHM

        final_burnup_padding_gwd: final burnup increment in GWd/MTIHM added
                                  only to generated high-order libraries

    Returns:

        dictionary with "burndata" values as TRITON would expect for its
        burndata block. If "final_burnup_padding_gwd" is positive, the final
        burndata row is padding for high-order library generation only.

    Examples:

        >>> import scale.olm as olm
        >>> data = olm.generate.time.constpower_burndata(
        ...     state={"specific_power": 40},
        ...     gwd_burnups=[0,10,20]
        ... )
        >>> data["burndata"]
        [{'power': 40.0, 'burn': 250.0}, {'power': 40.0, 'burn': 250.0}]
        >>> data["final_burnup_padding_gwd"]
        0.0

    """

    # Calculate cumulative time to achieve each burnup.
    gwd_burnups = sorted(gwd_burnups)
    burnups = [float(x) * 1e3 for x in gwd_burnups]
    days = [burnup / state.specific_power for burnup in burnups]

    # Check warnings and errors.
    if burnups[0] > 0:
        raise ValueError("Burnup step 0.0 GWd/MTHM must be included.")

    # Create the burndata block.
    burndata = []
    if len(days) > 1:
        for i in range(len(days) - 1):
            burndata.append(
                {"power": state.specific_power, "burn": (days[i + 1] - days[i])}
            )
    else:
        burndata.append({"power": state.specific_power, "burn": 0})

    if final_burnup_padding_gwd > 0.0:
        burndata.append(
            {
                "power": state.specific_power,
                "burn": final_burnup_padding_gwd * 1e3 / state.specific_power,
            }
        )

    return {
        "burndata": burndata,
        "final_burnup_padding_gwd": final_burnup_padding_gwd,
    }
