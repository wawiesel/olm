from pydantic import BaseModel
from typing import List, Literal
import scale.olm.internal as internal

__all__ = ["full_hypercube"]

_TYPE_FULL_HYPERCUBE = "scale.olm.generate.states:full_hypercube"


def _schema_full_hypercube(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_FULL_HYPERCUBE, with_state=with_state)
    return _schema


def _test_args_full_hypercube(with_state: bool = False):
    args = {
        "_type": _TYPE_FULL_HYPERCUBE,
        "coolant_density": [0.4, 0.7, 1.0],
        "enrichment": [1.5, 3.5, 4.5],
        "specific_power": [42.0],
    }
    return args


def full_hypercube(_type: Literal[_TYPE_FULL_HYPERCUBE] = None, **states):
    """Generate all the permutations assuming a dense N-dimensional space.

    Args:
        states list[ list[float] ]

    """
    import numpy as np
    import scale.olm.internal as internal

    dims = []
    axes = []
    for dim in states:
        axes.append(sorted(states[dim]))
        internal.logger.debug(f"Processing dimension '{dim}'")
        dims.append(dim)

    permutations = []
    grid = np.array(np.meshgrid(*axes)).T.reshape(-1, len(dims))
    for x in grid:
        y = dict()
        for i in range(len(dims)):
            y[dims[i]] = x[i]
        internal.logger.debug(f"Generated permutation '{y}'")
        permutations.append(y)

    return permutations
