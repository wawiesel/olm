def full_hypercube(**states):
    """Generate all the permutations assuming a dense N-dimensional space."""
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
