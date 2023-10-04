def scipy_interp(state_var: str, data_pairs, state, method: str = "linear"):
    import scipy as sp
    import scale.olm.internal as internal

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
