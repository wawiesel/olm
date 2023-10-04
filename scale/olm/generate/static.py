"""
Generate functions for static data. 

"""


def pass_through(**x):
    """Simple pass through of static data.

    Examples:

        >>> data={'x': 'sally', 'y': 9.0}
        >>> pass_through(**data)
        {'x': 'sally', 'y': 9.0}

    """
    return x
