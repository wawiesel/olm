"""

The OLM :obj:`root` module contains functions which manage the root expansion of 
a generate block. Each root generation function should perform the following 
key actions:

1. traverse all other included data blocks and expand their data
2. write this data to disk for each permutation
3. generate and write a SCALE input file for each permutation
2. return a dictionary containing the information on the written files

Once the pre-processing is complete, you are ready to execute the calculations
using :obj:`scale.olm.run`.

Return specification
^^^^^^^^^^^^^^^^^^^^

The return value for any :obj:`generate` function should be a dictionary, 
defined as follows.

.. code::

    {
        "work_dir": "/Users/ww5/olm/mox_quick/_work",
        "static": {...},
        "perms": [
            {
                "static": {...},
                "comp": {...},
                "time": {...},
                "state": {...},
                "dynamic": {...},
                "input_file": "perms/9a1cb1ff6fdee2d9825ef4086f6f42a3/model_6f42a3.inp",
                "_": {...}
            },
            {
                "static": {...},
                "comp": {...},
                "time": {...},
                "state": {...},
                "dynamic": {...},
                "input_file": "perms/9b1c4e1794359aaa8b55cee6cbe4e281/model_e4e281.inp",
                "_": {...}
            },
            {...},
            {...},
            {...},
            {...}
        ]
    }

At the top level there are the following keys:

- :code:`work_dir` - the working directory where all files are written
- :code:`static` - a dictionary of static variables which are the same for all permutations
- :code:`perms` - a list of permutations

Inside each element of the :code:`perms` list are the variables which are available when
the input file is written. Variables may be organized hierarchically. Even though
the static variables do not change for each permutation, they are copied into each permutation
data set so that all data available for a permutation is available locally in each
list element.

Possible implementation
^^^^^^^^^^^^^^^^^^^^^^^

By the time the above data is returned from a generate function, the input files have
already been written and at this stage, we do not retain how these inputs were generated.
It could be from a template system like Jinja or any other system. In the case of a
template system, like Jinja, the input files should be able to be recreated with the 
following code.

.. code::
    
    # For a specific generator, f.
    y = generate.f(x)

    # Within the generate function is the equivalent of the following code to generate
    # the input_file with the data contained in the permutation.
    for perm in y["perms"]:
        input_file = expand_template(template_file, data=perm)

See the :obj:`jt_expander` for a concrete function which follows this specification.
     
"""

# Prepend these type hints with _ to prevent showing in autocomplete as members of
# this package.
from typing import Union as _Union

_Static = dict[str, any]
_States = dict[str, any]
_OneComp = dict[str, any]
_NestedComp = dict[str, _OneComp]
_Time = dict[str, any]
_Dynamic = dict[str, any]


def jt_expander(
    template: str,
    static: _Static,
    states: _States,
    comp: _Union[_OneComp, _NestedComp],
    time: _Time,
    _model={},
    _env={},
    dynamic: _Union[_Dynamic, None] = None,
):
    """Expand a template with the result of user-specified operations.

    There are two groups of arguments. The first argument for the template file name
    contains the Jinja directives and variables. The remainder of the arguments are
    special dictionaries which are processed according to a special :code:`_type`
    directive. See examples below.

    The processing results in a number of data sets defined. Each one of these data
    packets is written to disk before it is substituted into the template. In this way,
    during model development when the typical "variable not found" errors are encountered,
    one can inspect this file for the actual variables available.

    .. note::

        The available data for template expansion is the *output* of running the
        function specified by :code:`_type` on the input passed to this function.
        See :obj:`scale.olm.create` for details.

    Args:

        template: Template file name relative to the directory containing
                  :code:`_env['config_file']`.

        static: Data dictionary that is independent of state. Data is passed through function
                :code:`_type` and then available as :code:`static.<key>` to
                Jinja for template expansion. See :obj:`scale.olm.internal.pass_through`
                for an example.

        states: Data dictionary that defines how to generate a number of states, each one
                leading to a permutation of the input model. Data is passed through
                :code:`_type` and then available as :code:`state.<key>` to Jinja for
                template expansion. Note that the input definition of multiple *states*
                becomes a list of *state*. For example, a states definition with
                two enrichments and three moderator densities, using the full_hypercube
                expansion, would lead to six states. Jinja is called six times with the
                data that results from each one of those states.
                See :obj:`scale.olm.generate.full_hypercube` for an example.

        comp: Data dictionary that defines the compositions. If more than one composition
              is needed, comp may be a dictionary of dictionaries, where the key for each
              dictionary is the name of the composition. Data is passed through
              :code:`_type` and then available as :code:`state.<key>` to Jinja for
              template expansion. For example, the following comp dictionary defines
              an inner and outer composition that will be available to the Jinja
              expansion as :code:`comp.inner` and :code:`comp.outer`.

              .. code::

                  comp = {
                      "inner": {"_type": "scale.olm.generate.comp:uo2_simple", ... },
                      "outer": {"_type": "scale.olm.generate.comp:uo2_simple", ... },
                 }

              See :obj:`scale.olm.generate.comp` functions such as
              :obj:`scale.olm.generate.comp:uo2_simple` for examples.

    """
    import scale.olm.internal as internal
    import scale.olm.core as core
    from pathlib import Path
    import numpy as np
    import math
    import json
    import copy
    import shutil

    internal.logger.info(f"Generating with scale.olm.jt_expander ...")

    # Handle parameters.
    static2 = internal._fn_redirect(**static)

    # Generate a list of states from the state specification.
    states2 = internal._fn_redirect(**states)

    # Useful paths.
    if not "work_dir" in _env:
        work_path = Path.cwd() / "_work"
    else:
        work_path = Path(_env["work_dir"])
    generate_path = work_path / "perms"

    # Load the template file.
    if not "config_file" in _env:
        template_path = template
    else:
        template_path = Path(_env["config_file"]).parent / template
    with open(template_path, "r") as f:
        template_text = f.read()

    internal.logger.info(
        "Expanding into permutations", template=str(template_path), nperms=len(states2)
    )

    # Create all the permutation information.
    perms2 = []
    i = 0
    td = core.TempDir()
    for state2 in states2:
        # For each state, generate the compositions.
        comp2 = {}
        if "_type" in comp:
            comp2 = internal._fn_redirect(**comp, state=state2)
        else:
            for k, v in comp.items():
                comp2[k] = internal._fn_redirect(**v, state=state2)

        # Handle dynamic (state-dependent) parameters.
        dynamic2 = {}
        if dynamic:
            for k, v in dynamic.items():
                dynamic2[k] = internal._fn_redirect(**v, state=state2)

        # For each state, generate a time list.
        time2 = internal._fn_redirect(**time, state=state2)

        # Generate all data.
        data = {
            "static": static2,
            "comp": comp2,
            "time": time2,
            "state": state2,
            "dynamic": dynamic2,
        }

        # Write data to a temporary file to get a hash of the contents.
        tf = td.write_file(json.dumps(data, indent=4), "temp.json")
        data_hash = core.FileHasher(tf).id

        # Save some info.
        input_path = generate_path / data_hash / ("model_" + data_hash[-6:] + ".inp")
        input_file = str(input_path.relative_to(work_path))
        data["input_file"] = input_file
        data["file"] = data["input_file"]  # deprecated alias
        data_path = input_path.parent / "data.olm.json"
        i += 1
        data_file = str(data_path.relative_to(work_path))
        data["_"] = {"model": _model, "data_hash": data_hash, "data_file": data_file}

        # Write the data file in the actual directory with input and hash added. This
        # is mainly so a user can see the data that is available for template expansion
        # beside a copy of the template.
        input_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "w") as f:
            json.dump(data, f, indent=4)

        # Expand the template and write the input to disk.
        internal.logger.info("Writing permutation", index=i, input_file=input_file)
        filled_text = core.TemplateManager.expand_text(
            template_text, data, src_path=str(template_path)
        )
        with open(input_path, "w") as f:
            f.write(filled_text)

        # Return the final thing in a permutations list.
        perms2.append(data)

    internal.logger.info(f"Finished generating with scale.olm.jt_expander!")

    return {"work_dir": _env["work_dir"], "perms": perms2, "static": static2}
