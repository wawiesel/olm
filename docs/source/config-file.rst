.. _config.olm.json:

config.olm.json
---------------

The configuration file contains all the necessary data to create an ORIGEN Reactor
Library. It is a `JSON <https://www.w3schools.com/jsref/jsref_obj_json.asp>`_
file, by default called :ref:`config.olm.json`. At the top level, it has the following
sections.

.. code:: JSON

    {
        "model":{}
        "generate":{}
        "run":{}
        "assemble":{}
        "check":{}
        "report":{}
    }

.. toctree::
    :maxdepth: 3

    create/model.rst
    create/generate.rst
    create/run.rst
    create/assemble.rst
    create/check.rst
    create/report.rst



