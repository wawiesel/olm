generate
~~~~~~~~

OLM uses an approach to creating the necessary inputs for ORIGEN library generation which
uses templates and data files. 

Here's an example of a generate section of :ref:`config.olm.json` 
using the `Jinja <https://jinja.palletsprojects.com/en/3.1.x/>`_ template expansion 
method, :code:`jt_expander`.

.. code:: JSON

    "generate": {
        "_type": "scale.olm.generate.root:jt_expander",
        "template": "model.jt.inp",

        "comp": {
            "fuel":{
                "_type": "scale.olm.generate.comp:uo2_nuregcr5625",
                "density": 10.4
            }
        },

        "static": {
            "_type": "scale.olm.generate.static:pass_through",
            "addnux": 2,
            "xslib": "v7-56",
            "pitch": 1.26,
            "fuelr": 0.4095,
            "cladr": 0.4750
        },

        "states": {
            "_type": "scale.olm.generate.states:full_hypercube",
            "coolant_density": [
                0.70,
                0.72,
                0.74
            ],
            "enrichment": [
                0.5,
                3,
                5,
                7
            ],
            "specific_power": [
                40
            ],
            "boron_ppm": [ 600 ]
        },

        "time": {
            "_type": "scale.olm.generate.time:constpower_burndata",
            "gwd_burnups": [
                0.0,
                1.0,
                10.0,
                25.0,
                50.0,
                70.0,
                90.0
            ]
        }
    }

------------------------------------------------------------------------------------------

Each of the five sections (static, dynamic, comp, time, states) are described in more
detail in the sections below.

.. toctree::

	generate.static.rst
	generate.dynamic.rst
	generate.comp.rst
	generate.time.rst
	generate.states.rst

------------------------------------------------------------------------------------------

.. include:: schema/scale.olm.generate.root.jt_expander.rst
