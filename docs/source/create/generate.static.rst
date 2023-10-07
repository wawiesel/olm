generate.static
^^^^^^^^^^^^^^^

Static data are fixed within a configuration (independent of state).
Here's an example of a static section (within the generate section).

.. code:: JSON

	"static": {
		"_type": "scale.olm.generate.static:pass_through",
		"addnux": 2,
		"xslib": "v7-56",
		"pitch": 1.26,
		"fuelr": 0.4095,
		"cladr": 0.4750
	}

There's no processing applied and these variables are directly available for use
in template files.

------------------------------------------------------------------------------------------
        
.. include:: schema/scale.olm.generate.static.pass_through.rst
