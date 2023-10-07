generate.states
^^^^^^^^^^^^^^^

The states input section defines the various states that will be considered. These states
provide the information across the relevant space of interest to create an interpolatable
database of ORIGEN transition coefficient data.

.. code:: JSON

	"states": {
		"_type": "scale.olm.generate.states:full_hypercube",
		"coolant_density": [
			0.1,
			0.3,
			0.5,
			0.7,
			0.9
		],
		"enrichment": [
			0.5,
			1.5,
			2.0,
			3.0,
			4.0,
			5.0,
			6.0,
			7.0,
			8.0,
			8.5
		],
		"wtpt_gd": [
			3.0
		],
		"specific_power": [
			25.0
		]
	}

------------------------------------------------------------------------------------------

.. include:: schema/scale.olm.generate.states.full_hypercube.rst
