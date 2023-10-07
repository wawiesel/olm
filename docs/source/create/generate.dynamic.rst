generate.dynamic
^^^^^^^^^^^^^^^^

Dynamic data are dependent on state in a simple way, e.g. interpolatable x/y pairs data.

The dynamic section in the :code:`config.olm.json` defines the different dynamic
variables as keys, e.g. "dancoff1" and "dancoff2" below. When the states are
expanded and each variable evaluated at each state, that final value is available
in the template file as :code:`dynamic.<KEY>`, i.e. :code:`dynamic.dancoff1` and
:code:`dynamic.dancoff2` would be defined (and potentially different)
for each state as a result of the definition below.

.. code:: JSON

	"dynamic": {
		"dancoff1": {
			"_type": "scale.olm.generate.dynamic:scipy_interp",
			"state_var": "coolant_density",
			"data_pairs": [
				[ 0.1, 0.4621 ],
				[ 0.3, 0.3368 ],
				[ 0.5, 0.2583 ],
				[ 0.7, 0.2044 ],
				[ 0.9, 0.1653 ]
			],
			"method": "linear"
		},
		"dancoff2": {
			"_type": "scale.olm.generate.dynamic:scipy_interp",
			"state_var": "coolant_density",
			"data_pairs": [
				[ 0.1, 0.2967 ],
				[ 0.3, 0.2194 ],
				[ 0.5, 0.1702 ],
				[ 0.7, 0.1359 ],
				[ 0.9, 0.1107 ]
			],
			"method": "linear"
		}
	}

The following sections show what would be *within* a single dynamic variable
block, based on the value of the "_type" within the block.

.. include:: schema/scale.olm.generate.dynamic.scipy_interp.rst


