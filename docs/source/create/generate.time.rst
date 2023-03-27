generate.time
^^^^^^^^^^^^^

The time section determines the time grid for all calculations. A typical time grid
for a production ORIGEN reactor library calculation may look like this.

.. code:: JSON

	"time": {
		"_type": "scale.olm.generate.time:constpower_burndata",
		"gwd_burnups": [
			0.0,
			0.04,
			1.04,
			3.0,
			5.0,
			7.5,
			10.5,
			13.5,
			16.5,
			19.5,
			22.5,
			25.5,
			28.5,
			31.5,
			34.5,
			37.5,
			40.5,
			43.5,
			46.5,
			49.5,
			52.5,
			55.5,
			58.5,
			61.5,
			64.5,
			67.5,
			70.5,
			73.5,
			76.5,
			79.5,
			82.5
		]
	}

------------------------------------------------------------------------------------------

.. include:: schema/scale.olm.generate.time.constpower_burndata.rst
