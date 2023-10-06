Quickstart
----------

This quickstart should take 5-10 minutes to complete, starting with a clean slate and
ending with a simple UOX ORIGEN reactor library.

.. note::

	By default OLM logs info, warning, and errors to the screen. To only output warnings
	and errors set the :code:`SCALE_LOG_LEVEL=30`.


.. code:: console

	# Initialize a configuration file for the uox_quick variant.
	$ olm init --variant uox_quick

  	# Create an ORIGEN library.
  	$ olm create -j6 uox_quick/config.olm.json
  	
  	# Open the generated report.
  	$ open uox_quick/_work/uox_quick.pdf

------------------------------------------------------------------------------------------

Executing the above commands will allow one to run ORIGAMI with the newly 
created library as shown below.

.. literalinclude:: origami.inp
	:language: scale
	:emphasize-lines: 6-8
	:linenos:

.. code:: console

	# Allow the local library to be found by olm link.
	$ export SCALE_OLM_PATH=$PWD/uox_quick/_work
	
	# Run the ORIGAMI calculation.
	$ $SCALE_DIR/bin/scalerte -m origami.inp
	

------------------------------------------------------------------------------------------

.. note:: **Generating** a ORIGEN reactor library as shown here requires
		  SCALE 6.3.2 and later. **Using a UOX** ORIGEN reactor library in ORIGAMI
		  as shown here only requires SCALE 6.2.4 at minimum. However, **using a MOX** ORIGEN
		  reactor library in ORIGAMI requires SCALE 7.0.

