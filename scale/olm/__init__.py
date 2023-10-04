"""
Welcome to the ORIGEN Library Manager (OLM) Python package!

The recommended usage is as follows. 

.. code::
	
	from scale.olm import core.BurnupHistory

The :code:`core` is only one of the many modules provided by OLM. The other 
modules are:

	- :obj:`scale.olm.core` for core classes
	- :obj:`scale.olm.create` for ORIGEN reactor library creation functions
	- :obj:`scale.olm.link` for ORIGEN reactor library linking functions
	- :obj:`scale.olm.complib` for a library of nuclear fuel composition functions
	- :obj:`scale.olm.contrib` for any contributed utility functions
	- :obj:`scale.olm.internal` for internal, private functions

See their respective documentation for the available classes and functions.

"""

import scale.olm.core as core
import scale.olm.create as create
import scale.olm.link as link
import scale.olm.complib as complib
import scale.olm.contrib as contrib
import scale.olm.internal as internal


internal.logger.debug("Initialized " + __name__)
