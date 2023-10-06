"""
Welcome to the ORIGEN Library Manager (OLM) Python package!

The main feature of this package is the command line tool :code:`olm`
which has its own help screens. However there is a ton of useful code
that :code:`olm` depends on that you may find useful from within your own
Python scripts or notebooks.

.. code::
	
	import scale.olm as olm    # use as olm.core.BurnupHistory

The :obj:`scale.olm.core` contains core classes used throughout OLM.

Additionally, there is a set of five modules for ORIGEN library creation, used by
:code:`olm create`. These contain functions that are called as part of the library creation 
process, based on contents of the model configuration file. 

	- :obj:`scale.olm.generate` for input generation functions (include submodules)
	- :obj:`scale.olm.run` for SCALE running functions
	- :obj:`scale.olm.assemble` for library assembly functions
	- :obj:`scale.olm.check` for checking functions
	- :obj:`scale.olm.report` for report generation functions

There is a module used by :code:`olm link` to link locally stored libraries into 
SCALE calculations.

	- :obj:`scale.olm.link` for library linking functions

Finally there are two additional modules for miscellaneous code.

	- :obj:`scale.olm.contrib` for any contributed utility functions
	- :obj:`scale.olm.internal` for internal, private functions

See their respective documentation for details.

"""

# core classes
import scale.olm.core as core

# functions related to creating ORIGEN reactor libraries
import scale.olm.generate as generate
import scale.olm.run as run
import scale.olm.assemble as assemble
import scale.olm.check as check
import scale.olm.report as report

# miscellaneous
import scale.olm.contrib as contrib
import scale.olm.internal as internal

internal.logger.debug("Initialized " + __name__)
