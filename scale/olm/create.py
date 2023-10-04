"""
The scale.olm.create module collects the five modules which are part of the OLM
create mode. Those modules are:

 - :obj:`scale.olm.generate`
 - :obj:`scale.olm.run`
 - :obj:`scale.olm.assemble`
 - :obj:`scale.olm.check`
 - :obj:`scale.olm.report`

.. code::

   from scale.olm import create  # call as create.generate.f(x)
    
"""
import scale.olm.generate as generate
import scale.olm.run as run
import scale.olm.assemble as assemble
import scale.olm.check as check
import scale.olm.report as report
