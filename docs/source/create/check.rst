check
~~~~~

Checking executes a sequence of checks on the quality of the ORIGEN reactor library.

Check templates may be local files beside ``config.olm.json`` or packaged templates
under ``scale/olm/templates``. Packaged template names are relative to that root,
for example ``model/origami/system-uox.jt.inp``.

When a ``LowOrderConsistency`` check uses a ``convergence`` block, its configured
low-order template must use ``check.convergence.nlib`` and
``check.convergence.nburn`` where ORIGAMI requires those values. Those fields are
not provided when the ``convergence`` block is omitted.

.. toctree::

	schema/scale.olm.check.GridGradient.rst
	schema/scale.olm.check.LowOrderConsistency.rst

------------------------------------------------------------------------------------------

.. include:: schema/scale.olm.check.sequencer.rst
 
