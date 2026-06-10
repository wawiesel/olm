check
~~~~~

Checking executes a sequence of checks on the quality of the ORIGEN reactor library.

Check templates may be local files beside ``config.olm.json`` or packaged templates
under ``scale/olm/templates``. Packaged template names are relative to that root,
for example ``model/origami/lumped0d-uox.jt.inp``.

``LowOrderConsistency`` can include an optional ``convergence`` block inside the
check sequence item. When that block is present, the configured low-order
template receives ``convergence_control.nlib`` and
``convergence_control.nburn``. Templates that use convergence must render those
values where ORIGAMI requires them. When the ``convergence`` block is omitted,
those fields are not provided and the template should not reference them.

.. toctree::

	schema/scale.olm.check.GridGradient.rst
	schema/scale.olm.check.LowOrderConsistency.rst

------------------------------------------------------------------------------------------

.. include:: schema/scale.olm.check.sequencer.rst
 
