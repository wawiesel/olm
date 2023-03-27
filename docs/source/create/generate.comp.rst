generate.comp
^^^^^^^^^^^^^

The composition (comp) section is used to initialize the assembly designs, typically
dependent on a state variable like :code:`enrichment` or :code:`pu239_frac`.
The composition specification is very short because it is relying on the :code:`_type`
and the state variables to transform a simple enrichment of fissile content into a 
full nuclide specification. The example below defines two different compositions,
each one with a slightly different density.

.. code:: JSON

	"comp": {
		"inner": {
			"_type": "scale.olm.generate.comp:uo2_simple",
			"density": 10.42
		},
		"outer": {
			"_type": "scale.olm.generate.comp:uo2_simple",
			"density": 10.38
		}
	}

------------------------------------------------------------------------------------------

.. include:: schema/scale.olm.generate.comp.uo2_simple.rst

.. include:: schema/scale.olm.generate.comp.uo2_vera.rst

.. include:: schema/scale.olm.generate.comp.uo2_nuregcr5625.rst

.. include:: schema/scale.olm.generate.comp.mox_ornltm2003_2.rst

.. include:: schema/scale.olm.generate.comp.mox_multizone_2023.rst



