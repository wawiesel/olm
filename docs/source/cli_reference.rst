.. _cli-reference:

CLI Reference
-------------

The main end-user functionality in OLM is the comand line interface (CLI) provided
by the :code:`olm` Python utility. This section documents the
various modes available.

.. note::

    For documentation of the :code:`scale.olm` Python package, see :ref:`api-reference`.


.. click:run::
	from scale.olm.__main__ import olm
	result = invoke(olm, args=["--help"])


.. click:: scale.olm.__main__:olm_init
  :prog: olm init

.. click:: scale.olm.__main__:olm_create
  :prog: olm create

.. click:: scale.olm.__main__:olm_install
  :prog: olm install

.. click:: scale.olm.__main__:olm_link
  :prog: olm link

.. click:: scale.olm.__main__:olm_check
  :prog: olm check