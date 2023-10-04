Quickstart
----------

.. only:: html

	.. image:: quickstart.gif

This quickstart should take 5-10 minutes to complete, starting with a clean slate and
ending with a simple UOX ORIGEN reactor library.

.. code::

    =shell
    olm link mypin
    end

    =origami

    libs=[ mypin ]

    fuelcomp{
        uox(fuel){ enrich=4.95 }
        mix(1){ comps[fuel=100] }
    }

    modz = [ 0.74 ]
    pz = [ 1.0 ]

    hist[
      cycle{ power=40 burn=1000 nlib=10 }
    ]

    end

.. note:: **Generating** a ORIGEN reactor library as shown here requires
		  SCALE 6.3.2 and later. **Using a UOX** ORIGEN reactor library in ORIGAMI
		  as shown here only requires SCALE 6.2.4 at minimum. **Using a MOX** ORIGEN
		  reactor library in ORIGAMI requires SCALE 7.0.


Installing
~~~~~~~~~~

Installing OLM is simple.

.. code::

    pip install scale-olm


Initializing a New Library
~~~~~~~~~~~~~~~~~~~~~~~~~~

Initialize a new reactor library directory as follows.

.. code::

    olm init --variant=uox_quick mypin

You should see something similar to the following output.

.. code:: text

	2023-09-16 11:46:43 [info     ] Creating init dir              config_dir=mypin
	2023-09-16 11:46:43 [info     ] Copying config from            destination=mypin/config.olm.json source=/Users/ww5/olm/scale/olm/init/uox_quick/config.olm.json
	2023-09-16 11:46:43 [info     ] Copying files for variant=uox_quick destination=mypin/model.jt.inp source=/Users/ww5/olm/scale/olm/templates/model/triton/pin-uox.jt.inp
	2023-09-16 11:46:43 [info     ] Copying files for variant=uox_quick destination=mypin/report.jt.rst source=/Users/ww5/olm/scale/olm/templates/report/scale-short.jt.rst


OLM uses templates for SCALE input files which are populated with data defined in special
JSON files with extension :code:`.olm.json`. The :code:`data.olm.json` contains the data that defines
the reactor library, such as the maximum burnup to consider or the enrichment range.
The :code:`template.inp` contains a basic TRITON input template from which to generate all the
permutations needed.


Creating the Library
~~~~~~~~~~~~~~~~~~~~

.. note:: This example is set up so that the entire process runs in less than a minute on 3 cores.

The :code:`olm create --run` requires access to the SCALE runtime, :code:`scalerte`. Operations in
the :code:`--assemble` stage require access to :code:`OBIWAN` which ships with SCALE. Setting the
environment variable :code:`SCALE_DIR` will find them both.

.. code:: text

    export SCALE_DIR=/Applications/SCALE-6.3.2.app/Contents/Resources

With that set, you can create an ORIGEN reactor library with :code:`olm create`
and the path to the configuration file, :code:`config.olm.json`.

.. code:: shell

    olm create -j6 mypin/config.olm.json

You should see screen output as the calculation proceeds. All calculations happen in a
working directory which is by default :code:`_work` in the same parent directory as
the configuration file. The :code:`-j6` specifies to use 6 processes to generate the
library.

.. note:: To see more output you can set the environment variable :code:`SCALE_LOG_LEVEL`,
		  where 10 is DEBUG, 20 is INFO, 30 is WARNING, 40 is ERROR only. See
		  `structlog levels <https://docs.python.org/3/library/logging.html#logging-levels>`_
		  for details.


Using the Library in ORIGAMI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`olm link` command can take a specific path with :code:`-p`, or it can read
from a list of paths in the :code:`SCALE_OLM_PATH` variable. This variable is just like
the Linux :code:`PATH` variable or :code:`PYTHON_PATH` used to find modules. It is a
colon-separated (:) list of paths, searched first to last.

The recommended way to use this library is to first set the path variable.

.. code:: text

    export SCALE_OLM_PATH=$PWD/mypin/_work

Then use a simple link command like so.

.. code:: text

    =shell
    olm link mypin
    end

    =origami

    libs=[ mypin ]

    fuelcomp{
        uox(fuel){ enrich=4.95 }
        mix(1){ comps[fuel=100] }
    }

    modz = [ 0.74 ]
    pz = [ 1.0 ]

    hist[
      cycle{ power=40 burn=1000 nlib=10 }
    ]

    end
