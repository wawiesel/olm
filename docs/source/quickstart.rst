Quickstart
----------

.. image:: quickstart.gif

This quickstart should take 5-10 minutes to complete, starting with a clean slate and
ending with a simple UOX ORIGEN reactor library that can be used in ORIGAMI from
SCALE 6.2.4 and later, as shown below.

.. code::

    =shell
    olm link uox_w17x17_pin
    end

    =origami

    libs=[ uox_w17x17_pin ]

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

.. note:: Generating the ORIGEN reactor library as shown here requires tools in SCALE 6.3.2 and later.


Installing
~~~~~~~~~~

Installing OLM is simple.

.. code::

    pip install scale-olm


Initializing a New Library
~~~~~~~~~~~~~~~~~~~~~~~~~~

Initialize a new reactor library directory as follows.

.. code::

    olm init --template quick_uox --name w17x17_pin examples/quick_uox

Inspect the files that have been created

.. code::

    tree examples/1

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

.. code::

    export SCALE_DIR=/Applications/SCALE-6.3.2.app/Contents/Resources

With that set, you can create an ORIGEN reactor library.

.. code::

    olm create examples/quick_uox/data.olm.json

You should see screen output as the calculation proceeds. In the final stage of reporting,
a PDF document is created. Open this document which contains a summary of the reactor
library.

Using the Library in ORIGAMI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This particular example requested a set of checking calculations running ORIGAMI to
confirm that the lower-order ORIGAMI method is giving a similar result to the higher-order
TRITON method. Inspect one of the inputs at :code:`check/check-origami`
to see how it links to the local library.

.. code::

    =shell
    olm link -p /Users/${USER}/examples/1/_work uox_w17x17_pin
    end

    =origami

    libs=[ w17x17_pin2 ]

    options{ ft71=all mtu=1.0 }

    fuelcomp{
        % uox(fuel){ enrich=0.5 }
        stdcomp(fuel){ base=uo2
            iso[
                92234=0.004450000936604491
                92235=0.5000000360977219
                92236=0.002300000110430568
                92238=99.49324996285524
            ]
        }
        mix(1){ comps[fuel=100] }
    }

    modz = [ 0.7 ]
    pz = [ 1.0 ]

    hist[
      cycle{ power=39.4934 burn=25.0 nlib=1 }
      cycle{ power=39.6148 burn=225.0 nlib=1 }
      cycle{ power=39.7587 burn=375.0 nlib=1 }
      cycle{ power=39.7962 burn=312.5 nlib=1 }
      cycle{ power=39.8139 burn=312.5 nlib=1 }
      cycle{ power=39.8257 burn=250.0 nlib=1 }
      cycle{ power=39.8328 burn=250.0 nlib=1 }
      cycle{ power=39.8382 burn=250.0 nlib=1 }
      cycle{ power=39.8423 burn=250.0 nlib=1 }
      cycle{ power=39.8455 burn=250.0 nlib=1 }
      cycle{ power=39.848 burn=250.0 nlib=1 }
    ]

    end

The :code:`olm link` command can take a specific path with :code:`-p`, or it can read
from a list of paths in the :code:`SCALE_OLM_PATH` variable. This variable is just like
the Linux :code:`PATH` variable or :code:`PYTHON_PATH` used to find modules. It is a
colon-separated (:) list of paths, searched first to last.

The recommended way to use this library is to first set the path variable.

.. code::

    export SCALE_OLM_PATH=/Users/${USER}/examples/1/_work

Then use a simple link command like so.

.. code::

    =shell
    olm link uox_w17x17_pin
    end

    =origami

    libs=[ uox_w17x17_pin ]

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
