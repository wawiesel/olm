.. _user-home:

Welcome to the ORIGEN Library Manager (OLM)
===========================================

OLM is a command line utility that streamlines aspects of using the ORIGEN
library to solve nuclide inventory generation problems. Install OLM using pip.

.. code:: console
		
    $ pip install scale-olm
		
------------------------------------------------------------------------------------------

Overview
--------

OLM focuses on managing the ORIGEN reactor library, which is a special
collection of data that enables rapid spent fuel calculations with SCALE/ORIGAMI.
It has the following basic modes.

- :code:`init` : to initialize a new reactor library directory
- :code:`create` : to create a reactor library, according to these stages

    - :code:`--generate` to generate the various permutations into a number of SCALE input files, e.g.
      covering enrichment, moderator density, and burnup space
    - :code:`--run` to run SCALE (e.g. TRITON or Polaris) to generate output
      reaction/transition data for across the interpolation space
    - :code:`--assemble` to assemble the output reaction/transition data into an ORIGEN
      reactor library
    - :code:`--check` to check the quality of the final ORIGEN reactor library
    - :code:`--report` to make a report summarizing the new reactor library
- :code:`install` : to install an ORIGEN reactor library
- :code:`link` : to link an ORIGEN reactor library into a SCALE (ORIGAMI or ARP) calculation


.. toctree::
   :maxdepth: 2
   :caption: Contents:

    Quickstart <quickstart.rst>
    Step-by-step <step-by-step.rst>
    config.olm.json <config-file.rst>
    CLI Reference <cli_reference.rst>
    API Reference <api_reference.rst>
    Contributing <developers.md>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
