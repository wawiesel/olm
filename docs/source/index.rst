Welcome to the ORIGEN Library Manager (OLM)
===========================================

OLM is a command line utility that streamlines aspects of using the ORIGEN
library to solve nuclide inventory generation problems.

.. code::

    pip install scale-olm


.. toctree::
   :maxdepth: 2
   :caption: Contents:

    Quickstart <quickstart.rst>
    Step-by-step <step-by-step.rst>
    For OLM Developers <developers.md>
    Code Reference <code_reference.rst>


Overview
--------

OLM currently centers around managing the ORIGEN reactor library, which is a special
collection of data that enables rapid spent fuel calculations with SCALE/ORIGAMI.

It has four basic modes.

- :code:`init` : to initialize a new reactor library directory
- :code:`create` : to create a reactor library, according to these stages

    - :code:`--generate` to generate the various permutations into a number of SCALE input files, e.g.
      covering enrichment, moderator density, and burnup space
    - :code:`--run` to run the SCALE inputs to generate output reaction/transition data for each
      point in the space
    - :code:`--compile` to compile the output reaction/transition data into an ORIGEN reactor library
    - :code:`--check` to check the quality of the final ORIGEN reactor library
    - :code:`--report` to make a report summarizing the new reactor library
- :code:`install` : to install an ORIGEN reactor library
- :code:`link` : to link an ORIGEN reactor library into a SCALE calculation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
