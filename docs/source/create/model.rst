model
~~~~~

The model section defines aspects of the model such as a short description, sources, 
and other reference data. It is intended to provide the front matter for a report or
to inject in an input file traceability data. Only the :code:`name` field is used
within OLM.

.. code:: JSON

    "model": 
    {
        "name": "uox_quick",
        "description": "A 2D t-depl pin cell of W17x17 type.",
        "sources":
        {
            "1":"Characteristics of Spent Fuel, High-Level Waste, and other Radioactive Wastes which May Require Long-Term Isolation, Appendix 2A. Physical Descriptions of LWR Fuel Assemblies, DOE/RW-0184, Volume 3 of 6, U.S. DOE Office of Civilian Radioactive Waste Management, 1987.",
            "2": "SCALE: A Comprehensive Modeling and Simulation Suite for Nuclear Safety Analysis and Design, ORNL/TM-2005/39, Version 6.1, Oak Ridge National Laboratory, Oak Ridge, Tennessee, June 2011.",
            "3": "H. Smith, J. Peterson, and J. Hu, Fuel Assembly Modeling for the Modeling and Simulation Toolset, ORNL/LTR-2012-555 Rev. 1, Oak Ridge National Laboratory, 2013."
        },
        "revision":
        [
            "Unknown authorship",
            "2014 - Ported into SLIG by B. R. Betzler",
            "2023 - Ported into OLM by W. A. Wieselquist"
        ],
        "notes":
        [
            "Fuel density, gap gas pressure from Appendix 2A of Reference [1_].",
            "Temperatures, moderator density, boron concentration from Table D1.A.2 of Reference [2_].",
            "All other dimensions, materials, etc. from Reference [3_]."
        ]
    }
