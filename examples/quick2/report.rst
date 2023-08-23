==========================================================================================
{{model.name}}
==========================================================================================

A 2D t-depl quarter assembly model for a Pressurized Water Reactor of the
Westinghouse 17x17 type.

SCALE version: {{run.version}}

Sources
-------

1. Characteristics of Spent Fuel, High-Level Waste, and other
   Radioactive Wastes which May Require Long-Term Isolation, Appendix 2A.
   Physical Descriptions of LWR Fuel Assemblies, DOE/RW-0184, Volume 3 of
   6, U.S. DOE Office of Civilian Radioactive Waste Management, 1987.
2. SCALE: A Comprehensive Modeling and Simulation Suite for Nuclear
   Safety Analysis and Design, ORNL/TM-2005/39, Version 6.1, Oak Ridge
   National Laboratory, Oak Ridge, Tennessee, June 2011.
3. H. Smith, J. Peterson, and J. Hu, Fuel Assembly Modeling for the
   Modeling and Simulation Toolset, ORNL/LTR-2012-555 Rev. 1, Oak Ridge
   National Laboratory, 2013.


Other Info
----------

- Fuel density, gap gas pressure from Appendix 2A of Reference 1.
- Temperatures, moderator density, boron concentration from Table D1.A.2 of Reference 2.
- All other dimensions, materials, etc. from Reference 3.


Revision Log
------------

- Rev 0: Unknown authorship
- Rev 1: Ported into SLIG, B. R. Betzler, June 2014
- Rev 2: Ported into OLM, W. A. Wieselquist, August 2023



Run Summary
-----------

The status of the various runs is show below.

.. list-table:: Run summary data
    :widths: 5 1
    :header-rows: 1

    *   - output
        - runtime (s)
    {% for x in tables.run_summary -%}
    *   {% for i in [0,1] -%}
        - {{x[i]}}
        {% endfor %}
    {% endfor -%}
    {{"    "}}

Input Summary
-------------

The various inputs to create this library are described below.

.. list-table:: Static summary data
    :widths: 1 1 5
    :header-rows: 1

    *   - name
        - value
        - description
    {% for x in tables.static_summary -%}
    *   {% for i in [0,1,2] -%}
        - {{x[i]}}
        {% endfor %}
    {% endfor -%}
    {{"    "}}




