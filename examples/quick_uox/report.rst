{{model.name}}
--------------

:Name: {{model.name}}
:Description: {{model.description}}
:Date: {{build.date}}
:SCALE: v{{run.version}}
:Runtime: {{run.total_runtime_hrs}} cpu-hours
:Sources:
    {% for k,v in model.sources.items() %}
    .. _{{k}}:

    [{{k}}] {{v}}

    {% endfor %}
:Revision Log:
    Rev. 0
        Unknown authorship
    Rev. 1
        Ported into SLIG, B. R. Betzler, June 2014
    Rev. 2
        Ported into OLM, W. A. Wieselquist, August 2023


.. list-table:: Interpolation Space
    :widths: 1 2 3
    :header-rows: 1

    *   - name
        - description
        - grid
    {% for k,v in build.space.items() %}
    *   - {{k}}
        - {{v.desc}}
        - {{v.grid-}}
    {% endfor %}


Model info
~~~~~~~~~~

The sources used in this model are summarized as follows.

- Fuel density, gap gas pressure from Appendix 2A of Reference [1_].
- Temperatures, moderator density, boron concentration from Table D1.A.2 of Reference [2_].
- All other dimensions, materials, etc. from Reference [3_].

This model introduces the following static parameters: {{generate.params.keys()|list}},
with values shown in the table below.

.. list-table:: Static model parameters
    :widths: 1 1
    :header-rows: 1

    *   - name
        - value
    {% for k,v in generate.params.items() %}
    *   - {{k}}
        - {{v-}}
    {% endfor %}

.. list-table:: Run summary data
    :widths: 3 1 1
    :header-rows: 1

    *   - output
        - success
        - runtime (hrs)
    {%- for row in run.perms %}
    *   - {{row.output}}
        - {{row.success}}
        - {{row.runtime_hrs}}
    {%- endfor %}





