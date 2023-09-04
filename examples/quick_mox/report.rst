{{model.name}}
------------------------------------------------------------------------------------------

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
        Some human
    Rev. 1
        Another human 2022


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


Consistency Check
~~~~~~~~~~~~~~~~~

These show the consistency between the high-order (:code:`hi=TRITON`) and low-order (:code:`lo=ORIGAMI`)
solutions.

.. list-table::

    * - ..  figure:: {{check.sequence[0].nuclide_compare['0092235'].image}}
            :alt: U235 hi/lo error

            U-235 error as a function of time
            (max: {{'%.2f' | format(100*check.sequence[0].nuclide_compare['0092235'].max_diff0)}}%)

      - .. figure::  {{check.sequence[0].nuclide_compare['0094239'].image}}
            :alt: Pu239 hi/lo error

            Pu-239 error as a function of time
            (max: {{'%.2f' | format(100*check.sequence[0].nuclide_compare['0094239'].max_diff0)}}%)


Model info
~~~~~~~~~~

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





