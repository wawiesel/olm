{{"" if check.test_pass else "FAILING "}}{{model.name}}
------------------------------------------------------------------------------------------

{% if not check.test_pass %}
.. warning::
    This library has failing checks. See below for details.
{% endif %}

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
    :widths: 30 70
    :header-rows: 1

    *   - name
        - grid
    {% for k,v in build.space.items() %}
    *   - {{k}}
        - {{v.grid-}}
    {% endfor %}


Consistency Check
~~~~~~~~~~~~~~~~~

.. list-table:: Consistency check summary
    :widths: 50 50
    :header-rows: 1

    *   - name
        - value
    {% for k,v in check.sequence[0].items() %}
    {% if not v is mapping %}
    *   - {{k}}
        - {{v-}}
    {% endif %}
    {% endfor %}

Histogram of relative versus absolute errors for all nuclides at all states and times.

..  image:: check/check-origami/hist.png
    :width: 90%


Nuclide checks
^^^^^^^^^^^^^^

These show the consistency between the high-order (:code:`hi=TRITON`) and low-order (:code:`lo=ORIGAMI`)
solutions. Each plot shows the range of the error across all permutations in the interpolation
space.

{% for k,v in check.sequence[0].nuclide_compare.items() %}
..  image:: {{v.image}}
    :width: 90%
{%- endfor %}


Model info
~~~~~~~~~~

This model introduces the following static parameters: {{generate.params.keys()|list}},
with values shown in the table below.

.. list-table:: Static model parameters
    :widths: 50 50
    :header-rows: 1

    *   - name
        - value
    {% for k,v in generate.params.items() %}
    *   - {{k}}
        - {{v-}}
    {% endfor %}

.. list-table:: Run summary data
    :widths: 50 25 25
    :header-rows: 1

    *   - output
        - success
        - runtime (hrs)
    {%- for row in run.perms %}
    *   - {{row.output}}
        - {{row.success}}
        - {{row.runtime_hrs}}
    {%- endfor %}


.. raw:: pdf

      PageBreak oneColumn

Example Generated Input
~~~~~~~~~~~~~~~~~~~~~~~

This is the TRITON input (:code:`perm00.inp`) for the first permutation out of {{run.perms|length}}.

.. include:: perm00/perm00.inp
    :literal:

