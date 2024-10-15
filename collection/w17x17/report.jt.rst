{{"" if check.test_pass else "FAILING "}}{{model.name}}
------------------------------------------------------------------------------------------

{% if not check.test_pass %}
.. warning::
    This library has failing checks. See below for details.
{% endif %}

:Name: {{model.name}}
:Description: {{model.description}}
:Date: {{assemble.date}}
:SCALE: v{{run.version}}
:Runtime: {{run.total_runtime_hrs}} cpu-hours
:Sources:
    {% for k,v in model.sources.items() %}
    .. _{{k}}:

    [{{k}}] {{v}}

    {% endfor %}
:Revision Log:
    {% for rev in model.revision %}
    Rev. {{loop.index}}
         {{rev}}
    {% endfor %}


.. list-table:: Interpolation Space
    :widths: 30 70
    :header-rows: 1

    *   - name
        - grid
    {% for k,v in assemble.space.items() %}
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
        - {% if v is number %} {{'{:0.3g}'.format(v)-}} {% else %} {{v-}} {% endif %}
    {% endif %}
    {% endfor %}


Model info
~~~~~~~~~~

This model is based on the following information.

{% for note in model.notes %}
    * {{note}}
{%- endfor %}


This model introduces the following static parameters: {{generate.static.keys()|list}},
with values shown in the table below.

.. list-table:: Static model parameters
    :widths: 50 50
    :header-rows: 1

    *   - name
        - value
    {% for k,v in generate.static.items() %}
    *   - {{k}}
        - {{v-}}
    {% endfor %}

.. list-table:: Run summary data
    :widths: 55 25 20
    :header-rows: 1

    *   - output
        - success
        - runtime (hrs)
    {%- for row in run.runs %}
    *   - :code:`{{row.output_file}}`
        - {{row.success}}
        - {{row.runtime_hrs}}
    {%- endfor %}


.. raw:: pdf

      PageBreak oneColumn

Example Generated Input
~~~~~~~~~~~~~~~~~~~~~~~

This is the TRITON input (:code:`{{run.runs[0].input_file}}`) for the first permutation out of {{run.runs|length}}.

.. include:: {{run.runs[0].input_file}}
    :literal:

