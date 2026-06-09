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
    {% if not (v is mapping) and not (v is sequence and not (v is string)) %}
    *   - {{k}}
        - {% if v is number %} {{'{:0.3g}'.format(v)-}} {% else %} {{v-}} {% endif %}
    {% endif %}
    {% endfor %}

{% if check.sequence[0].hist_image is defined %}
Histogram of relative versus absolute errors for all nuclides at all states and times.

..  image:: {{check.sequence[0].hist_image}}
    :width: 90%
{% endif %}

{% if check.sequence[0].time_quality is defined %}
Q Scores By Time
^^^^^^^^^^^^^^^^

The q1 and q2 scores are recalculated independently at each high-order time point.

..  image:: {{check.sequence[0].time_quality_image}}
    :width: 90%

{% if check.sequence[0].worst_time_quality is defined %}
:Worst time q-score: {{check.sequence[0].worst_time_quality.limiting_score}}
:Worst time: {{'{:0.6g}'.format(check.sequence[0].worst_time_quality.time_days)}} days
{% if check.sequence[0].worst_time_quality.burnup_gwd_per_mtihm is defined %}
:Worst burnup: {{'{:0.6g}'.format(check.sequence[0].worst_time_quality.burnup_gwd_per_mtihm)}} GWd/MTIHM
{% endif %}
:Worst score value: {{'{:0.3g}'.format(check.sequence[0].worst_time_quality.limiting_score_value)}}
:Worst score target: {{'{:0.3g}'.format(check.sequence[0].worst_time_quality.limiting_score_target)}}
:Worst score shortfall: {{'{:0.3g}'.format(check.sequence[0].worst_time_quality.limiting_score_shortfall)}}
{% endif %}

.. list-table:: Q-score by high-order time
    :widths: 20 20 20 20 20
    :header-rows: 1

    *   - time (days)
        - burnup (GWd/MTIHM)
        - q1
        - q2
        - pass
    {% for row in check.sequence[0].time_quality %}
    *   - {{'{:0.6g}'.format(row.time_days)}}
        - {% if row.burnup_gwd_per_mtihm is defined %}{{'{:0.6g}'.format(row.burnup_gwd_per_mtihm)}}{% endif %}
        - {{'{:0.3g}'.format(row.q1)}}
        - {{'{:0.3g}'.format(row.q2)}}
        - {{row.test_pass}}
    {% endfor %}
{% endif %}

{% if check.sequence[0].convergence_time_quality is defined %}
Convergence Q Scores
^^^^^^^^^^^^^^^^^^^^

For each convergence run, q1 and q2 are the minimum values over all high-order
time points.

..  image:: {{check.sequence[0].convergence_time_quality_image}}
    :width: 90%

.. list-table:: Worst q-score by convergence run
    :widths: 12 12 14 14 14 34
    :header-rows: 1

    *   - nlib
        - nburn
        - q1
        - q2
        - pass
        - time
    {% for row in check.sequence[0].convergence_time_quality %}
    *   - {{row.nlib}}
        - {{row.nburn}}
        - {{'{:0.3g}'.format(row.q1)}}
        - {{'{:0.3g}'.format(row.q2)}}
        - {{row.pass}}
        - {{'{:0.6g}'.format(row.time_days)}} days{% if row.burnup_gwd_per_mtu is defined %}, {{'{:0.6g}'.format(row.burnup_gwd_per_mtu)}} GWd/MTU{% endif %}
    {% endfor %}
{% endif %}


{% if check.sequence[0].nuclide_compare is defined %}
Nuclide checks
^^^^^^^^^^^^^^

These show the consistency between the high-order (:code:`hi=TRITON`) and low-order (:code:`lo=ORIGAMI`)
solutions. Each plot shows the range of the error across all permutations in the interpolation
space.

{% for k,v in check.sequence[0].nuclide_compare.items() %}
..  image:: {{v.image}}
    :width: 90%
{%- endfor %}
{% endif %}


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
