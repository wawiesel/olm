{% macro qscore(value) -%}{{"{:.3f}".format(value)}}{%- endmacro %}
{% macro compact(value) -%}{{"{:.4g}".format(value)}}{%- endmacro %}
{% macro percent(value) -%}{{"{:.3g}".format(value)}}{%- endmacro %}
{% macro sci(value) -%}{{"{:.3e}".format(value)}}{%- endmacro %}
{% macro count(value) -%}{{"{:.0f}".format(value)}}{%- endmacro %}
{% macro pass_text(value) -%}{{"pass" if value else "fail"}}{%- endmacro %}
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


.. raw:: pdf

      PageBreak oneColumn

Consistency Check
~~~~~~~~~~~~~~~~~

.. class:: olm-summary-table

.. list-table:: Consistency check summary
    :widths: 58 42
    :header-rows: 1

    *   - name
        - value
    *   - overall
        - {{pass_text(check.sequence[0].test_pass)}}
    *   - q1
        - {{qscore(check.sequence[0].q1)}} / {{qscore(check.sequence[0].target_q1)}}
    *   - q2
        - {{qscore(check.sequence[0].q2)}} / {{qscore(check.sequence[0].target_q2)}}
    {% if check.sequence[0].test_pass_time is defined %}
    *   - time q-scores
        - {{pass_text(check.sequence[0].test_pass_time)}}
    {% endif %}
    *   - metric
        - {{check.sequence[0].metric}}
    {% if check.sequence[0].units is defined %}
    *   - units
        - {{check.sequence[0].units}}
    {% endif %}
    *   - eps0
        - {{sci(check.sequence[0].eps0)}}
    *   - epsa
        - {{sci(check.sequence[0].epsa)}}
    *   - epsr
        - {{sci(check.sequence[0].epsr)}}
    *   - values
        - {{count(check.sequence[0].m)}}
    *   - relative failures
        - {{count(check.sequence[0].wr)}} ({{percent(100.0 * check.sequence[0].fr)}}%)
    *   - absolute failures
        - {{count(check.sequence[0].wa)}} ({{percent(100.0 * check.sequence[0].fa)}}%)

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

{% if check.sequence[0].worst_time_quality is defined and check.sequence[0].worst_time_quality %}
{% set worst_time = check.sequence[0].worst_time_quality %}
{% set first_failed_time = namespace(row=none) %}
{% if check.sequence[0].first_failed_time_quality is defined and check.sequence[0].first_failed_time_quality %}
{% set first_failed_time.row = check.sequence[0].first_failed_time_quality %}
{% elif check.sequence[0].time_quality is defined %}
{% for row in check.sequence[0].time_quality %}
{% if not first_failed_time.row and not row.test_pass %}
{% set first_failed_time.row = row %}
{% endif %}
{% endfor %}
{% endif %}

.. class:: olm-quality-location-table

.. list-table:: Time-quality locations
    :widths: 24 16 22 13 13 12
    :header-rows: 1

    *   - statistic
        - days
        - GWd/MTIHM
        - q1
        - q2
        - limit
    {% if first_failed_time.row %}
    *   - first fail
        - {{compact(first_failed_time.row.time_days)}}
        - {% if first_failed_time.row.burnup_gwd_per_mtihm is defined %}{{compact(first_failed_time.row.burnup_gwd_per_mtihm)}}{% endif %}
        - {{qscore(first_failed_time.row.q1)}}
        - {{qscore(first_failed_time.row.q2)}}
        - {{first_failed_time.row.limiting_score}}
    {% endif %}
    *   - worst shortfall
        - {{compact(worst_time.time_days)}}
        - {% if worst_time.burnup_gwd_per_mtihm is defined %}{{compact(worst_time.burnup_gwd_per_mtihm)}}{% endif %}
        - {{qscore(worst_time.q1)}}
        - {{qscore(worst_time.q2)}}
        - {{worst_time.limiting_score}}
{% endif %}

.. class:: olm-qscore-table

.. list-table:: Q-score by high-order time
    :widths: 16 23 14 14 13
    :header-rows: 1

    *   - days
        - GWd/MTIHM
        - q1
        - q2
        - pass
    {% for row in check.sequence[0].time_quality %}
    *   - {{compact(row.time_days)}}
        - {% if row.burnup_gwd_per_mtihm is defined %}{{compact(row.burnup_gwd_per_mtihm)}}{% endif %}
        - {{qscore(row.q1)}}
        - {{qscore(row.q2)}}
        - {{pass_text(row.test_pass)}}
    {% endfor %}
{% endif %}

{% if check.sequence[0].convergence_time_quality is defined %}
Convergence Q Scores
^^^^^^^^^^^^^^^^^^^^

For each convergence run, q1 and q2 are the minimum values over all high-order
time points.

..  image:: {{check.sequence[0].convergence_time_quality_image}}
    :width: 90%

.. class:: olm-convergence-table

.. list-table:: Worst q-score by convergence run
    :widths: 11 13 13 13 16 34
    :header-rows: 1

    *   - nlib
        - nburn
        - q1
        - q2
        - pass
        - time/burnup
    {% for row in check.sequence[0].convergence_time_quality %}
    *   - {{row.nlib}}
        - {{row.nburn}}
        - {{qscore(row.q1)}}
        - {{qscore(row.q2)}}
        - {{row.pass}}
        - {{compact(row.time_days)}} d{% if row.burnup_gwd_per_mtu is defined %}, {{compact(row.burnup_gwd_per_mtu)}} GWd/MTU{% endif %}
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

.. class:: olm-static-table

.. list-table:: Static model parameters
    :widths: 50 50
    :header-rows: 1

    *   - name
        - value
    {% for k,v in generate.static.items() %}
    *   - {{k}}
        - {{v-}}
    {% endfor %}

.. raw:: pdf

      PageBreak oneColumn

.. class:: olm-run-table

.. list-table:: Run summary data
    :widths: 58 18 24
    :header-rows: 1

    *   - output
        - success
        - runtime (hrs)
    {%- for row in run.runs %}
    *   - :code:`{{row.output_file.split('/')[-1]}}`
        - {{pass_text(row.success)}}
        - {% if row.runtime_hrs is number %}{{compact(row.runtime_hrs)}}{% else %}{{row.runtime_hrs}}{% endif %}
    {%- endfor %}


.. raw:: pdf

      PageBreak oneColumn

Example Generated Input
~~~~~~~~~~~~~~~~~~~~~~~

This is the TRITON input (:code:`{{run.runs[0].input_file.split('/')[-1]}}`) for the first permutation out of {{run.runs|length}}.

.. include:: {{run.runs[0].input_file}}
    :literal:
