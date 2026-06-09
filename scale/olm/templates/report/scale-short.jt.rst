{% macro qscore(value) -%}{{"{:.3f}".format(value)}}{%- endmacro %}
{% macro compact(value) -%}{{"{:.4g}".format(value)}}{%- endmacro %}
{% macro percent(value) -%}{{"{:.3g}".format(value)}}{%- endmacro %}
{% macro sci(value) -%}{{"{:.3e}".format(value)}}{%- endmacro %}
{% macro count(value) -%}{{"{:.0f}".format(value)}}{%- endmacro %}
{% macro summary_row(name, value) -%}{{"{:<20} {:>21}".format(name, value)}}{%- endmacro %}
{% macro location_row(name, days, burnup, q1, q2, limit) -%}{{"{:<14} {:>8.4g} {:>11} {:>7.3f} {:>7.3f} {:>5}".format(name, days, burnup, q1, q2, limit)}}{%- endmacro %}
{% macro time_quality_row(days, burnup, q1, q2, passed) -%}{{"{:>8.4g} {:>11} {:>7.3f} {:>7.3f} {:>6}".format(days, burnup, q1, q2, pass_text(passed))}}{%- endmacro %}
{% macro convergence_row(nlib, nburn, q1, q2, passed, days, burnup) -%}{{"{:>5.0f} {:>6.0f} {:>7.3f} {:>7.3f} {:>10} {:>8.4g} {:>10}".format(nlib, nburn, q1, q2, passed, days, burnup)}}{%- endmacro %}
{% macro static_row(name, value) -%}{{"{:<12} {:>17}".format(name, value)}}{%- endmacro %}
{% macro run_row(output, success, runtime) -%}{{"{:<18} {:>7} {:>12.4g}".format(output, success, runtime)}}{%- endmacro %}
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

Consistency Check
~~~~~~~~~~~~~~~~~

Consistency check summary::

    {{summary_row("name", "value")}}
    {{summary_row("overall", pass_text(check.sequence[0].test_pass))}}
    {{summary_row("q1", "{:>6.3f} / {:>6.3f}".format(check.sequence[0].q1, check.sequence[0].target_q1))}}
    {{summary_row("q2", "{:>6.3f} / {:>6.3f}".format(check.sequence[0].q2, check.sequence[0].target_q2))}}
{% if check.sequence[0].test_pass_time is defined %}
    {{summary_row("time q-scores", pass_text(check.sequence[0].test_pass_time))}}
{% endif %}
    {{summary_row("metric", check.sequence[0].metric)}}
{% if check.sequence[0].units is defined %}
    {{summary_row("units", check.sequence[0].units)}}
{% endif %}
    {{summary_row("eps0", "{:>10.3e}".format(check.sequence[0].eps0))}}
    {{summary_row("epsa", "{:>10.3e}".format(check.sequence[0].epsa))}}
    {{summary_row("epsr", "{:>10.3e}".format(check.sequence[0].epsr))}}
    {{summary_row("values", "{:>8.0f}".format(check.sequence[0].m))}}
    {{summary_row("relative failures", "{:>8.0f} ({:>5.3g}%)".format(check.sequence[0].wr, 100.0 * check.sequence[0].fr))}}
    {{summary_row("absolute failures", "{:>8.0f} ({:>5.3g}%)".format(check.sequence[0].wa, 100.0 * check.sequence[0].fa))}}

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

Time-quality locations::

    {{"{:<14} {:>8} {:>11} {:>7} {:>7} {:>5}".format("statistic", "days", "GWd/MTIHM", "q1", "q2", "limit")}}
{% if first_failed_time.row %}
    {{location_row("first fail", first_failed_time.row.time_days, compact(first_failed_time.row.burnup_gwd_per_mtihm) if first_failed_time.row.burnup_gwd_per_mtihm is defined else "", first_failed_time.row.q1, first_failed_time.row.q2, first_failed_time.row.limiting_score)}}
{% endif %}
    {{location_row("worst", worst_time.time_days, compact(worst_time.burnup_gwd_per_mtihm) if worst_time.burnup_gwd_per_mtihm is defined else "", worst_time.q1, worst_time.q2, worst_time.limiting_score)}}
{% endif %}

Q-score by high-order time::

    {{"{:>8} {:>11} {:>7} {:>7} {:>6}".format("days", "GWd/MTIHM", "q1", "q2", "pass")}}
{% for row in check.sequence[0].time_quality %}
    {{time_quality_row(row.time_days, compact(row.burnup_gwd_per_mtihm) if row.burnup_gwd_per_mtihm is defined else "", row.q1, row.q2, row.test_pass)}}
{% endfor %}
{% endif %}

{% if check.sequence[0].convergence_time_quality is defined %}
Convergence Q Scores
^^^^^^^^^^^^^^^^^^^^

For each convergence run, q1 and q2 are the minimum values over all high-order
time points.

..  image:: {{check.sequence[0].convergence_time_quality_image}}
    :width: 90%

Worst q-score by convergence run::

    {{"{:>5} {:>6} {:>7} {:>7} {:>10} {:>8} {:>10}".format("nlib", "nburn", "q1", "q2", "pass", "days", "GWd/MTU")}}
{% for row in check.sequence[0].convergence_time_quality %}
    {{convergence_row(row.nlib, row.nburn, row.q1, row.q2, row.pass, row.time_days, compact(row.burnup_gwd_per_mtu) if row.burnup_gwd_per_mtu is defined else "")}}
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

Static model parameters::

    {{static_row("name", "value")}}
{% for k,v in generate.static.items() %}
    {{static_row(k, compact(v) if v is number else v)}}
{% endfor %}

Run summary data::

    {{"{:<18} {:>7} {:>12}".format("output", "success", "runtime")}}
{% for row in run.runs %}
    {{run_row(row.output_file.split('/')[-1], pass_text(row.success), row.runtime_hrs if row.runtime_hrs is number else row.runtime_hrs|float)}}
{% endfor %}


.. raw:: pdf

      PageBreak oneColumn

Example Generated Input
~~~~~~~~~~~~~~~~~~~~~~~

This is the TRITON input (:code:`{{run.runs[0].input_file.split('/')[-1]}}`) for the first permutation out of {{run.runs|length}}.

.. include:: {{run.runs[0].input_file}}
    :literal:
