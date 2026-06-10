{% macro qscore(value) -%}{{"{:.3f}".format(value)}}{%- endmacro %}
{% macro compact(value) -%}{{"{:.4g}".format(value)}}{%- endmacro %}
{% macro burnup(value) -%}{{"{:.2e}".format(value)}}{%- endmacro %}
{% macro percent(value) -%}{{"{:.3g}".format(value)}}{%- endmacro %}
{% macro sci(value) -%}{{"{:.3e}".format(value)}}{%- endmacro %}
{% macro count(value) -%}{{"{:.0f}".format(value)}}{%- endmacro %}
{% macro qpair(row) -%}{{qscore(row.q_r)}}/{{qscore(row.q_ar)}}{%- endmacro %}
{% macro grid_text(name, values) -%}{% if name == "burnup" %}[{% for value in values %}{{burnup(value)}}{{", " if not loop.last}}{% endfor %}]{% else %}{{values}}{% endif %}{%- endmacro %}
{% macro pass_text(value) -%}{{"pass" if value else "fail"}}{%- endmacro %}
{% macro metric_text(value) -%}{{"g/gIHM" if value == "grams_per_initial_hm" else value}}{%- endmacro %}
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
        - {{grid_text(k, v.grid)-}}
    {% endfor %}

Consistency Check
~~~~~~~~~~~~~~~~~

For each compared inventory value :math:`i`, the absolute and relative
differences are:

.. class:: center

.. math::

    a_i = \left|lo_i - hi_i\right|

.. class:: center

.. math::

    r_i = \left|\frac{lo_i + \epsilon_0}{hi_i + \epsilon_0} - 1\right|

For :math:`m` compared values, define:

.. class:: center

.. math::

    w_r = \mathrm{count}(r_i > \epsilon_r)

.. class:: center

.. math::

    w_{ar} = \mathrm{count}(r_i > \epsilon_r \;\mathrm{and}\; a_i > \epsilon_a)

.. class:: center

.. math::

    f_r = \frac{w_r}{m}, \quad f_{ar} = \frac{w_{ar}}{m}

The quality scores are:

.. class:: center

.. math::

    q_r = 1 - f_r

.. class:: center

.. math::

    q_{ar} = 1 - 0.9 f_{ar} - 0.1 f_r

:math:`q_r` is the fraction of values that pass the relative-difference
tolerance. :math:`q_{ar}` gives most of its penalty to values that fail both the
relative and absolute tolerances, so small absolute differences do not dominate
the result. Higher scores are better.

The pass/fail result is determined by four subtests: :math:`q_r(0) = 1.000`,
:math:`q_{ar}(0) = 1.000`, :math:`q_r(t)` meeting its target at every high-order
time point, and :math:`q_{ar}(t)` meeting its target at every high-order time
point.

.. list-table:: Consistency check summary
    :widths: 30 18 36 16
    :header-rows: 1

    *   - test
        - result
        - score >= target
        - days
    *   - overall
        - .. class:: right

          {{pass_text(check.sequence[0].test_pass)}}
        -
          tests 1.1-2.2
        -
    {% if check.sequence[0].initial_time_quality is defined and check.sequence[0].initial_time_quality %}
    *   - 1.1 :math:`q_r(0)`
        - .. class:: right

          {{pass_text(check.sequence[0].test_pass_initial_q_r if check.sequence[0].test_pass_initial_q_r is defined else check.sequence[0].test_pass_initial)}}
        - .. class:: right

          {{qscore(check.sequence[0].initial_time_quality.q_r)}} = 1.000
        - .. class:: right

          {{compact(check.sequence[0].initial_time_quality.time_days)}}
    *   - 1.2 :math:`q_{ar}(0)`
        - .. class:: right

          {{pass_text(check.sequence[0].test_pass_initial_q_ar if check.sequence[0].test_pass_initial_q_ar is defined else check.sequence[0].test_pass_initial)}}
        - .. class:: right

          {{qscore(check.sequence[0].initial_time_quality.q_ar)}} = 1.000
        - .. class:: right

          {{compact(check.sequence[0].initial_time_quality.time_days)}}
    {% endif %}
    {% if check.sequence[0].minimum_q_r_time_quality is defined and check.sequence[0].minimum_q_r_time_quality %}
    *   - 2.1 :math:`\min_t q_r(t)`
        - .. class:: right

          {{pass_text(check.sequence[0].minimum_q_r_time_quality.test_pass_score)}}
        - .. class:: right

          {{qscore(check.sequence[0].minimum_q_r_time_quality.score)}} >= {{qscore(check.sequence[0].minimum_q_r_time_quality.target)}}
        - .. class:: right

          {{compact(check.sequence[0].minimum_q_r_time_quality.time_days)}}
    {% endif %}
    {% if check.sequence[0].minimum_q_ar_time_quality is defined and check.sequence[0].minimum_q_ar_time_quality %}
    *   - 2.2 :math:`\min_t q_{ar}(t)`
        - .. class:: right

          {{pass_text(check.sequence[0].minimum_q_ar_time_quality.test_pass_score)}}
        - .. class:: right

          {{qscore(check.sequence[0].minimum_q_ar_time_quality.score)}} >= {{qscore(check.sequence[0].minimum_q_ar_time_quality.target)}}
        - .. class:: right

          {{compact(check.sequence[0].minimum_q_ar_time_quality.time_days)}}
    {% endif %}

.. raw:: pdf

      FrameBreak 155

.. list-table:: Consistency check parameters
    :widths: 58 42
    :header-rows: 1

    *   - quantity
        - value
    *   - convergence search
        - .. class:: right

          {% if check.sequence[0].convergence_history is defined and check.sequence[0].convergence_history %}enabled{% if check.sequence[0].nlib is defined and check.sequence[0].nburn is defined %}; final nlib/nburn {{count(check.sequence[0].nlib)}}/{{count(check.sequence[0].nburn)}}{% endif %}{% else %}skip{% endif %}
    *   - metric
        - .. class:: right

          {{metric_text(check.sequence[0].metric)}}
    *   - :math:`\epsilon_0`
        - .. class:: right

          {{sci(check.sequence[0].eps0)}}
    *   - :math:`\epsilon_a`
        - .. class:: right

          {{sci(check.sequence[0].epsa)}}
    *   - :math:`\epsilon_r`
        - .. class:: right

          {{sci(check.sequence[0].epsr)}}

{% if check.sequence[0].m is defined %}
These aggregate count data are diagnostic only. They are accumulated over all
compared interpolation points, nuclides, and high-order time points; the
consistency check result is determined by Tests 1.1-2.2 above.

.. list-table:: Diagnostic aggregate Q-score counts
    :widths: 58 42
    :header-rows: 1

    *   - quantity
        - value
    {% if check.sequence[0].m is defined %}
    *   - :math:`m`
        - .. class:: right

          {{count(check.sequence[0].m)}}
    {% endif %}
    {% if check.sequence[0].w_r is defined %}
    *   - :math:`w_r`
        - .. class:: right

          {{count(check.sequence[0].w_r)}}
    *   - :math:`f_r`
        - .. class:: right

          {{percent(100.0 * check.sequence[0].f_r)}}%
    {% endif %}
    {% if check.sequence[0].w_ar is defined %}
    *   - :math:`w_{ar}`
        - .. class:: right

          {{count(check.sequence[0].w_ar)}}
    *   - :math:`f_{ar}`
        - .. class:: right

          {{percent(100.0 * check.sequence[0].f_ar)}}%
    {% endif %}
{% endif %}

{% if check.sequence[0].time_quality is defined %}
Consistency Tests
^^^^^^^^^^^^^^^^^

Four subtests determine the consistency-check result.

Test 1.1 requires :math:`q_r(0) = 1.000`; Test 1.2 requires
:math:`q_{ar}(0) = 1.000`. Test 2.1 requires :math:`q_r(t)` to meet its target
at every high-order time point; Test 2.2 requires :math:`q_{ar}(t)` to meet its
target at every high-order time point.

{% endif %}

{% if check.sequence[0].failure_reasons is defined and check.sequence[0].failure_reasons %}
Why This Check Failed
^^^^^^^^^^^^^^^^^^^^^

{% for reason in check.sequence[0].failure_reasons %}
* {{reason}}
{% endfor %}

{% endif %}

{% if check.sequence[0].hist_image is defined %}
Histogram of relative versus absolute errors for all nuclides at all states and times.

..  image:: {{check.sequence[0].hist_image}}
    :width: 90%
{% endif %}

{% if check.sequence[0].time_quality is defined %}
Quality Scores By Time
^^^^^^^^^^^^^^^^^^^^^^

The :math:`q_r` and :math:`q_{ar}` scores are recalculated independently at each high-order time point.
Burnup values are mean high-order history endpoint burnups in GWd/MTIHM.

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

.. list-table:: Time-quality locations
    :widths: 24 18 24 17 17
    :header-rows: 1

    *   - statistic
        - days
        - burnup
        - :math:`q_r`
        - :math:`q_{ar}`
    {% if first_failed_time.row %}
    *   - first {{first_failed_time.row.limiting_score}}
        - .. class:: right

          {{compact(first_failed_time.row.time_days)}}
        - .. class:: right

          {% if first_failed_time.row.burnup_gwd_per_mtihm is defined %}{{burnup(first_failed_time.row.burnup_gwd_per_mtihm)}}{% endif %}
        - .. class:: right

          {{qscore(first_failed_time.row.q_r)}}
        - .. class:: right

          {{qscore(first_failed_time.row.q_ar)}}
    {% endif %}
    *   - worst {{worst_time.limiting_score}}
        - .. class:: right

          {{compact(worst_time.time_days)}}
        - .. class:: right

          {% if worst_time.burnup_gwd_per_mtihm is defined %}{{burnup(worst_time.burnup_gwd_per_mtihm)}}{% endif %}
        - .. class:: right

          {{qscore(worst_time.q_r)}}
        - .. class:: right

          {{qscore(worst_time.q_ar)}}
{% endif %}

.. list-table:: Q-score by high-order time
    :widths: 16 23 14 14 13
    :header-rows: 1

    *   - days
        - burnup
        - :math:`q_r`
        - :math:`q_{ar}`
        - pass
    {% for row in check.sequence[0].time_quality %}
    *   - .. class:: right

          {{compact(row.time_days)}}
        - .. class:: right

          {% if row.burnup_gwd_per_mtihm is defined %}{{burnup(row.burnup_gwd_per_mtihm)}}{% endif %}
        - .. class:: right

          {{qscore(row.q_r)}}
        - .. class:: right

          {{qscore(row.q_ar)}}
        - {{pass_text(row.test_pass)}}
    {% endfor %}
{% endif %}

Convergence
^^^^^^^^^^^

Convergence is a search used to choose nlib/nburn for Test 2; it is not a
separate pass/fail test. The search stops when successive minimum time-point
:math:`q_r/q_{ar}` values change by no more than the stop criteria, or when
:math:`q_r/q_{ar}` increase and cross above the pass targets.

{% if check.sequence[0].convergence_history is defined and check.sequence[0].convergence_history %}
{% if check.sequence[0].convergence_quality_image is defined %}
..  image:: {{check.sequence[0].convergence_quality_image}}
    :width: 90%
{% endif %}

{% if check.sequence[0].convergence_status is defined %}
.. list-table:: Convergence search status
    :widths: 18 82
    :header-rows: 1

    *   - param
        - detail
    {% for name, status in check.sequence[0].convergence_status.items() %}
    *   - {{name}}
        - {{status.result}}; value/max {{count(status.value)}}/{{count(status.max)}}{% if status.delta_q_r_text or status.delta_q_ar_text %};
          min-time :math:`\Delta q_r` {{status.delta_q_r_text or "n/a"}};
          min-time :math:`\Delta q_{ar}` {{status.delta_q_ar_text or "n/a"}}{% endif %}{% if status.reason %};
          {{status.reason}}{% endif %}
    {% endfor %}
{% endif %}

.. list-table:: Convergence summary
    :widths: 25 30 25 20
    :header-rows: 1

    *   - nlib/nburn
        - min-time :math:`q_r/q_{ar}`
        - result
        - days
    {% for row in check.sequence[0].convergence_history %}
    *   - .. class:: convergence-table-body-right

          {{count(row.nlib)}}/{{count(row.nburn)}}
        - .. class:: convergence-table-body-right

          {{qpair(row)}}
        - .. class:: convergence-table-body

          {{row.result}}
        - .. class:: convergence-table-body-right

          {% if row.time_days is defined %}{{compact(row.time_days)}}{% endif %}
    {% endfor %}
{% elif check.sequence[0].convergence_status is defined %}
.. list-table:: Convergence search status
    :widths: 18 82
    :header-rows: 1

    *   - param
        - detail
    {% for name, status in check.sequence[0].convergence_status.items() %}
    *   - {{name}}
        - {{status.result}}; value/max {{count(status.value)}}/{{count(status.max)}}{% if status.delta_q_r_text or status.delta_q_ar_text %};
          min-time :math:`\Delta q_r` {{status.delta_q_r_text or "n/a"}};
          min-time :math:`\Delta q_{ar}` {{status.delta_q_ar_text or "n/a"}}{% endif %}{% if status.reason %};
          {{status.reason}}{% endif %}
    {% endfor %}
{% else %}
.. list-table:: Convergence search status
    :widths: 25 75
    :header-rows: 1

    *   - status
        - detail
    *   - skip
        - convergence search was not enabled; Test 2 uses the configured single nlib/nburn run
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
        - {% if v is number %}.. class:: right

          {{compact(v)}}{% else %}{{v-}}{% endif %}
    {% endfor %}

.. list-table:: Run summary data
    :widths: 58 14 28
    :header-rows: 1

    *   - output
        - pass
        - runtime (hrs)
    {%- for row in run.runs %}
    *   - :code:`{{row.output_file.split('/')[-1]}}`
        - {{pass_text(row.success)}}
        - .. class:: right

          {% if row.runtime_hrs is number %}{{compact(row.runtime_hrs)}}{% else %}{{row.runtime_hrs}}{% endif %}
    {%- endfor %}


.. raw:: pdf

      PageBreak oneColumn

Appendix: Example Generated Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the TRITON input (:code:`{{run.runs[0].input_file.split('/')[-1]}}`) for the first permutation out of {{run.runs|length}}.

.. class:: appendix-code

.. include:: {{run.runs[0].input_file}}
    :literal:
