"""
The scale.olm.core module contains classes that have a core capability that could be
used by any component of OLM and even on their own outside of OLM. Maybe one day
they will become their own package as scale-core instead of scale-olm-core. Here are
some principles for the core.

- The core should only contain classes.
- Each class should have doctests demonstrating the happy path, i.e. not errors
  or edge cases.
- The unit tests for a core class are at `testing/core_ClassName.test.py` file and should
  address edge cases and other behavior.
- Classes can be further demonstrated in notebooks, `notebooks/core_ClassName_demo.ipynb`.
"""
import json
from pathlib import Path
import pathlib
import os
import sys
import copy
import subprocess
import shutil
import re
import h5py
import numpy as np
from tqdm import tqdm, tqdm_notebook
import time
import matplotlib.pyplot as plt

# TODO: remove this dependency ASAP.
from scale.olm.internal import run_command


class TemplateManager:
    """Manage jinja templates.

    Finds all jinja templates in provided paths, including the local "templates"
    path inside OLM and the OLM_TEMPLATES_PATH environment variable.

    Paths are searched in order starting with those provided to init, then
    the OLM_TEMPLATES_PATH then the local.

    Templates MUST have a double extension like my.jt.inp with *.jt.* to be recognized.

    Args:
        paths list[str]: additional user paths to search

    Attributes:
        templates dict[str,pathlib.Path]: dict of all available template names, paths
        paths list[str]: list of paths that were searched (including local and environment)

    Examples:

    Create a temporary directory and write two template files to it.

    >>> td = TempDir()
    >>> path1 = td.write_file("Hello {{noun}}.","line1.jt.inp")
    >>> subdir = td.path / 'y' / 'z'
    >>> subdir.mkdir(parents=True)
    >>> path2 = td.write_file("Is there {{noun}} out there?","y/z/line2.jt.inp")

    Create a template manager which will find those paths.

    >>> tm = TemplateManager(paths=[td.path], include_env=False)
    >>> tm.names()
    ['line1.jt.inp', 'y/z/line2.jt.inp']

    >>> tm.expand('line1.jt.inp',{"noun":"hello"})
    'Hello hello.'

    >>> tm.expand('y/z/line2.jt.inp',{"noun":"anybody"})
    'Is there anybody out there?'

    """

    def __init__(self, paths=[], include_env=True):
        import copy
        import glob
        import os

        # Initialize paths variable.
        self.paths = []
        for p in paths:
            self.paths.append(Path(p).resolve())

        # Include environment paths.
        if include_env:
            if "OLM_TEMPLATES_PATH" in os.environ:
                for p in os.environ["OLM_TEMPLATES_PATH"].split(os.pathsep):
                    self.paths.append(Path(p).resolve())
            local = Path(__file__).parent / "templates"
            self.paths.append(local.resolve())

        # Search for template.
        self.templates = {}
        for p in self.paths:
            for v in glob.glob(str(Path(p) / "**/*.jt.*"), recursive=True):
                k = str(Path(v).relative_to(p))
                if not k in self.templates:
                    self.templates[k] = v

    def names(self):
        """Return the names of all the templates.

        Returns:
            list[str]: names of templates

        """
        return list(self.templates.keys())

    def path(self, name: str):
        """Return a template file by name.

        Args:
            name: template name

        Returns:
            str: full path to the template
        """
        return self.templates[name]

    def expand(self, name: str, data: dict):
        """Expand a template by name using provided data.

        Args:
            name: template name
            data: dictionary with all the data

        Returns:
            str: text from the expanded template and data
        """
        return TemplateManager.expand_file(self.templates[name], data)

    @staticmethod
    def _jinja2_render_traceback(src_path):
        from sys import exc_info

        tb_frame_re = re.compile(
            r"<frame at 0x[a-z0-9]*, file '(.*)', line (\d+), (?:code top-level template code|code template)>"
        )
        traceback_print = ""

        # Get traceback objects
        typ, value, tb = exc_info()
        # Iterate over nested traceback frames
        while tb:
            # Parse traceback frame string
            tb_frame_str = str(tb.tb_frame)
            tb_frame_match = tb_frame_re.match(tb_frame_str)
            tb_frame_istemplate = False
            # Identify frames corresponding to Jinja2 templates
            if tb.tb_frame.f_code.co_filename == "<template>":
                # Top-most template
                tb_src_path = src_path
                tb_lineno = tb.tb_lineno
                tb_frame_istemplate = True
            elif tb_frame_match:
                # nested child templates
                tb_src_path = tb_frame_match.group(1)
                tb_lineno = tb_frame_match.group(2)
                tb_frame_istemplate = True
            # Factorized string formatting
            if tb_frame_istemplate:
                traceback_print += f"    Template '{tb_src_path}', line {tb_lineno}\n"
                # Fetch the line raising the exception
                with open(tb_src_path, "r") as tb_src_file:
                    for lineno, line in enumerate(tb_src_file):
                        if lineno == int(tb_lineno) - 1:
                            traceback_print += "        " + line.strip() + "\n"
                            break
            tb = tb.tb_next

        # Strip the final line jump
        return traceback_print[:-1]

    @staticmethod
    def _tree_print(data, path=""):
        if isinstance(data, dict):
            new = ""
            if path == "":
                root = ""
            else:
                root = path + "."
            for k, v in data.items():
                new += TemplateManager._tree_print(v, path=root + k)
            return new
        elif isinstance(data, list):
            new = ""
            for i in range(len(data)):
                new += TemplateManager._tree_print(data[i], path=path + f"[{i}]")
            return new
        else:
            return f"{path}={data}\n"

    @staticmethod
    def expand_text(text: str, data: dict, src_path: str = ""):
        """Returns the expanded text with data.

        Use jinja to expand the text with data.

        Args:
            text: text containing jinja directives
            data: dictionary containing data

        Raises:
            ValueError: if jinja raises an undefined variable error

        Returns:
            str: expanded text
        """
        from jinja2 import Template, StrictUndefined, exceptions, TemplateError

        j2t = Template(text, undefined=StrictUndefined)

        # Catch specific types of error.
        try:
            return j2t.render(data)
        except exceptions.TemplateError as ve:
            trace = TemplateManager._jinja2_render_traceback(src_path=src_path)
            raise ValueError(
                "Available data: \n"
                + TemplateManager._tree_print(data)
                + "\n"
                + "Template error: "
                + str(ve)
                + "\n"
                + trace
                + "\n"
            )

    @staticmethod
    def expand_file(path: pathlib.Path, data: dict):
        """Returns expanded text from a file.

        Args:
            path: path containing a file to read
            data: dictionary containing data

        Returns:
            str: expanded text
        """
        with open(path, "r") as f:
            text = f.read()
            return TemplateManager.expand_text(text, data, src_path=str(path))


class CompositionManager:
    """Stores the basic nuclide data and provides calculations.

    Args:
        data: The basic data in ii.json-style format.

    Attributes:
        data (dict): nuclide data

    Examples:

    Initialize with fake data.

    >>> data = {
    ...     "0001001": {
    ...         "IZZZAAA": "0001001",
    ...         "atomicNumber": 1,
    ...         "element": "H",
    ...         "isomericState": 0,
    ...         "mass": 1.007830023765564,
    ...         "massNumber": 1
    ...     },
    ...     "0001002": {
    ...         "IZZZAAA": "0001002",
    ...         "atomicNumber": 1,
    ...         "element": "H",
    ...         "isomericState": 0,
    ...         "mass": 2.0141000747680664,
    ...         "massNumber": 2
    ...     }
    ... }
    >>> cm = CompositionManager(data)

    Output the mass.

    >>> cm.mass("h2")
    2.0141000747680664

    Do name conversions.

    >>> cm.eam("0001002")
    'h2'
    >>> cm.eam("h2")
    'h2'
    >>> cm.izzzaaa("h2")
    '0001002'

    """

    def __init__(self, data):
        self.data = data
        self.e_to_z = {}
        self.z_to_e = {}
        for izzzaaa, d in self.data.items():
            z = int(d["atomicNumber"])
            e = d["element"].lower()
            self.e_to_z[e] = z
            self.z_to_e[z] = e

    @staticmethod
    def parse_eam_to_eai(id: str):
        """Parse an eam nuclide identifier into an e,a,i tuple."""
        regexp = r"^([a-z][a-z]?)(\d+)(m?\d?)$"
        match = re.search(regexp, id.lower())
        if not match:
            raise ValueError(
                "nuclide id={id} did not match regular expression for eam '{regexp}'"
            )
        e = match.group(1)
        a = int(match.group(2))
        i = 0
        if match.group(3):
            mstr = match.group(3)
            if mstr == "m":
                i = 1
            else:
                i = int(mstr.replace("m", ""))

        return e, a, i

    @staticmethod
    def parse_izzzaaa(id: str):
        """Parse an izzzaaa nuclide identifier into an i,z,a triplet."""
        regexp = r"(\d)(\d\d\d)(\d\d\d)"
        match = re.search(regexp, id)
        if not match:
            raise ValueError(
                "nuclide id={id} did not match regular expression for izzzaaa'{regexp}'"
            )
        i = int(match.group(1))
        z = int(match.group(2))
        a = int(match.group(3))
        return i, z, a

    @staticmethod
    def form_izzzaaa(i: int, z: int, a: int) -> str:
        """Form an izzzaaa identifier from i,z,a triplet."""
        return "{:01d}{:03d}{:03d}".format(i, z, a)

    @staticmethod
    def form_eam_from_eai(e: str, a: int, i: int) -> str:
        """Form an eam identifier from e,a,i triplet."""
        if i == 1:
            m = "m"
        elif i > 1:
            m = "m" + str(i)
        else:
            m = ""
        return e + str(a) + m

    def izzzaaa(self, id: str) -> str:
        """Return an izzzaaa like '1095242' from either an eam or izzzaaa identifier.

        Args:
            id: The identifier either an eam like 'am242m' or izzzaaa.

        Returns:
            str: An izzzaaa nuclide identifier.
        """
        import re

        # Quick return if already an izzzaaa.
        if re.search(r"^\d+", id):
            return id

        e, a, i = self.parse_eam_to_eai(id)
        z = self.e_to_z[e]
        return self.form_izzzaaa(i, z, a)

    def eam(self, id: str) -> str:
        """Return an eam like 'am242m' from either an eam or izzzaaa identifier.

        Args:
            id: The identifier either an eam or izzzaaa like '1095242'.

        Returns:
            str: An eam nuclide identifier.
        """
        import re

        # Quick return if already an eam.
        if not re.search(r"^\d+", id):
            return id

        i, z, a = self.parse_izzzaaa(id)
        e = self.z_to_e[z]
        return self.form_eam_from_eai(e, a, i)

    def mass(self, id: str, default: float = None) -> float:
        """Return the mass for a specific nuclide."""
        backup = {"mass": default}
        return self.data.get(self.izzzaaa(id), backup)["mass"]

    @staticmethod
    def renormalize_wtpt(wtpt0, sum0, key_filter=""):
        """Renormalize to sum0 any keys matching filter."""
        # Calculate the sum of filtered elements. Copy into return value.
        wtpt = {}
        sum = 0.0
        for k, v in wtpt0.items():
            if k.startswith(key_filter):
                sum += v
                wtpt[k] = v

        # Renormalize.
        norm = sum / sum0 if sum0 > 0.0 else 0.0
        if norm>0.0:
          for k in wtpt:
              wtpt[k] /= norm
        return wtpt, norm

    @staticmethod
    def grams_per_mol(
        iso_wts: dict[str, float], m_data: dict[str, float] = {"am241": 241.0568}
    ) -> float:
        """Calculate the grams per mole of a mass fraction mixture.

        Use the formula to calculate the total mixture molar mass :math:`m` according to

        .. math::
                1/m = \\sum_i w_i / m_i

                \\sum_i w_i = 1.0

        for each nuclide :math:`i`.

        The values for the individual molar masses are :math:`m_i` are provided in
        the m_data dict. This is not very important for the purposes of this code
        that these values be precise. If not present in the dict, the simple
        mass number is used, i.e. 242 for am242m.

        Args:
            iso_wts: Dictionary with keys as nuclide names (e.g. 'am241') and values
                     proportional to mass fraction.
            m_data: Optional molar masses (grams/mol).

        Returns:
            The total molar mass m from the above formula.

        Examples:
            Force the use of mass numbers by providing empty mass data

            >>> CompositionManager.grams_per_mol({'u235': 50, 'pu239': 50}, m_data={})
            236.9831223628692
        """
        import re

        # Renormalize to 1.0.
        iso_wts0, norm0 = CompositionManager.renormalize_wtpt(iso_wts, 1.0)
        m = 0.0
        for iso, wt in iso_wts0.items():
            m_default = re.sub("^[a-z]+", "", iso)
            m_default = re.sub("m[0-9]*$", "", m_default)
            m_iso = m_data.get(iso, float(m_default))
            m += wt / m_iso if m_iso > 0.0 else 0.0

        return 1.0 / m if m > 0.0 else 0.0

    @staticmethod
    def calculate_hm_oxide_breakdown(x):
        """Calculate the oxide breakdown from weight percentages in x for all heavy metal."""
        hm_iso, hm_norm = CompositionManager.renormalize_wtpt(x, 100.0)

        pu_iso, pu_norm = CompositionManager.renormalize_wtpt(hm_iso, 100.0, "pu")
        am_iso, am_norm = CompositionManager.renormalize_wtpt(hm_iso, 100.0, "am")
        u_iso, u_norm = CompositionManager.renormalize_wtpt(hm_iso, 100.0, "u")

        return {
            "uo2": {"iso": u_iso, "dens_frac": u_norm},
            "puo2": {"iso": pu_iso, "dens_frac": pu_norm},
            "amo2": {"iso": am_iso, "dens_frac": am_norm},
            "hmo2": {"iso": hm_iso, "dens_frac": 1.0},
        }

    @staticmethod
    def approximate_hm_info(comp):
        """Approximate some heavy metal information."""

        # Calculate molar masses.
        m_pu = CompositionManager.grams_per_mol(comp["puo2"]["iso"])
        m_am = CompositionManager.grams_per_mol(comp["amo2"]["iso"])
        m_u = CompositionManager.grams_per_mol(comp["uo2"]["iso"])
        m_hm = CompositionManager.grams_per_mol(comp["hmo2"]["iso"])
        m_o2 = 2 * 15.9994

        # Calculate heavy metal fractions of oxide (approximate).
        puo2_hm_frac = m_pu / (m_pu + m_o2)
        amo2_hm_frac = m_am / (m_am + m_o2)
        uo2_hm_frac = m_u / (m_u + m_o2)
        hmo2_hm_frac = m_hm / (m_hm + m_o2)

        # Calculate heavy metal densities.
        am_dens = amo2_hm_frac * comp["amo2"]["dens_frac"]
        pu_dens = puo2_hm_frac * comp["puo2"]["dens_frac"]
        u_dens = uo2_hm_frac * comp["uo2"]["dens_frac"]

        # Back out some useful quantities. Despite being called
        # fractions, these are in percent, as per convention.
        pu239 = comp["hmo2"]["iso"].get("pu239", 0.0)
        pu241 = comp["hmo2"]["iso"].get("pu241", 0.0)
        am241 = comp["hmo2"]["iso"].get("am241", 0.0)
        dummy, am_frac = CompositionManager.renormalize_wtpt(
            comp["hmo2"]["iso"], 1.0, key_filter="am"
        )
        dummy, pu_frac = CompositionManager.renormalize_wtpt(
            comp["hmo2"]["iso"], 1.0, key_filter="pu"
        )
        pu_frac += am241
        norm = (pu_frac + am_frac)
        if norm > 0.0:
            pu239_frac = 100 * pu239 / norm
            am241_frac = 100 * am241 / norm
            fiss_pu_frac = 100 * (pu239 + pu241) / norm
        else:
            pu239_frac = 0
            am241_frac = 0
            fiss_pu_frac = 0


        return {
            "am241_frac": am241_frac,
            "pu239_frac": pu239_frac,
            "pu_frac": pu_frac,
            "fiss_pu_frac": fiss_pu_frac,
            "m_o2": m_o2,
            "m_u": m_u,
            "m_am": m_am,
            "m_pu": m_pu,
            "m_hm": m_hm,
            "puo2_hm_frac": puo2_hm_frac,
            "amo2_hm_frac": amo2_hm_frac,
            "uo2_hm_frac": uo2_hm_frac,
            "hmo2_hm_frac": hmo2_hm_frac,
        }


class BurnupHistory:
    """Manages a time versus burnup power history.

    Args:
        time:  List of cumulative times, monotonically strictly increasing, each time
               must be greater than the last.
        burnup: list of cumulative burnups, monotonically increasing, each burup must
                be greater or equal to the last.
        epsilon_dbu: The small value of burnup changes to disregard (default: 0), can be useful for
                     avoiding tiny numerical precision issues when the input burnups
                     are subtracted.


    """

    @staticmethod
    def _reconstruct(t0, b0, intervals):
        """Helper to reconstruct some attribute variables."""
        import numpy as np

        time = [t0]
        burnup = [b0]
        interval_power = []
        interval_burnup = []
        interval_time = []
        for tb in intervals:
            time.append(time[-1] + tb[0])
            burnup.append(burnup[-1] + tb[1])
            interval_time.append(tb[0])
            interval_burnup.append(tb[1])
            interval_power.append(tb[2])
        return time, burnup, interval_time, interval_burnup, interval_power

    def __init__(self, time, burnup, epsilon_dbu: float = 0.0):
        self.intervals = []
        self.epsilon_dbu = epsilon_dbu
        skipped_dbu = 0
        for j in range(1, len(time)):
            dt = time[j] - time[j - 1]
            dbu = burnup[j] - burnup[j - 1] + skipped_dbu
            if dbu <= epsilon_dbu:
                dbu = 0

            if dt > 0:
                self.intervals.append([dt, dbu, dbu / dt])
                skipped_dbu = 0
            else:
                skipped_dbu += dbu

        (
            self.time,
            self.burnup,
            self.interval_time,
            self.interval_burnup,
            self.interval_power,
        ) = self._reconstruct(time[0], burnup[0], self.intervals)

    @staticmethod
    def union_times(a, b, kind="mergesort"):
        """Union two time grids and sort properly."""
        c = np.concatenate((a, b))
        c.sort(kind=kind)
        flag = np.ones(len(c), dtype=bool)
        np.not_equal(c[1:], c[:-1], out=flag[1:])
        return c[flag]

    def _get_cycle_intervals(
        self, min_shutdown_time, min_shutdown_power, initial_shutdown
    ):
        """Get the cycle intervals."""

        cycle_intervals = []
        eoc_cooling = 0.0
        shutdown = initial_shutdown
        for j in range(len(self.interval_power)):

            # We are at power now.
            if self.interval_power[j] > min_shutdown_power:

                # But we were shutdown, so this is new cycle.
                if shutdown:
                    shutdown = False
                    cycle_intervals.append([j, None])
                    eoc_cooling = 0.0
                # Update the end of cycle.
                cycle_intervals[-1][1] = j + 1
            # Otherwise, we are cooling but not yet sure if shutdown.
            else:
                # We exceeded the cooling limit so this is now shutdown.
                eoc_cooling += self.interval_time[j]
                if eoc_cooling > min_shutdown_time:
                    shutdown = True

        return cycle_intervals

    def classify_operations(
        self,
        min_shutdown_time: float = 0.0,
        min_shutdown_power: float = 0.0,
        starts_within_cycle: int = None,
    ):
        """Classify the operating history into cycles, downtime, etc.

        The output of this function is a dictionary of information that can be used
        to classify the operating history into cycles. By default we consider
        that any time with zero power is a shutdown and thus starts a new cycle.

        Here is an example of the dictionary returned.

        .. code::

            time =   [0, 5,  10,  50,  55,  100,  105]
            burnup = [0, 0, 100, 500, 500, 1000, 1000]
            bh = BurnupHistory(time, burnup)
            x = bh.classify_operations()

        .. code:: json

            {
                "options": {
                    "min_shutdown_time": 0.0,
                    "min_shutdown_power": 0.0,
                    "starts_within_cycle": null
                },
                "operations": [
                    {
                        "cycle": "",
                        "within_cycle": false,
                        "start": 0,
                        "end": 1
                    },
                    {
                        "cycle": "1",
                        "within_cycle": true,
                        "start": 1,
                        "end": 3
                    },
                    {
                        "cycle": "",
                        "within_cycle": false,
                        "start": 3,
                        "end": 4
                    },
                    {
                        "cycle": "2",
                        "within_cycle": true,
                        "start": 4,
                        "end": 5
                    },
                    {
                        "cycle": "",
                        "within_cycle": false,
                        "start": 5,
                        "end": 6
                    }
                ]
            }

        Args:
            min_shutdown_time: Minimum time to consider as a shutdown.
            min_shutdown_power: Minimum power to consider as a shutdown.
            starts_within_cycle: Instead of starting with cycle 1, start with this.

        Returns:
            list[dict]: A list of dictionaries described above.

        Examples:

            Create a simple operating history. Our numbers can be interpreted
            to be in days and MWd/MTU for time and burnup respectively.

            >>> time =   [0, 5,  10,  50,  55,  100,  105]
            >>> burnup = [0, 0, 100, 500, 500, 1000, 1000]
            >>> bh = BurnupHistory(time, burnup)

            The time for each interval is calculated.

            >>> bh.interval_time
            [5, 5, 40, 5, 45, 5]

            The power for each interval is calculated as burnup/time.

            >>> bh.interval_power
            [0.0, 20.0, 10.0, 0.0, 11.11111111111111, 0.0]

            We will classify operations on this operating history to demonstrate that the
            defaults will result in the short 5 days of zero power being considered part
            of the cycle.

            >>> x = bh.classify_operations()

            Let's create a function to print the information.

            >>> def print_classification(x):
            ...     print(x['options'])
            ...     for op in x['operations']:
            ...         msg = 'during cycle ' + op['cycle'] if op['within_cycle'] else 'shutdown'
            ...         for i in range(op['start'],op['end']):
            ...             print("interval {} is {} with power {:.4g} MW/MTU for {:.4g} days".format(
            ...                 i,
            ...                 msg,
            ...                 bh.interval_power[i],
            ...                 bh.interval_time[i])
            ...             )
            >>> print_classification(x)
            {'min_shutdown_time': 0.0, 'min_shutdown_power': 0.0, 'starts_within_cycle': None}
            interval 0 is shutdown with power 0 MW/MTU for 5 days
            interval 1 is during cycle 1 with power 20 MW/MTU for 5 days
            interval 2 is during cycle 1 with power 10 MW/MTU for 40 days
            interval 3 is shutdown with power 0 MW/MTU for 5 days
            interval 4 is during cycle 2 with power 11.11 MW/MTU for 45 days
            interval 5 is shutdown with power 0 MW/MTU for 5 days

            Now let's classify again but with a minimum shutdown time of 10 days so that
            the intra-cycle power dip does not appear as a reload and we just have one cycle.

            >>> x = bh.classify_operations(min_shutdown_time=10.0)
            >>> print_classification(x)
            {'min_shutdown_time': 10.0, 'min_shutdown_power': 0.0, 'starts_within_cycle': None}
            interval 0 is shutdown with power 0 MW/MTU for 5 days
            interval 1 is during cycle 1 with power 20 MW/MTU for 5 days
            interval 2 is during cycle 1 with power 10 MW/MTU for 40 days
            interval 3 is during cycle 1 with power 0 MW/MTU for 5 days
            interval 4 is during cycle 1 with power 11.11 MW/MTU for 45 days
            interval 5 is shutdown with power 0 MW/MTU for 5 days

        """

        options = {
            "min_shutdown_time": min_shutdown_time,
            "min_shutdown_power": min_shutdown_power,
            "starts_within_cycle": starts_within_cycle,
        }

        cycle_intervals = self._get_cycle_intervals(
            min_shutdown_time=min_shutdown_time,
            min_shutdown_power=min_shutdown_power,
            initial_shutdown=starts_within_cycle == None,
        )

        # No cycles still means there could be decay.
        if len(cycle_intervals) == 0:
            return {
                "options": options,
                "operations": [
                    {
                        "cycle": "",
                        "within_cycle": False,
                        "start": 0,
                        "end": len(self.interval_time),
                    }
                ],
            }

        # Get the cycle offset.
        if starts_within_cycle == None:
            cycle_offset = 1
        else:
            cycle_offset = starts_within_cycle

        # Initialize with pre-operations decay, if any.
        operations = []
        if cycle_intervals[0][0] > 0:
            operations.append(
                {
                    "cycle": "",
                    "within_cycle": False,
                    "start": 0,
                    "end": cycle_intervals[0][0],
                }
            )

        for cycle in range(len(cycle_intervals)):
            # Add the at-power operations.
            operations.append(
                {
                    "cycle": str(cycle + cycle_offset),
                    "within_cycle": True,
                    "start": cycle_intervals[cycle][0],
                    "end": cycle_intervals[cycle][1],
                }
            )
            # Get the next start interval, defaulting to end.
            next_start = len(self.interval_time)
            if cycle + 1 < len(cycle_intervals):
                next_start = cycle_intervals[cycle + 1][0]

            # Add the shutdown cooling if it exists.
            if next_start > cycle_intervals[cycle][1]:
                operations.append(
                    {
                        "cycle": "",
                        "within_cycle": False,
                        "start": cycle_intervals[cycle][1],
                        "end": next_start,
                    }
                )

        return {"options": options, "operations": operations}

    def plot_power_history(self, label=None, add_to_existing=False, **kwargs):
        """Return a plot of the power in each interval."""
        import matplotlib.pyplot as plt

        if not (add_to_existing and plt.gca().has_data()):
            plt.figure()

        plt.step(
            self.time,
            [self.interval_power[0], *self.interval_power],
            where="pre",
            label=label,
            **kwargs,
        )
        plt.legend()

    @staticmethod
    def _testing_data_sfcompo1():
        """This is special testing data from SFCOMPO that has some peculiarities which
        is useful for testing we can handle real-world data."""
        # fmt: off
        time0 = [
            0, 351, 714, 715, 716.5, 718.5, 719.5, 720.5, 830.5, 831,
            907, 907.5, 930.5, 930.7, 934.2, 934.7, 1004.2, 1021.8,
            1021.8, 1073, 1073.5, 1075.5, 1077, 1078, 1079, 1081.5,
            1082, 1094, 1094.5, 1146, 1147, 1285, 1285.2, 1368.2,
            1369.2, 1435.2, 1435.2, 1507, 1507.5, 1509.5, 1511.5,
            1513.5, 1548.5, 1586.5, 1587.5, 1716.7, 1716.7, 1817, 1825,
            1893.9, 1896.4, 1900.4, 1905.4, 1907.4, 1967.3, 2027.2,
            2087.1, 2137, 2157, 2157, 2205, 2211, 2213, 2214, 2300.7,
            2387.3, 2474, 2475, 2519.8, 2520.9, 2537.7, 2539.3, 2539.3,
            2569.3, 2577.1, 2637.7, 2638.7, 2696.3, 2753.9, 2754.9,
            2808.6, 2809.6, 2875, 2876, 2896.5, 2897.5, 2897.5, 2903.3,
        ]
        burnup0 = [
            0, 0, 0, 7, 30, 77, 110, 126, 3685, 3697, 6143, 6150, 6869,
            6872, 6981, 6993, 9184, 9719, 9719, 9719, 9722, 9748, 9778,
            9804, 9831, 9899, 9909, 10239, 10249, 11665, 11685, 15448,
            15452, 17699, 17712, 19515, 19515, 19515, 19519, 19547,
            19593, 19650, 20756, 21945, 21960, 25834, 25834, 25834,
            25988, 28747, 28798, 28956, 29157, 29196, 31537, 33886,
            36179, 38051, 38658, 38658, 38658, 38747, 38794, 38819,
            41793, 44779, 47713, 47728, 49175, 49211, 49740, 49762,
            49762, 49762, 49861, 51491, 51504, 53110, 54796, 54817,
            56367, 56380, 58257, 58286, 58842, 58854, 58854, 58854,
        ]
        # fmt: on
        return time0, burnup0

    def get_cycle_time(self, x):
        """Return a new time array for each cycle from the output of classify_operations.

        One way this class is useful is to pass into the regrid function,
        which takes a list of (cumulative) times.

        .. plot::
            :include-source: True
            :show-source-link: False
            :caption: Using get_cycle_time to determine cycle-average powers.

            import matplotlib.pyplot as plt
            from scale.olm.core import BurnupHistory

            time0,burnup0 = BurnupHistory._testing_data_sfcompo1()
            bh = BurnupHistory(time0, burnup0)

            bh.plot_power_history(label='original')

            x = bh.classify_operations(min_shutdown_time=10.0)
            new_time = bh.get_cycle_time(x)

            bh2 = bh.regrid(new_time)
            bh2.plot_power_history(label='regrid', add_to_existing=True)

            plt.legend()
            plt.show()

        """
        cycle_id = None
        new_time = []
        dt = 0.0
        for op in x["operations"]:
            if op["cycle"] != cycle_id:
                new_time.append(dt)
            for i in range(op["start"], op["end"]):
                dt += self.interval_time[i]
        if dt > new_time[-1]:
            new_time.append(dt)
        return new_time

    def regrid(self, time, burnup_interp=None):
        """Project to a new time grid."""
        import numpy as np

        if burnup_interp:
            burnup = burnup_interp(time)
        else:
            # Linear interp
            burnup = np.interp(time, self.time, self.burnup)

        dbu = np.diff(burnup)
        if np.any(dbu < 0):
            ValueError(
                f"applying passed in burnup interpolator resulted in non-monotonic burnup: {burnup}"
            )

        return BurnupHistory(time, burnup)


class FileHasher:
    """Hashes the content of a file.

    Args:
        file: Path to an existing file.

    Attributes:
        id (str): Hash of the contents of the file.

    Examples:

        >>> from scale.olm.core import TempDir
        >>> td = TempDir()
        >>> a_path = td.write_file("some duplicate content","a.txt")
        >>> b_path = td.write_file("some duplicate content","b.txt")
        >>> FileHasher(a_path).id
        '161da18e656052f506f6283f71168d6a'
        >>> FileHasher(b_path).id
        '161da18e656052f506f6283f71168d6a'
    """

    def __init__(self, file: pathlib.Path):
        from imohash import hashfile

        self.id = str(hashfile(file, hexdigest=True))


class TempDir:
    """Creates a temporary directory that self-deletes.

    Deletes the directory when the class goes out of scope.

    Examples:
        Create a temporary directory.

        >>> from scale.olm.core import TempDir
        >>> td = TempDir()
        >>> path = td.path
        >>> path.exists() and path.is_dir()
        True

        When the object goes out of scope, the directory is deleted.

        >>> td = None
        >>> path.exists()
        False

    Attributes:
        path: path to the temporary directory

    """

    def __init__(self):
        import tempfile

        self._td_obj = tempfile.TemporaryDirectory()
        self.path = Path(self._td_obj.name)

    def write_file(self, text: str, name: str) -> pathlib.Path:
        """Write text to a file name in the temporary directory.

        Args:
            text: Text to write.
            name: Name of the file to write to in the temp directory.

        Returns:
            Filename that was created.

        Examples:
            Create a file.

            >>> td = TempDir()
            >>> file = td.write_file("CONTENT","my.txt")
            >>> file.name
            'my.txt'
            >>> file.parent == td.path
            True
        """
        file = self.path / name
        with open(file, "w") as f:
            f.write(text)
        return file


class ThreadPoolExecutor:
    """Executes in parallel using threads.

    Args:
        max_workers: Number of parallel workers to use.
        progress_bar: If progress bar output should be enabled--it can be useful to
            disable for tests and other non-interactive use cases.

    Examples:

        >>> from scale.olm.core import ThreadPoolExecutor
        >>> thread_pool_executor = ThreadPoolExecutor(max_workers=5,progress_bar=False)
        >>> def my_func(input):
        ...     output=input.upper()
        ...     return input,output
        >>> input_list = ["a","b"]
        >>> thread_pool_executor.execute(my_func,input_list)
        {'a': 'A', 'b': 'B'}

    Note that the input must be a string and should not have overlap with any other
    runs. For the purposes here, the input is almost always the name of an
    input file that is desired to be operated on.

    Attributes:
        max_workers: Input from __init__.
        progress_bar: Input from __init__.
    """

    def __init__(self, max_workers: int = 2, progress_bar: bool = True):
        self.max_workers = max_workers
        self.progress_bar = progress_bar

    def execute(self, my_func, input_list: list[str]):
        """Run a list of inputs through a function.

        Args:
            my_func: A function that takes a single string argument, which will
                be the elements of input_list, one at a time. This function must
                return a tuple (input,output) where input is the element of the input
                list and output is anything.
            input_list: A list of strings that represent the input for each run of my_func.

        Returns:
            A dictionary of results, results[input]=output.
        """
        import concurrent.futures

        # We can use a with statement to ensure threads are cleaned up promptly
        results = {}
        with tqdm(total=len(input_list), disable=not self.progress_bar) as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Start the load operations and mark each future with its URL
                submits = {
                    executor.submit(my_func, input): input for input in input_list
                }
                for future in concurrent.futures.as_completed(submits):
                    input, output = future.result()
                    results[input] = output
                    pbar.update(1)
        return results


class ScaleOutfile:
    """Extracts basic information from a SCALE main output file.

    Args:
        outfile (str): The path to the SCALE output file.

    Examples:

        >>> info = ScaleOutfile(scale_outfile)
        >>> info.outfile == scale_outfile
        True
        >>> info.version
        '6.3.0'
        >>> len(info.sequence_list)
        1
        >>> s0 = info.sequence_list[0]
        >>> s0['sequence']
        't-depl'
        >>> s0['product']
        'TRITON'
        >>> s0['runtime_seconds']
        35.2481

    Attributes:
        outfile (str): The path to the SCALE .out file.
        sequence_list (list[dict]): Information extracted for each sequence in order
           - sequence (str)
           - runtime_seconds (float)
           - product (str)
    """

    @staticmethod
    def get_product_name(sequence: str) -> str:
        """Maps the sequence information to a product name.

        Args:
            sequence: sequence name.

        Returns:
            Corresponding product name.

        Examples:

            >>> ScaleOutfile.get_product_name('t-depl-1d')
            'TRITON'

        """
        products = {"t-depl-1d": "TRITON", "t-depl": "TRITON"}
        return products.get(sequence, "UNKNOWN")

    def __init__(self, outfile: str):
        """
        Initializes the ScaleOutfile instance.

        Args:
            outfile (str): The path to the SCALE output file.

        Examples:

        >>> info = ScaleOutfile(scale_outfile)
        >>> info.version
        '6.3.0'

        """
        self.outfile = Path(outfile)
        self.version = None

        self._parse_info()
        for sequence in self.sequence_list:
            sequence["product"] = ScaleOutfile.get_product_name(sequence["sequence"])

    def _parse_info(self) -> None:
        """
        Internal routine to parse the runtime and sequence information from the output file.
        """

        self.sequence_list = []
        with open(self.outfile, "r") as f:
            for line in f.readlines():
                version_match = re.search(r"\*\s+SCALE (\d+[^ ]+)", line)
                if version_match:
                    self.version = version_match.group(1).strip()
                runtime_match = re.search(
                    r"([^\s]+) finished\. used (\d+\.\d+) seconds\.", line
                )
                if runtime_match:
                    self.sequence_list.append(
                        {
                            "runtime_seconds": float(runtime_match.group(2)),
                            "sequence": runtime_match.group(1),
                        }
                    )

    @staticmethod
    def parse_burnups_from_triton_output(output):
        """Parse the table that looks like this:

        .. code:: text

            Sub-Interval   Depletion   Sub-interval    Specific      Burn Length  Decay Length   Library Burnup
                 No.       Interval     in interval  Power(MW/MTIHM)     (d)          (d)           (MWd/MTIHM)
            ----------------------------------------------------------------------------------------------------
            ----------------------------------------------------------------------------------------------------
                    0     ****Initial Bootstrap Calculation****                                      0.00000E+00
                    1          1                1          40.000      25.000         0.000          5.00000e+02
                    2          1                2          40.000     300.000         0.000          7.00000e+03
                    3          1                3          40.000     300.000         0.000          1.90000e+04
                    4          1                4          40.000     312.500         0.000          3.12500e+04
                    5          1                5          40.000     312.500         0.000          4.37500e+04
                    6          1                6          40.000     333.333         0.000          5.66667e+04
                    7          1                7          40.000     333.333         0.000          7.00000e+04
                    8          1                8          40.000     333.333         0.000          8.33333e+04
            ----------------------------------------------------------------------------------------------------

        """
        burnup_list = []
        with open(output, "r") as f:
            n = 0
            found = False
            for line in f.readlines():
                words = line.split()
                if words == [
                    "Sub-Interval",
                    "Depletion",
                    "Sub-interval",
                    "Specific",
                    "Burn",
                    "Length",
                    "Decay",
                    "Length",
                    "Library",
                    "Burnup",
                ]:
                    found = True
                if found:
                    n += 1
                    if n > 4 and line.strip().startswith("-----"):
                        found = False
                    elif n > 4:
                        bu = float(line.split()[-1])
                        burnup_list.append(bu)
        return burnup_list

    @staticmethod
    def get_runtime(output):
        with open(output, "r") as f:
            for line in f.readlines():
                tokens = line.strip().split(" ")
                # t-depl finished. used 35.2481 seconds.
                if len(tokens) >= 5:
                    if (
                        tokens[1] == "finished."
                        and tokens[2] == "used"
                        and tokens[4] == "seconds."
                    ):
                        return float(tokens[3])
        return 0


class Obiwan:
    """Wrap obiwan."""

    def __init__(self, obiwan):
        self.obiwan = obiwan

    @staticmethod
    def get_history_from_f71(obiwan, f71, caseid0):
        """
        Parse the history of the form as follows for 6.3 series:

        .. code:: text

             pos         time        power         flux      fluence       energy    initialhm libpos   case   step DCGNAB
             (-)          (s)         (MW)    (n/cm2-s)      (n/cm2)        (MWd)      (MTIHM)    (-)    (-)    (-)    (-)
               1  0.00000e+00  4.00000e+01  8.11143e+14  0.00000e+00  0.00000e+00  1.00000e+00      1      1      0 DC----
               2  2.16000e+06  4.00000e+01  6.22529e+14  1.53582e+21  1.00000e+03  1.00000e+00      1      1     10 DC----
               3  2.16000e+07  4.00000e+01  4.26681e+14  8.78948e+21  1.00000e+04  1.00000e+00      2      1     10 DC----
               4  5.40000e+07  4.00000e+01  4.26566e+14  1.34274e+22  2.50000e+04  1.00000e+00      3      1     10 DC----
               5  1.08000e+08  4.00000e+01  4.31263e+14  2.30677e+22  5.00000e+04  1.00000e+00      4      1     10 DC----
               6  1.51200e+08  4.00000e+01  4.32303e+14  1.86058e+22  7.00000e+04  1.00000e+00      5      1     10 DC----
               7  1.94400e+08  4.00000e+01  4.33742e+14  1.86669e+22  9.00000e+04  1.00000e+00      6      1     10 DC----
               8  2.37600e+08  4.00000e+01  4.35733e+14  1.87415e+22  1.10000e+05  1.00000e+00      7      1     10 DC----

        Parse the history of the form as follows for 7.0 series:

        .. code:: text

            pos         time        power         flux      fluence       energy    initialhm       volume libpos   case   step DCGNAB
            (-)          (s)         (MW)   (n/cm^2-s)     (n/cm^2)        (MWd)      (MTIHM)       (cm^3)    (-)    (-)    (-)    (-)
              1  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  1.00000e+00  1.09091e+05      1     10      0 DC----
              2  2.16000e+06  3.99302e+01  2.77611e+14  5.99639e+20  9.98255e+02  1.00000e+00  1.09091e+05      2     10      1 DC----
              3  2.16000e+07  3.99294e+01  2.88762e+14  6.21316e+21  9.98238e+03  1.00000e+00  1.09091e+05      3     10      2 DC----
              4  5.40000e+07  3.99271e+01  3.13691e+14  1.63767e+22  2.49551e+04  1.00000e+00  1.09091e+05      4     10      3 DC----
              5  8.10000e+07  3.99215e+01  3.42857e+14  2.56339e+22  3.74305e+04  1.00000e+00  1.09091e+05      5     10      4 DC----
              6  1.08000e+08  3.99155e+01  3.70174e+14  3.56286e+22  4.99041e+04  1.00000e+00  1.09091e+05      6     10      5 DC----
              7  1.29600e+08  3.99087e+01  3.95311e+14  4.41673e+22  5.98813e+04  1.00000e+00  1.09091e+05      7     10      6 DC----
              8  1.51200e+08  3.99026e+01  4.18116e+14  5.31986e+22  6.98569e+04  1.00000e+00  1.09091e+05      8     10      7 DC----

        """
        # TODO: REMOVE THIS CALL TO RUN COMMAND IN FAVOR OF Obiwan class.
        text0 = run_command(f"{obiwan} view -format=info {f71}", echo=False)

        # Start the text with " pos " which should be the first thing on the header column
        # line always and the last thing the "D - state definition present" label.
        text = text0[text0.find(" pos ") : text0.find("D - state definition present")]
        header = text.split("\n")[0]
        columns = header.split()
        j_caseid = columns.index("case")
        ncolumns = j_caseid + 3

        burndata = list()
        initialhm0 = None
        last_days = 0.0
        i = 0
        for line in text.split("\n")[2:]:
            i += 1
            tokens = line.rstrip().split()
            if len(tokens) != ncolumns:
                break
            caseid = tokens[j_caseid]
            if str(caseid0) == str(caseid):
                days = float(tokens[1]) / 86400.0
                if days == 0.0:
                    initialhm0 = float(tokens[6])
                else:
                    burndata.append(
                        {"power": float(tokens[2]), "burn": (days - last_days)}
                    )
                last_days = days
        return {"burndata": burndata, "initialhm": initialhm0}


class ScaleRunner:
    """A basic wrapper class around SCALE Runtime Environment (scalerte).

    Initialize and check the runtime for various key quantities.

    Args:
        scalerte_path: The path to the runtime, e.g. /my/install/bin/scalerte

    Examples:

    .. code:: python

        runner = ScaleRunner('/my/install/bin/scalerte')
        runner.version
        runner.run('/path/to/scale.inp')

    Attributes:
        scalerte_path`: The path to the scalerte executable.
        args: Arguments to use when invoking SCALE (see set_args)
        version: The version of the SCALE Runtime Environment.
        data_dir: The path to the SCALE data directory.
        data_size: The size of the SCALE data directory.
    """

    @staticmethod
    def _default_do_not_run():
        """Provided as a way to override behavior for testing."""
        return False

    def __init__(self, scalerte_path: pathlib.Path, do_not_run: bool = None):
        if do_not_run == None:
            do_not_run = ScaleRunner._default_do_not_run()
        self.do_not_run = do_not_run

        self.scalerte_path = Path(scalerte_path)
        if not self.do_not_run:
            if not self.scalerte_path.exists():
                raise ValueError(
                    f"Path to SCALE Runtime Environment, {self.scalerte_path} does not exist!"
                )
        self.args = ""
        self.version = self._get_version(self.scalerte_path, do_not_run)
        self.data_dir = self._get_data_dir(self.scalerte_path)
        self.data_size = self._get_data_size(self.data_dir)

    def __str__(self):
        p = {}
        for k, v in self.__dict__.items():
            p[k] = str(v)
        return json.dumps(p, indent=4)

    @staticmethod
    def _get_version(scalerte_path: str, do_not_run: bool = False) -> str:
        """Internal method to get the SCALE version.

        Returns:
            The version string as MAJOR.MINOR.PATCH
        """
        if do_not_run:
            return ""

        version = subprocess.run(
            [scalerte_path, "-V"],
            capture_output=True,
            text=True,
        ).stdout.split(" ")[2]
        return version

    @staticmethod
    def _get_data_dir(scalerte_path: str) -> pathlib.Path:
        """Internal method to get the SCALE data directory.

        If the environmental variable DATA
        or SCALE_DATA_DIR is set then prefer that. Otherwise use the installation convention of
        scalerte being installed to /x/bin and data at /x/data.

        NOTE: This does not verify if the data directory exists or not!

        Returns:
            The expected data directory location.
        """
        if "SCALE_DATA_DIR" in os.environ:
            data_dir = os.environ["SCALE_DATA_DIR"]
        elif "DATA" in os.environ:
            data_dir = os.environ["DATA"]
        else:
            data_dir = Path(scalerte_path).parent.parent / "data"

        return Path(data_dir)

    @staticmethod
    def _rerun_cache_name(output_file: str) -> pathlib.Path:
        """Internal method to return the cache name corresponding to an output file.

        Args:
            output_file: The output file name.

        Returns:
            The cache name.
        """
        return Path(output_file).with_suffix(".run.json")

    @staticmethod
    def _get_data_size(data_dir: str) -> int:
        """Internal method to calculate the total file size of the contents of the data directory.

        Args:
            data_dir: The path to the SCALE data directory.

        Returns:
            Size in bytes of the data directory or -1 if the path does not exist.
        """
        if not Path(data_dir).exists():
            return -1

        return int(
            sum(
                element.stat().st_size
                for element in pathlib.Path(data_dir).glob("**/*")
                if element.is_file()
            )
        )

    @staticmethod
    def _determine_if_rerun(output_file, input_file_hash, data_size, version):
        """Internal method to check that input_file is the same.

        Uses a hash of the input, as well as data_size and version before committing
        to rerun SCALE. Based on the internal convention that this ScaleRunner wrapper
        drops a file of known suffix.

        Args:
            output_file (pathlib.Path): The path to the output file.
            input_file (pathlib.Path): The path to the input file.
            data_size (int): The size of the data directory.
            version (str): The version of the SCALE Runtime Environment.

        Returns:
            bool: True if the input file needs to be rerun, False otherwise.
            dict: Empty if rerunning needed, what was read off disk otherwise.
        """
        if output_file.exists():
            sr = ScaleRunner._rerun_cache_name(output_file)
            if sr.exists():
                with open(sr, "r") as f:
                    old = json.load(f)
                    if (
                        old["input_file_hash"] == input_file_hash
                        and old["data_size"] == data_size
                        and old["version"] == version
                    ):
                        return False, old
        return True, {
            "input_file_hash": input_file_hash,
            "data_size": data_size,
            "version": version,
        }

    @staticmethod
    def _scrape_errors_from_message_file(message_file):
        """Internal method to scrape errors from the SCALE message (.msg) file.

        Args:
            message_file (str): name of message file

        Returns:
            list[str]: list of errors
        """
        errors = []
        try:
            with open(message_file, "r") as f:
                for line in f:
                    if "Error" in line:
                        errors.append(line.strip())
        except (IOError, FileNotFoundError) as e:
            errors.append(
                f"Error occurred while trying to scrape errors from {message_file}: {e}"
            )
        return errors

    def set_args(self, args):
        """Set the arguments to use for subsequent run calls.

        Args:
            args (str): arguments as a single string
        """
        self.args = args

    def _run_kernel(command_line: str, input_file: pathlib.Path) -> int:
        """Kernel for running, mostly for enabling mocking/testing.

        Keep it simple, just return the returncode.

        Returns:

        """
        result = subprocess.run(
            command_line + " " + str(input_file),
            shell=True,
            capture_output=True,
            text=True,
        )
        return result.returncode

    def run(self, input_file):
        """Run an input file through SCALE.

        Verify first that the file was not already successfully run.

        Args:
            input_file: path to the input file
            args: arguments

        Raises:
            ValueError: If the input file does not exist.
            ValueError: If the SCALE data directory does not exist.

        Returns:
            str: path to input_file as pssed in
            dict[str,]: dictionary of arbitrary runtime data to pass out to the user
        """
        if not Path(input_file).exists():
            raise ValueError(f"SCALE input file, {input_file} does not exist!")

        input_file_hash = FileHasher(input_file).id
        output_file = Path(input_file).with_suffix(".out")
        rerun, data = self._determine_if_rerun(
            output_file, input_file_hash, self.data_size, self.version
        )
        command_line = f"{self.scalerte_path} {self.args}"
        message_file = output_file.with_suffix(".msg")

        start = time.time()
        if rerun:
            if self.do_not_run:
                returncode = 0
            else:
                if not self.data_dir.exists():
                    raise ValueError(
                        f"Path to SCALE Data was not found! Either 1) set the environment variable DATA or 2) symlink the data directory to {self.data_dir}."
                    )
                returncode = ScaleRunner._run_kernel(command_line, input_file)

            # If run was not successful, move output to .FAILED
            success = returncode == 0
            errors = []
            if not success:
                output_file0 = str(output_file)
                output_file = output_file0 + ".FAILED"
                shutil.move(output_file0, output_file)
                errors = self._scrape_errors_from_message_file(message_file)
            data = {
                "returncode": returncode,
                "success": success,
                "errors": errors,
                "command_line": command_line,
                "input_file": str(input_file),
                "output_file": str(output_file),
                "message_file": str(message_file),
                "data_size": self.data_size,
                "scale_runtime_seconds": float(time.time() - start),
                "data_dir": str(self.data_dir),
                "scalerte_path": str(self.scalerte_path),
                "input_file_hash": str(FileHasher(input_file).id),
                "version": self.version,
            }
            sr = ScaleRunner._rerun_cache_name(output_file)
            with open(sr, "w") as f:
                json.dump(data, f)

        runtime_seconds = float(time.time() - start)
        data["runtime_seconds"] = runtime_seconds
        data["rerun"] = rerun

        return str(input_file), data


class ArpInfo:
    """
    Handle the ARPDATA.TXT format for ORIGEN reactor libraries.


    """

    def __init__(self):
        self.name = ""
        self.lib_list = None
        self.perm_index = None
        self.fuel_type = ""
        self.block = ""
        self.burnup_list = []

    def init_block(self, name, block):
        """Initialize data from a single block of arpdata WITHOUT the ! line"""

        self.name = name
        self.block = block

        if self.name.startswith("mox_"):
            self.fuel_type = "MOX"
        elif self.name.startswith("act_"):
            self.fuel_type = "ACT"
        else:
            self.fuel_type = "UOX"

        self.lib_list = []

        tokens = self.block.split()
        if self.fuel_type == "UOX":
            ne = int(tokens[0])
            nm = int(tokens[1])
            nb = int(tokens[2])
            s = 3
            self.enrichment_list = [float(x) for x in tokens[s : s + ne]]
            s += ne
            self.mod_dens_list = [float(x) for x in tokens[s : s + nm]]
            s += nm
            for i in range(ne * nm):
                filename = tokens[s].replace("'", "").replace('"', "")
                self.lib_list.append(filename)
                s += 1
            self.burnup_list = [float(x) for x in tokens[s : s + nb]]

        elif self.fuel_type == "MOX":
            np = int(tokens[0])
            ne = int(tokens[1])
            nd = int(tokens[2])
            nm = int(tokens[3])
            nb = int(tokens[4])
            s = 5
            self.pu_frac_list = [float(x) for x in tokens[s : s + np]]
            s += np
            self.pu239_frac_list = [float(x) for x in tokens[s : s + ne]]
            s += ne
            dummy_list = [float(x) for x in tokens[s : s + nd]]
            s += nd  # Skip dummy entry
            self.mod_dens_list = [float(x) for x in tokens[s : s + nm]]
            s += nm
            for i in range(np * ne * nd * nm):
                filename = tokens[s].replace("'", "").replace('"', "")
                self.lib_list.append(filename)
                s += 1
            self.burnup_list = [float(x) for x in tokens[s : s + nb]]
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unknown (UOX/MOX)".format(self.fuel_type)
            )

    @staticmethod
    def parse_arpdata(file):
        """Simple function to parse the blocks of arpdata.txt"""
        blocks = dict()
        with open(file, "r") as f:
            for line in f.readlines():
                if line.startswith("!"):
                    name = line.strip()[1:]
                    blocks[name] = ""
                else:
                    blocks[name] += line
        return blocks

    def init_uox(self, name, lib_list, enrichment_list, mod_dens_list):
        """Initialize UOX data for arpdata.txt format from a list of data."""

        # Convert to interpolation space, assuming correct set up.
        self.name = name
        self.fuel_type = "UOX"
        self.enrichment_list = sorted(set(enrichment_list))
        self.mod_dens_list = sorted(set(mod_dens_list))
        self.burnup_list = []
        self.block = ""

        # Initialize permutation_index storage.
        n = self.num_libs()
        self.perm_index = [None] * n
        self.lib_list = [None] * n
        nperm = len(lib_list)
        if nperm != n:
            raise ValueError(
                f"number of permutations {nperm} must match number of libraries in the grid {n}"
            )

        # Lists come in in permutation order.
        for k in range(n):
            # Get the data in permutation order, find the right index in arp order.
            e = enrichment_list[k]
            m = mod_dens_list[k]
            ie = self.enrichment_list.index(e)
            im = self.mod_dens_list.index(m)

            # Arp index to permutation index.
            i = self.get_index_by_dim((ie, im))
            self.perm_index[i] = k
            self.lib_list[i] = lib_list[k]

    def init_mox(self, name, lib_list, pu239_frac_list, pu_frac_list, mod_dens_list):
        """Initialize MOX data for arpdata.txt format from a list of data."""

        # Convert to interpolation space, assuming correct set up.
        self.name = name
        self.fuel_type = "MOX"
        self.pu239_frac_list = sorted(set(pu239_frac_list))
        self.pu_frac_list = sorted(set(pu_frac_list))
        self.mod_dens_list = sorted(set(mod_dens_list))
        self.burnup_list = []
        self.block = ""

        # Initialize permutation_index storage.
        n = self.num_libs()
        self.perm_index = [None] * n
        self.lib_list = [None] * n
        nperm = len(lib_list)
        if nperm != n:
            raise ValueError(
                f"number of permutations {nperm} must match number of libraries in the grid {n}"
            )

        # Lists come in in permutation order.
        for k in range(n):
            # Get the data in permutation order, find the right index in arp order.
            e = pu239_frac_list[k]
            p = pu_frac_list[k]
            m = mod_dens_list[k]
            ie = self.pu239_frac_list.index(e)
            ip = self.pu_frac_list.index(p)
            im = self.mod_dens_list.index(m)

            # Arp index to permutation index.
            i = self.get_index_by_dim((im, ie, ip))
            self.perm_index[i] = k
            self.lib_list[i] = lib_list[k]

    def get_canonical_filename(self, dim, ext):
        if self.fuel_type == "UOX":
            (ie, im) = dim
            e = self.enrichment_list[ie]
            m = self.mod_dens_list[im]
            return "{}_e{:04d}w{:04d}{}".format(
                self.name, int(100 * e), int(1000 * m), ext
            )
        elif self.fuel_type == "MOX":
            (im, ie, ip) = dim
            m = self.mod_dens_list[im]
            e = self.pu239_frac_list[ie]
            p = self.pu_frac_list[ip]
            return "{}_e{:04d}v{:04d}w{:04d}{}".format(
                self.name,
                int(100 * p),
                int(100 * e),
                int(1000 * m),
                ext,
            )
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unknown (UOX/MOX)".format(self.fuel_type)
            )

    def set_canonical_filenames(self, ext):
        # We can keep track of filename counts so we are sure we don't create a duplicate.
        counts = set()

        n = self.num_libs()
        self.origin_lib_list = [None] * n
        for i in range(n):
            dim = self.get_dim_by_index(i)
            filename = self.get_canonical_filename(dim, ext)
            if filename in counts:
                raise ValueError(
                    f"canonical filename={filename} has already been used--most likely due to too small grid spacing!"
                )
            counts.add(filename)
            self.origin_lib_list[i] = self.lib_list[i]
            self.lib_list[i] = filename

    def get_perm_by_index(self, i):
        """Get the original permutation index that created this ARP point in space by flat index of libraries."""
        return self.perm_index[i]

    def get_lib_by_index(self, i):
        """Get the library by flat index."""
        if self.fuel_type == "UOX":
            return self.lib_list[i]
        elif self.fuel_type == "MOX":
            return self.lib_list[i]
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unknown (UOX/MOX)".format(self.fuel_type)
            )

    def get_index_by_dim(self, dim):
        """Get the flat index by a dimensional tuple."""

        # Dimension sizes.
        dims = self.get_dims()

        # Initialize tuple of lists, one for each dimension.
        n = self.num_libs()
        multi_index = tuple([dim[i]] for i in range(len(dims)))

        # Generate a flat index by that dimension.
        a = np.ravel_multi_index(multi_index, dims)
        return a[0]

    def get_dims(self):
        """Get the total dimension size."""
        if self.fuel_type == "UOX":
            ne = len(self.enrichment_list)
            nm = len(self.mod_dens_list)
            return (ne, nm)
        elif self.fuel_type == "MOX":
            np = len(self.pu_frac_list)
            ne = len(self.pu239_frac_list)
            nm = len(self.mod_dens_list)
            return (nm, ne, np)
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unknown (UOX/MOX)".format(self.fuel_type)
            )

    def get_space(self):
        """Get the dictionary that describes this point in space."""
        if self.fuel_type == "UOX":
            return {
                "mod_dens": {
                    "grid": self.mod_dens_list,
                    "desc": "Moderator density (g/cc)",
                },
                "enrichment": {
                    "grid": self.enrichment_list,
                    "desc": "U-235 enrichment (wt%)",
                },
                "burnup": {
                    "grid": self.burnup_list,
                    "desc": "energy release/burnup (MWd/MTIHM)",
                },
            }
        elif self.fuel_type == "MOX":
            return {
                "mod_dens": {"grid": self.mod_dens_list, "desc": ""},
                "pu_frac": {"grid": self.pu_frac_list, "desc": ""},
                "pu239_frac": {"grid": self.pu239_frac_list, "desc": ""},
                "burnup": {"grid": self.burnup_list, "desc": ""},
            }
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unknown (UOX/MOX)".format(self.fuel_type)
            )

    @staticmethod
    def _find_closest(old_list, new_list):
        # Find closest indices in `old_list` for each value in `new_list`
        indices = [np.abs(np.asarray(old_list) - val).argmin() for val in new_list]
        return sorted(list(set(indices)))

    def restrict(self, axis_name, keep_values):
        """Restrict values on the axis.

        Returns a new arpinfo object.

        Note that it does not modify the libraries themselves!
        So burnups must be modified in place on the libraries outside this function.
        """
        # Initialize new arpinfo with restricted data.
        arpinfo = ArpInfo()

        # UOX option
        if self.fuel_type=="UOX":
            (ne,nm) = self.get_dims()
            nb = len(self.burnup_list)
            ie_list = range(ne)
            im_list = range(nm)
            ib_list = range(nb)
            if axis_name=="enrichment":
                ie_list = ArpInfo._find_closest(self.enrichment_list,keep_values)
            elif axis_name=="mod_dens":
                im_list = ArpInfo._find_closest(self.mod_dens_list,keep_values)
            elif axis_name=="times" or axis_name=="burnup":
                ib_list = ArpInfo._find_closest(self.burnup_list,keep_values)
            else:
                raise ValueError(f"Restriction can only be called on enrichment, mod_dens, times, burnup axes. Not {axis_name}.")

            new_lib_list = []
            new_burnup_list = []
            new_enrichment_list = []
            new_mod_dens_list = []
            for im in im_list:
                for ie in ie_list:
                    i = self.get_index_by_dim((ie,im))
                    new_lib_list.append( self.get_lib_by_index(i) )
                    new_mod_dens_list.append( self.mod_dens_list[im] )
                    new_enrichment_list.append( self.enrichment_list[ie] )
            for ib in ib_list:
                new_burnup_list.append( self.burnup_list[ib] )

            # Update with restricted data.
            arpinfo.init_uox(self.name,new_lib_list,new_enrichment_list,new_mod_dens_list)
            arpinfo.burnup_list = new_burnup_list

        elif self.fuel_type=="MOX":
            raise ValueError(f"MOX restrict not yet implemented.")
        else:
            raise ValueError(f"fuel_type must be MOX or UOX, found: {self.fuel_type}")
        return arpinfo


    def num_libs(self):
        """Get the total number of libraries."""
        return np.prod(self.get_dims())

    def get_dim_by_index(self, i):
        """Get the dimension tuple from the flat index."""
        return np.unravel_index(i, self.get_dims())

    def interptags_by_index(self, i):
        """Get the interpolation tags from the flat index."""
        if self.fuel_type == "UOX" or self.fuel_type == "MOX":
            d = self.interpvars_by_index(i)
            y = ["{}={}".format(x, d[x]) for x in d]
            return ",".join(y)
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unknown (UOX/MOX)".format(self.fuel_type)
            )

    def interpvars_by_index(self, i):
        """Get the interpolation variables from the flat index."""
        if self.fuel_type == "UOX":
            (ie, im) = self.get_dim_by_index(i)
            return {
                "enrichment": self.enrichment_list[ie],
                "mod_dens": self.mod_dens_list[im],
            }
        elif self.fuel_type == "MOX":
            (im, ie, ip) = self.get_dim_by_index(i)
            return {
                "pu239_frac": self.pu239_frac_list[ie],
                "pu_frac": self.pu_frac_list[ip],
                "mod_dens": self.mod_dens_list[im],
            }
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unknown (UOX/MOX)".format(self.fuel_type)
            )

    def get_arpdata(self):
        """Return the arpdata.txt file block for this data."""
        entry = ""
        if self.fuel_type == "UOX":
            ne = len(self.enrichment_list)
            nm = len(self.mod_dens_list)
            nb = len(self.burnup_list)
            entry += "{} {} {}\n".format(ne, nm, nb)
            entry += "\n".join([str(x) for x in self.enrichment_list]) + "\n"
            entry += "\n".join([str(x) for x in self.mod_dens_list]) + "\n"
            for i in range(ne * nm):
                entry += "'{}'\n".format(self.lib_list[i])
            entry += "\n".join([str(x) for x in self.burnup_list])
        elif self.fuel_type == "MOX":
            np = len(self.pu_frac_list)
            ne = len(self.pu239_frac_list)
            nm = len(self.mod_dens_list)
            nb = len(self.burnup_list)
            entry += "{} {} 1 {} {}\n".format(np, ne, nm, nb)
            entry += "\n".join([str(x) for x in self.pu_frac_list]) + "\n"
            entry += "\n".join([str(x) for x in self.pu239_frac_list]) + "\n"
            entry += "1\n"  # dummy entry
            entry += "\n".join([str(x) for x in self.mod_dens_list]) + "\n"
            for i in range(nm * np * ne):
                entry += "'{}'\n".format(self.lib_list[i])
            entry += "\n".join([str(x) for x in self.burnup_list])
        else:
            raise ValueError(
                "ArpInfo.fuel_type={} unknown (UOX/MOX)".format(self.fuel_type)
            )

        self.block = entry
        return "!{}\n{}".format(self.name, self.block)

    def create_temp_archive(self, arpdata_txt, temp_arc):
        """Create a temporary HDF5 archive file from an arpdata.txt."""
        # Remove the existing archive if it exists
        if os.path.exists(temp_arc):
            os.remove(temp_arc)

        h5arc = None
        arpdir = arpdata_txt.parent / "arplibs"
        for i in range(self.num_libs()):
            lib = Path(self.lib_list[i])
            # Create an archive and delete the library dataset so we can use
            # links.
            if not h5arc:
                shutil.copyfile(arpdir / lib, temp_arc)
                h5arc = h5py.File(temp_arc, "a")
                del h5arc["incident"]["neutron"]["lib1"]

            h5arc["incident"]["neutron"][f"lib{i+1}"] = h5py.ExternalLink(
                str(arpdir / lib), "/incident/neutron/lib1"
            )

        return h5arc


class ReactorLibrary:
    """
    Simple class to read an ORIGEN ReactorLibrary into memory. The hierarchy of ORIGEN
    data is a Transition Matrix is the necessary computational piece. A Library is a
    time-dependent sequence of Transition Matrices. A ReactorLibrary is a multi-dimensional
    interpolatable space of Libraries.
    """

    @staticmethod
    def _extract_transitions(h5_data):
        """Extract nuclide and transition identifiers.

        SIZZZAAA is an 8-digit nuclide identifier
            S   - single digit sublib (light nuclide LT, actinide AC, fission product FP)
            I   - isomeric state (0 ground, 1 first metastable)
            ZZZ - atomic number
            AAA - mass number

        It is possible that S=0 in a data set without sublibs.

        Here's a conversion from SIZZZAAA to TYP:EAm
        21092235 <-> AC:U235

        """
        nuclide_ids = [int(id) for id in h5_data["decay"]["nuclide_list"]]

        ts = h5_data["incident"]["neutron"]["TransitionStructure"]
        num_parents = ts["num_parents"][:]
        num_decay_parents = ts["num_decay_parents"][:]
        parent_positions = ts["parent_positions"][:]
        transition_ids = ts["transition_ids"][:]

        transitions = []
        tind = -1

        for i, did in enumerate(nuclide_ids):
            # Initialize parent to reaction map
            for count in range(num_parents[i]):
                tind += 1
                pid = nuclide_ids[parent_positions[tind] - 1]
                transitions.append( (did,transition_ids[tind],pid) )

        return nuclide_ids, transitions

    @staticmethod
    def duplicate_degenerate_axis_value(x0):
        """Create a second axis value for degenerate axes to enable gradient calculations.
        
        When an axis has only one value, numpy.gradient cannot compute gradients.
        This function creates a second value with positive spacing from the original.
        
        Args:
            x0: The original axis value
            
        Returns:
            The second axis value (x1) such that x1 > x0
            
        Examples:
            >>> ReactorLibrary.duplicate_degenerate_axis_value(0.723)
            0.773
            >>> ReactorLibrary.duplicate_degenerate_axis_value(-1.0)
            -0.95
            >>> ReactorLibrary.duplicate_degenerate_axis_value(0.0)
            0.05
            >>> ReactorLibrary.duplicate_degenerate_axis_value(100.0)
            105.0
        """
        # Always add a small positive offset to ensure dx > 0
        # regardless of the sign or magnitude of x0
        delta = max(0.05, 0.05 * abs(x0)) if abs(x0) > 1e-10 else 0.05
        return x0 + delta

    def __init__(self, file, name="", progress_bar=True):
        self.file = file

        # Initialize in-memory data structure.
        self.arpinfo = None
        self.arc = None
        if self.file.name == "arpdata.txt":
            blocks = ArpInfo.parse_arpdata(file)
            if name=="":
                raise ValueError(f"The `name` argument must be provided with arpdata.txt file formats, e.g. 'w17x17'.")
            self.arc = file.with_suffix(".arc.h5")
            self.name = name
            self.arpinfo = ArpInfo()
            self.arpinfo.init_block(name, blocks[name])
            h5 = self.arpinfo.create_temp_archive(file, self.arc)
        else:
            h5 = h5py.File(self.file, "r")
            self.arc = self.file

        # Get important axis variables.
        (
            self.axes_names,
            self.axes_values,
            self.axes_shape,
            self.ncoeff,
            self.nvec,
        ) = ReactorLibrary.extract_axes(h5)
        
        # Get nuclides and coefficient names.
        self.nuclide_ids, self.transitions = ReactorLibrary._extract_transitions(h5)

        # Populate coefficient data.
        self.coeff = np.zeros((*self.axes_shape, self.ncoeff))
        data = h5["incident"]["neutron"]
        for i in tqdm(data.keys(), disable=not progress_bar):
            if i != "TransitionStructure":
                d = ReactorLibrary.get_indices(
                    self.axes_names, self.axes_values, data[i]["tags"]["continuous"]
                )
                dn = (*d, slice(None), slice(None)) #state,time,transition
                self.coeff[dn] = data[i]["matrix"]

        # Add another point if the dimension only has one so that we can make it easier
        # to do operations like gradients.
        # This handles "degenerate axes" where all libraries have the same value for a
        # particular interpolation parameter. Without this, gradient calculations would
        # fail because numpy.gradient requires at least 2 points along each axis.
        n = len(self.axes_shape)
        for i in range(n):
            if self.axes_shape[i] == 1:
                self.axes_shape[i] = 2
                x0 = self.axes_values[i][0]
                x1 = ReactorLibrary.duplicate_degenerate_axis_value(x0)
                self.axes_values[i] = np.append(self.axes_values[i], x1)
                self.coeff = np.repeat(self.coeff, 2, axis=i)

    def restrict(self, axis_name, keep_values):
        """Restrict the data set returning a new one."""
        import copy
        new = copy.deepcopy(self)

        # Get the axis index.
        if axis_name not in self.axes_names:
            raise ValueError(f"Axis '{axis_name}' not found in axes_names.")
        axis_idx = list(self.axes_names).index(axis_name)

        # Restrict `self.axes_values` to the specified indices along the `axis_idx`
        axis_values = self.axes_values[axis_idx]
        keep_indices = ArpInfo._find_closest(axis_values,keep_values)

        new.axes_values[axis_idx] = axis_values[keep_indices]
        new.axes_shape[axis_idx] = len(new.axes_values[axis_idx])

        # Restrict `self.coeff` data along the specified axis
        # Always keep last index for coefficients, hence the +1.
        slicer = [slice(None)] * (len(self.axes_shape)+1)
        slicer[axis_idx] = keep_indices  # Apply restriction along `axis_idx`
        new.coeff = self.coeff[tuple(slicer)]

        # Restrict arpinfo.
        if new.arpinfo is not None:
            new.arpinfo = self.arpinfo.restrict(axis_name, keep_values)

        return new

    def save(self):

        # Write new arpdata.txt.
        if self.arpinfo!=None:
            self.arpinfo.get_arpdata
            with open(self.file, "w") as f:
                f.write(self.arpinfo.get_arpdata())

        # Write new data.
        with h5py.File(self.arc, "r+") as h5:
            data = h5["incident"]["neutron"]
            old_burnups = data["lib1"]["burnups"]
            new_burnups = self.axes_values[-1] # time is always last
            keep_burnup_indices = ArpInfo._find_closest(old_burnups,new_burnups)

            for i in data.keys():
                if i.startswith("lib"):
                    for s in ["burnups","fission_xs","flux","kappa_capture","kappa_fission","loss_xs","matrix","neutron_yields"]:
                        filtered = data[i][s][keep_burnup_indices]
                        del data[i][s]
                        data[i].create_dataset(s, data=filtered)

    @staticmethod
    def get_indices(axes_names, axes_values, point_data):
        y = [0] * len(point_data)
        for name in point_data:
            i = np.flatnonzero(axes_names == name)[0]
            iaxis = axes_values[i]
            value = point_data[name]
            diff = np.absolute(axes_values[i] - value)
            j = np.argmin(diff)
            y[i] = j
        return tuple(y)

    @staticmethod
    def extract_axes(h5):
        data = h5["incident"]["neutron"]
        dim_names = list()
        libs = list()
        ncoeff = 0
        nvec = 0
        for i in data.keys():
            if i != "TransitionStructure":
                libs.append(data[i])
                ncoeff = np.shape(data[i]["matrix"])[1]
                nvec = np.shape(data[i]["loss_xs"])[1]
                labels = data[i]["tags"]["continuous"]
                for x in labels:
                    dim_names.append(x)
        dim_names = list(np.unique(np.asarray(dim_names)))

        # create 1d dimensions array
        n = len(libs)
        dims = dict()
        for name in dim_names:
            dims[name] = np.zeros(n)
        times = list()
        D = 6
        for i in range(n):
            for name in libs[i]["tags"]["continuous"]:
                value = libs[i]["tags"]["continuous"][name]
                dims[name][i] = value[()].round(decimals=D)
            times.append(np.asarray(libs[i]["burnups"]).round(decimals=D))

        # determine values in each dimension and add time
        axes_names = list(dim_names)
        ndims = len(dim_names) + 1
        axes_values = [0] * ndims
        i = 0
        for name in dims:
            axes_values[i] = np.unique(dims[name].round(decimals=D))
            i += 1
        axes_names.append("times")
        axes_values[i] = times[0]

        # determine the shape/size of each dimension
        axes_shape = list(axes_values)
        for i in range(ndims):
            axes_shape[i] = len(axes_values[i])

        # convert names and shapes to np array before leaving
        axes_names = np.asarray(axes_names)
        axes_shape = np.asarray(axes_shape)
        return axes_names, axes_values, axes_shape, ncoeff, nvec


class RelAbsHistogram:
    def __init__(self, rhist, ahist):
        self.rhist = rhist
        self.ahist = ahist

    @staticmethod
    def plot_hist(
        x, image="", xlabel=r"$\log \tilde{h}_{ijk}$", ylabel=r"$\log h_{ijk}$"
    ):
        """Plot histograms from relative and absolute histogram data (rhist,ahist)."""
        from matplotlib.ticker import MaxNLocator,MultipleLocator

        plt.figure()
        eps = 1e-10
        vmin = 0
        vmax = 1
        cmin = 1e-5

        min_lim = int(np.log10(eps))
        max_lim = max(
            int(np.amax([np.log10(eps + x.rhist), np.log10(eps + x.ahist)])),
            -min_lim
        )
        nbins = max_lim-min_lim+1
        bins = np.linspace(min_lim, max_lim, nbins)

        h = plt.hist2d(
            np.log10(eps + x.rhist),
            np.log10(eps + x.ahist),
            bins=bins,
            alpha=1.0,
            cmin=cmin,
            vmin=vmin,
            vmax=vmax,
            density=True,
        )
        plt.colorbar(h[3])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().set_xticks(bins)  # Set x-axis ticks to match bin edges
        plt.gca().set_yticks(bins)  # Set y-axis ticks to match bin edges

        # Limit the number of tick labels to at most 5 without removing gridlines
        plt.gca().xaxis.set_major_locator(MultipleLocator(1))  # Ensure integer grid lines
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{int(val)}" if int(val) % 5 == 0 else ""))
        plt.gca().yaxis.set_major_locator(MultipleLocator(1))  # Ensure integer grid lines
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f"{int(val)}" if int(val) % 5 == 0 else ""))
        plt.grid(True)

        if image == "":
            plt.show()
        else:
            plt.savefig(image, bbox_inches="tight")


def _is_active_doctest():
    """Check if running the current module as a doctest.

    Note, this may not cover enough edge cases for widespread usage. It is
    just intended to enable OLM doctests.
    """
    import sys

    return __name__ == "__main__" or "_pytest.doctest" in sys.modules


# This enables succinct doctests for methods by using the existing data in extraglobs.
if _is_active_doctest():
    import doctest

    testing_temp_dir = TempDir()
    testing_rte_path = testing_temp_dir.write_file("", "scalerte")
    scale_outfile = testing_temp_dir.write_file(
        """
*                              SCALE 6.3.0                            *
t-depl finished. used 35.2481 seconds.""",
        "example.out",
    )
    doctest.testmod(
        extraglobs={
            "scalerte": ScaleRunner(testing_rte_path, do_not_run=True),
            "thread_pool_executor": ThreadPoolExecutor(
                max_workers=2, progress_bar=False
            ),
            "scale_outfile": scale_outfile,
        },
    )


class NuclideInventory:
    """Manages a time-dependent nuclide inventory."""

    def __init__(
        self,
        composition_manager,
        time,
        nuclide_amount,
        time_units="SECONDS",
        amount_units="MOLES",
    ):
        assert time_units == "SECONDS"
        assert amount_units == "MOLES"
        self.composition_manager = composition_manager
        self.time = time
        self.nuclide_amount = nuclide_amount

    @staticmethod
    def _nuclide_color(id, weight=0.95):
        import hashlib

        f_color = (
            int.from_bytes(hashlib.md5(id.encode("utf-8")).digest(), "big") % 256
        ) / 256.0
        color = np.asarray(plt.get_cmap("jet")(f_color))
        color[0:3] *= weight  # Leave out alpha
        return color

    @staticmethod
    def _nice_label0(composition_manager, id):
        eam = composition_manager.eam(id)
        e, a, i = composition_manager.parse_eam_to_eai(eam)
        Ee = e[0].upper()
        if len(e) > 1:
            Ee += e[1:]
        if i == 0:
            m = ""
        elif i == 1:
            m = "m"
        else:
            m = "m" + str(i)
        return r"${^{" + str(a) + r"\mathrm{" + m + r"}}}\mathrm{" + Ee + "}$"

    def _nice_label(self, id):
        return NuclideInventory._nice_label0(self.composition_manager, id)

    def get_hm_mass(self, min_z=92, max_z=1000):
        cm = self.composition_manager
        hm_mass = np.zeros(len(self.time))
        for id, amount in self.nuclide_amount.items():
            m = cm.mass(id)
            i, z, a = cm.parse_izzzaaa(id)
            if z >= min_z and z <= max_z:
                hm_mass += amount * m
        return hm_mass

    def get_amount(self, nuclide, units="MOLES"):
        id = self.composition_manager.izzzaaa(nuclide)
        if units == "GRAMS":
            d = self.composition_manager.mass(id)
        elif units == "MOLES":
            d = 1.0
        else:
            raise ValueError(f"amount units {units} not recognized!")
        amount = np.array(self.nuclide_amount[id]) * d
        return amount

    def get_time(self, units="SECONDS"):
        time_conv = {
            "DAYS": 1.0 / 86400.0,
            "HOURS": 1.0 / 3600.0,
            "MINUTES": 1.0 / 60.0,
            "SECONDS": 1.0,
            "YEARS": 1.0 / (86400.0 * 365.25),
        }
        assert units.upper() in time_conv
        c = time_conv[units.upper()]
        return self.time * c

    def get_nuclides(self,nice_label=False):
        """Return the nuclides in this system"""
        nuclides = list(self.nuclide_amount.keys())
        if nice_label:
            nuclides = [self._nice_label(id) for id in nuclides]
        return nuclides

    def wrel_diff(self, nuclide, other_self, units="GRAMS"):
        """Create a weighted relative difference."""
        test = self.get_amount(nuclide, units=units)
        ref = other_self.get_amount(nuclide, units=units)
        max = np.amax(ref)
        return (test - ref) / max

    def rel_diff(self, nuclide, other_self, units="GRAMS", eps=1e-12):
        """Create a relative difference."""
        test = self.get_amount(nuclide, units=units)
        ref = other_self.get_amount(nuclide, units=units)
        return (test + eps) / (ref + eps) - 1.0

    def plot_nuclide_amounts(
        self,
        nuclide_list,
        time_units="DAYS",
        amount_units="GRAMS",
        color_weight=0.95,
        plot_fun=plt.plot,
        amount_mult=1.0,
        **kwargs,
    ):
        time = self.get_time(units=time_units)
        amount_map = {}
        for nuclide in nuclide_list:
            iden = self.composition_manager.izzzaaa(nuclide)
            amount = amount_mult * self.get_amount(iden, units=amount_units)
            amount_map[nuclide] = amount
            plot_fun(
                time,
                amount,
                color=NuclideInventory._nuclide_color(iden, weight=color_weight),
                label=self._nice_label(iden),
                **kwargs,
            )
        plt.xlabel("Time ({})".format(time_units.lower()))
        plt.ylabel("Amount ({})".format(amount_units.lower()))
        plt.legend()
        return amount_map


class InventoryInterface:
    """Loads/saves and extracts data from the inventory interface file."""

    def __init__(self, input):
        self.data = copy.deepcopy(input["data"])
        self.responses = copy.deepcopy(input["responses"])
        self.definitions = copy.deepcopy(input["definitions"])
        self.composition_manager = CompositionManager(self.data["nuclides"])

    def names(self):
        return self.responses.keys()

    def nuclide_inventory(self, name):
        response = self.responses[name]
        nvh = response["nuclideVectorHash"]
        ids = self.definitions["nuclideVectors"][nvh]
        time = np.array(response["time"])
        amount = response["amount"]
        nuclide_amount = {}
        for i in range(len(ids)):
            y0 = []
            for j in range(len(time)):
                y0.append(amount[j][i])
            nuclide_amount[ids[i]] = np.array(y0)

        return NuclideInventory(
            self.composition_manager,
            time,
            nuclide_amount,
            time_units=response["timeUnits"],
            amount_units=response["amountUnits"],
        )
