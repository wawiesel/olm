import numpy as np
from tqdm import tqdm, tqdm_notebook
import scale.olm.common as common
import scale.olm.run as run
import json
from pathlib import Path
import copy
import os


class CheckInfo:
    def __init__(self):
        self.test_pass = False


def sequencer(model, sequence):
    output = []

    try:
        # Process all the input.
        run_list = []
        i = 0
        for s in sequence:
            # Set the full name.
            name = s["_type"]
            if name.find(":") == -1:
                name = "scale.olm.check:" + name
            s["_type"] = name
            s["model"] = model

            common.logger.info(
                "Checking options for check={}, sequence={}".format(name, i)
            )
            i += 1

            # Initialize the class.
            this_class = common.fn_redirect(**s)
            run_list.append(this_class)

        # Read the archive.
        work_dir = Path(model["work_dir"])
        common.logger.info(f"Running checking in work dir={work_dir}")
        arpdata_txt = work_dir / "arpdata.txt"
        if arpdata_txt.exists():
            archive = common.Archive(arpdata_txt, model["name"])
        else:
            archive = common.Archive(f"{name}.arc.h5")

        # Execute in sequence.
        test_pass = True
        i = 0
        for r in run_list:
            common.logger.info("Running checking sequence={}".format(i))

            info = r.run(archive)
            output.append(info.__dict__)
            i += 1

            if not info.test_pass:
                test_pass = False

        common.logger.info("Finished without exception test_pass={}".format(test_pass))

    except ValueError as ve:
        common.logger.error(str(ve))

    return {"test_pass": test_pass, "sequence": output}


class GridGradient:
    """Class to compute the grid gradients"""

    @staticmethod
    def describe_params():
        return {
            "eps0": "minimum value",
            "epsa": "absolute epsilon",
            "epsr": "relative epsilon",
        }

    @staticmethod
    def default_params():
        c = GridGradient()
        return {
            "eps0": c.eps0,
            "epsa": c.epsa,
            "epsr": c.epsr,
            "target_q2": c.target_q2,
            "target_q1": c.target_q1,
        }

    def __init__(
        self, model=None, eps0=1e-20, epsa=1e-1, epsr=1e-1, target_q1=0.5, target_q2=0.7
    ):
        self.eps0 = eps0
        self.epsa = epsa
        self.epsr = epsr
        self.target_q1 = target_q1
        self.target_q2 = target_q2

    def run(self, archive):
        """Run the calculation and return post-processed results"""

        common.logger.info(
            "Running "
            + self.__class__.__name__
            + " check with params={}".format(json.dumps(self.__dict__))
        )
        self.__calc(archive)

        # After calc the self.ahist, rhist, khist, and rel_axes variables are ready to
        # compute metrics.
        info = self.info()
        common.logger.info(
            "Completed "
            + self.__class__.__name__
            + " with q1={:.2f} and q2={:.2f}".format(info.q1, info.q2)
        )

        return info

    def info(self):
        info = CheckInfo()
        info.name = self.__class__.__name__
        info.eps0 = self.eps0
        info.epsa = self.epsa
        info.epsr = self.epsr
        info.wa = int(
            np.logical_and((self.ahist > self.epsa), (self.rhist > self.epsr)).sum()
        )
        info.wr = int((self.rhist > self.epsr).sum())
        info.m = int(len(self.ahist))
        info.fr = float(info.wr) / info.m
        info.q1 = 1.0 - info.fr
        info.fa = float(info.wa) / info.m
        info.q2 = 1.0 - 0.9 * info.fa - 0.1 * info.fr

        info.test_pass_q1 = info.q1 >= info.target_q1
        info.test_pass_q2 = info.q2 >= info.target_q2
        info.test_pass = info.test_pass_q1 and info.test_pass_q2

        return info

    def __calc(self, archive):
        """Drives the set up for the kernel with archive as input"""

        self.rel_axes = list()
        for x_list in archive.axes_values:
            dx = x_list[-1] - x_list[0]
            x0 = x_list[0]
            z = list()
            for x in x_list:
                z.append((x - x0) / dx)
            self.rel_axes.append(z)
        common.logger.info("Finished computing relative values on axes")

        self.yreshape = np.moveaxis(archive.coeff, [-1], [0])
        common.logger.info("Finished reshaping coefficients")

        common.logger.info("Computing grid gradients ...")
        self.ahist, self.rhist, self.khist = GridGradient.__kernel(
            self.rel_axes, self.yreshape, self.eps0
        )
        common.logger.info("Finished computing grid gradients")

    @staticmethod
    def __kernel(rel_axes, yreshape, eps0):
        """Lowest level kernel for the calculation"""

        # Number of dimensions.
        n = len(rel_axes)

        # Number of coefficients.
        ncoeff = np.shape(yreshape)[0]

        # Initialize histogram variables.
        rhist = np.zeros(n * n * ncoeff)
        ahist = np.zeros(n * n * ncoeff)
        khist = np.zeros(n * n * ncoeff)

        # For each coefficient in the transition matrix.
        for k in tqdm(range(ncoeff)):
            # Get just the grid of values for this coefficient.
            y = yreshape[k, ...]

            # Compute the maximum value at any point in the grid.
            max_y = np.amax(y)
            if max_y <= 0:
                max_y = eps0

            # Compute the gradient dy/dx for all axes.
            yp = np.asarray(np.gradient(y, *rel_axes))

            # Iterate through every combination of two dimensions.
            # TODO: generalize to all dimensions
            for i in range(n):
                ypi = yp[i, ...]

                for j in range(n):
                    # Calculate the flat index.
                    iu = k * n * n + i * n + j

                    # Calculate the difference between gradients.
                    yda = np.absolute(np.diff(ypi, axis=j))
                    ahist[iu] = np.amax(yda)

                    # Calculate relative difference using max_y calculated earlier.
                    ydr = yda / max_y
                    rhist[iu] = np.amax(ydr)

                    # Remember the coefficient index of this particular gradient difference.
                    khist[iu] = k

        return ahist, rhist, khist


class LowOrderConsistency:
    """Check that we are consistent with the original calculation."""

    @staticmethod
    def describe_params():
        return {"nprocs": "number of processes to use to run consistency check"}

    @staticmethod
    def default_params():
        c = LowOrderConsistency()
        return {
            "eps0": c.eps0,
            "epsa": c.epsa,
            "epsr": c.epsr,
            "target_q2": c.target_q2,
            "target_q1": c.target_q1,
            "nprocs": c.nprocs,
        }

    def __init__(
        self,
        model=None,
        template=None,
        eps0=1e-12,
        epsa=1e-6,
        epsr=1e-3,
        target_q1=0.9,
        target_q2=0.95,
        nprocs=3,
        nuclide_compare=["0092235", "0094239"],
    ):
        self.template = template
        self.nprocs = nprocs
        self.nuclide_compare = nuclide_compare
        self.scalerte = model["scalerte"]
        self.config_dir = Path(model["dir"])
        self.work_dir = Path(model["work_dir"])
        self.check_dir = self.work_dir / "check" / Path(self.template).stem
        self.name = model["name"]
        self.obiwan = model["obiwan"]
        self.eps0 = eps0
        self.epsa = epsa
        self.epsr = epsr
        self.target_q1 = target_q1
        self.target_q2 = target_q2

    def info(self):
        info = CheckInfo()
        info.name = self.__class__.__name__

        info.eps0 = self.eps0
        info.epsa = self.epsa
        info.epsr = self.epsr
        info.target_q1 = self.target_q1
        info.target_q2 = self.target_q2
        if not self.run_success:
            info.test_pass = False
            return info

        # Create a base comparison data structure to repeat for every permutation.
        common.logger.info("Setting up detailed comparison structures...")
        info.nuclide_compare = dict()
        ntime = len(self.time_list)
        for nuclide in self.nuclide_compare:
            i = self.names.index(nuclide)
            common.logger.info(
                f"Found nuclide={nuclide} at index {i} for detailed comparison"
            )
            info.nuclide_compare[nuclide] = {
                "nuclide_index": i,
                "nuclide": nuclide,
                "time": self.time_list,
                "max_diff": list(np.zeros(ntime)),
                "min_diff": list(np.zeros(ntime)),
                "perms": [],
            }

        self.ahist = np.array(self.lo_list)
        self.rhist = np.array(self.lo_list)
        self.hi = np.array(self.hi_list)
        self.lo = np.array(self.lo_list)

        # For each permutation.
        common.logger.info("Calculating all comparison histogram data...")
        for k in range(len(self.lo_list)):
            # For each time.
            for j in range(len(self.lo_list[k])):
                osum = self.lo_list[k][j].sum()
                tsum = self.hi_list[k][j].sum()
                oden = self.lo_list[k][j] / osum
                tden = self.hi_list[k][j] / tsum
                self.lo[k, j, :] = oden
                self.hi[k, j, :] = tden
                self.ahist[k, j, :] = np.absolute(oden - tden)
                self.rhist[k, j, :] = np.absolute(
                    (oden + self.eps0) / (tden + self.eps0) - 1.0
                )

        # Extract each nuclide time series.
        common.logger.info("Calculating nuclide-wise comparisons...")
        for n in info.nuclide_compare:
            i_nuclide = info.nuclide_compare[n]["nuclide_index"]
            for k in range(len(self.lo_list)):
                lo = self.lo[k, :, i_nuclide]
                hi = self.hi[k, :, i_nuclide]
                err = (lo - hi) / (self.eps0 + np.amax(np.absolute(hi)))
                info.nuclide_compare[n]["perms"].append(
                    {
                        "k": k,
                        "lo": list(lo),
                        "hi": list(hi),
                        "(lo-hi)/max(|hi|)": list(err),
                    }
                )

        # Get maximum and min error across all permutations.
        common.logger.info("Calculating max/min across permutations...")
        for n, d in info.nuclide_compare.items():
            i_nuclide = d["nuclide_index"]
            for k in range(len(self.lo_list)):
                err = d["perms"][k]["(lo-hi)/max(|hi|)"]
                for j in range(len(self.time_list)):
                    d["max_diff"][j] = np.amax([err[j], d["max_diff"][j]])
                    d["min_diff"][j] = np.amin([err[j], d["min_diff"][j]])
            d["max_diff0"] = np.amax(
                [np.absolute(d["max_diff"]), np.absolute(d["min_diff"])]
            )

        self.ahist = np.ndarray.flatten(self.ahist)
        self.rhist = np.ndarray.flatten(self.rhist)

        info.wa = int(
            np.logical_and((self.ahist > self.epsa), (self.rhist > self.epsr)).sum()
        )
        info.wr = int((self.rhist > self.epsr).sum())
        info.m = int(len(self.ahist))
        info.fr = float(info.wr) / info.m
        info.q1 = 1.0 - info.fr
        info.fa = float(info.wa) / info.m
        info.q2 = 1.0 - 0.9 * info.fa - 0.1 * info.fr
        info.test_pass_q1 = info.q1 >= info.target_q1
        info.test_pass_q2 = info.q2 >= info.target_q2
        info.test_pass = info.test_pass_q1 and info.test_pass_q2
        # Other stats.
        info.mean_abs_diff = np.mean(self.ahist)
        info.mean_rel_diff = np.mean(self.rhist)
        info.std_abs_diff = np.std(self.ahist)
        info.std_rel_diff = np.std(self.rhist)

        return info

    def __run_lo_fidelity(self, do_run):
        """Run the LOWER fidelity calculation which should be consistent as possible with
        the already-complete higher order calculation."""

        # Load the template file.
        with open(self.config_dir / self.template, "r") as f:
            template_text = f.read()

        # Load the generate data.
        generate_json = self.work_dir / "generate.json"
        with open(generate_json, "r") as f:
            generate_d = json.load(f)

        # Load the build data.
        build_json = self.work_dir / "build.json"
        with open(build_json, "r") as f:
            build_d = json.load(f)

        # For each permutation.
        f71_list = list()
        for j in range(len(build_d["perms"])):
            # Convenience variables.
            generate = generate_d["perms"][j]
            build = build_d["perms"][j]
            base = generate["file"]

            # Save HIGH fidelity and LOWER fidelity f71 in a list.
            run_input = self.work_dir / base
            run_f71 = run_input.with_suffix(".f71")
            check_input = self.check_dir / base
            check_f71 = check_input.with_suffix(".f71")
            f71_list.append((run_f71, check_f71))

            if do_run:
                # Extract the fuel power / burnup output from base f71.
                history = common.get_history_from_f71(
                    self.obiwan, run_f71, self.hi_case
                )

                # Fill the template.
                filled_text = common.expand_template(
                    template_text,
                    {
                        "history": history,
                        "model": {"work_dir": self.work_dir, "name": self.name},
                        "build": build,
                        "generate": generate,
                    },
                )

                # Write the check input file.
                check_input.parent.mkdir(parents=True, exist_ok=True)
                common.logger.info(
                    f"Writing input file={input} for LowOrderConsistency check"
                )
                with open(check_input, "w") as f:
                    f.write(filled_text)

        # Run all the check inputs.
        run.makefile(
            model={"scalerte": self.scalerte, "work_dir": self.check_dir},
            dry_run=not do_run,
            nprocs=self.nprocs,
        )

        return f71_list

    def __write_ii_json(self, do_run, f71_list):
        """Run obiwan to write the ii.json data to disk."""

        # Conver the f71 to ii.json and extract the relevant information into memory.
        ii_json_list = list()
        for hi_f71, lo_f71 in f71_list:
            hi_ii_json = hi_f71.with_suffix(".ii.json")
            lo_ii_json = lo_f71.with_suffix(".ii.json")
            if do_run:
                common.run_command(
                    f"{self.obiwan} view -format=ii.json {hi_f71} -cases='[{self.hi_case}]' >{hi_ii_json}"
                )
                common.run_command(
                    f"{self.obiwan} view -format=ii.json {lo_f71} -cases='[{self.lo_case}]' >{lo_ii_json}"
                )
            ii_json_list.append((hi_ii_json, lo_ii_json))

        return ii_json_list

    def __load_ii_json(self, ii_json_list):
        """Load the ii.json data that exists on disk for HIGH and LOWER fidelity into memory."""

        # Conver the f71 to ii.json and extract the relevant information into memory.
        self.hi_list = list()
        self.lo_list = list()
        for hi_ii_json, lo_ii_json in ii_json_list:
            # Load the json data into HIGH fidelity and LOWER fidelity data structures.
            with open(hi_ii_json, "r") as f:
                jt = json.load(f)
                case = jt["responses"][f"case({self.hi_case})"]
                hi = np.array(case["amount"])
                hi_vector = case["nuclideVectorHash"]
                self.hi_list.append(hi)
                self.names = jt["definitions"]["nuclideVectors"][hi_vector]
                self.time_list = case["time"]

            with open(lo_ii_json, "r") as f:
                jo = json.load(f)
                case = jo["responses"][f"case({self.lo_case})"]
                lo = np.array(case["amount"])
                lo_time = case["time"]
                if not np.array_equal(lo_time, self.time_list):
                    raise ValueError(
                        f"HIGH fidelity list of times={self.time_list} is inconsistent with LOWER fidelity list of times {lo_time}"
                    )
                lo_vector = case["nuclideVectorHash"]
                if not lo_vector == hi_vector:
                    raise ValueError(
                        f"HIGH fidelity nuclide vector hash {hi_vector} is not the same as LOWER fidelity vector hash {lo_vector}, meaning the two nuclide sets are somehow inconsistent, which should not be possible."
                    )
                self.lo_list.append(lo)

    def run(self, archive):
        """Run a consistent set of LOWER fidelity calculations which also produce an
        f71--typically ORIGAMI."""

        # TODO: Allow input to change this or other smart way to determine if the data
        # does not need to be regenerated. Here, this is just for development iterations
        # to disable long SCALE runs while trying to debug checking.
        do_run = os.environ.get("SCALE_OLM_DO_RUN", "True") in ["True"]
        if not do_run:
            common.logger.warning(
                "Runs suppressed by environment variable SCALE_OLM_DO_RUN!"
            )

        # Set the case identifiers for the high and low problems.
        self.hi_case = -2
        self.lo_case = 1

        try:
            f71_list = self.__run_lo_fidelity(do_run)
            ii_json_list = self.__write_ii_json(do_run, f71_list)
            self.__load_ii_json(ii_json_list)
            self.run_success = True

        except ValueError as ve:
            self.run_success = False
            common.logger.error(str(ve))

        return self.info()


class Continuity:
    @staticmethod
    def describe_params():
        return {
            "c": "continuity order (0 or 1)",
            "eps": "relative epsilon for continuity check",
        }

    @staticmethod
    def default_params():
        c = Continuity()
        return {"c": c.c, "eps": c.eps}

    def __init__(self, model=None, c=0, eps=1e-3):
        self.c = c
        self.eps = eps

    def info(self):
        return CheckInfo()

    def run(self, archive):
        return self.info()
