import numpy as np
from tqdm import tqdm, tqdm_notebook
import scale.olm.common as common
import scale.olm.run as run
import json
from pathlib import Path
import copy


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
            name = s[".type"]
            if name.find(":") == -1:
                name = "scale.olm.check:" + name
            s[".type"] = name
            s["model"] = model

            common.logger.info(
                "Checking options for check={}, sequence={}".format(name, i)
            )
            i += 1

            # Initialize the class.
            this_class = common.fn_redirect(s)
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
    ):
        self.template = template
        self.nprocs = nprocs
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

        self.ahist = np.array(self.origami_list)
        self.rhist = np.array(self.origami_list)
        self.triton = np.array(self.triton_list)
        self.origami = np.array(self.origami_list)
        for k in range(len(self.origami_list)):
            for j in range(len(self.origami_list[k])):
                osum = self.origami_list[k][j].sum()
                tsum = self.triton_list[k][j].sum()
                oden = self.origami_list[k][j] / osum
                tden = self.triton_list[k][j] / tsum
                self.origami[k, j, :] = oden
                self.triton[k, j, :] = tden
                self.ahist[k, j, :] = np.absolute(oden - tden)
                self.rhist[k, j, :] = np.absolute(
                    (oden + self.eps0) / (tden + self.eps0) - 1.0
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

    def run(self, archive):
        try:
            # Load the template file.
            with open(self.config_dir / self.template, "r") as f:
                template_text = f.read()

            # Load the build data.
            build_json = self.work_dir / "build.json"
            with open(build_json, "r") as f:
                build_d = json.load(f)

            # Load the generate data.
            generate_json = self.work_dir / "generate.json"
            with open(generate_json, "r") as f:
                generate_d = json.load(f)

            # For each permutation.
            triton_case = -2
            origami_case = 1
            f71_list = list()
            for j in range(len(build_d["perms"])):
                # Extract the fuel power / burnup output from base f71.
                build = build_d["perms"][j]
                f71 = self.work_dir / build["input"]
                f71 = f71.with_suffix(".f71")
                history = common.get_history_from_f71(self.obiwan, f71, triton_case)

                # Fill the template.
                filled_text = common.expand_template(
                    template_text,
                    {
                        "history": history,
                        "model": {"work_dir": self.work_dir, "name": self.name},
                        "build": build,
                        "generate": generate_d["perms"][j],
                    },
                )

                # Write the input file.
                input = self.check_dir / build["input"]
                input.parent.mkdir(parents=True, exist_ok=True)
                common.logger.info(
                    f"Writing input file={input} for LowOrderConsistency check"
                )

                with open(input, "w") as f:
                    f.write(filled_text)

                # Save TRITON and ORIGAMI f71 in a list.
                f71_list.append((f71, input.with_suffix(".f71")))

            run.makefile(
                {"scalerte": self.scalerte, "work_dir": self.check_dir}, self.nprocs
            )

            # Save as ii.json.
            for t, o in f71_list:
                it = t.with_suffix(".ii.json")
                io = o.with_suffix(".ii.json")
                common.run_command(
                    f"{self.obiwan} view -format=ii.json {t} -cases='[{triton_case}]' >{it}"
                )
                common.run_command(f"{self.obiwan} view -format=ii.json {o} >{io}")

                # Load the json data into TRITON and ORIGAMI data structures.
                self.triton_list = list()
                with open(it, "r") as f:
                    jt = json.load(f)
                    case = jt["responses"][f"case({triton_case})"]
                    triton = np.array(case["amount"])
                    self.triton_list.append(triton)
                    self.names = jt["definitions"]["nuclideVectors"][
                        case["nuclideVectorHash"]
                    ]

                self.origami_list = list()
                with open(io, "r") as f:
                    jo = json.load(f)
                    origami = np.array(
                        jo["responses"][f"case({origami_case})"]["amount"]
                    )
                    self.origami_list.append(origami)

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
