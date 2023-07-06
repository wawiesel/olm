import numpy as np
from tqdm import tqdm, tqdm_notebook
import scale.olm.common as common
import json


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

            common.logger.info(
                "Checking options for check={}, sequence={}".format(name, i)
            )
            i += 1

            # Initialize the class.
            this_class = common.fn_redirect(s)
            run_list.append(this_class)

        # Read the archive.
        archive = common.Archive(model["archive_file"])

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
        return {"eps0": c.eps0, "epsa": c.epsa, "epsr": c.epsr}

    def __init__(self, eps0=1e-20, epsa=1e-1, epsr=1e-1):
        self.eps0 = eps0
        self.epsa = epsa
        self.epsr = epsr

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

        info.test_pass = info.q1 > 0 and info.q2 > 0  # Fix this to real conditions

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

    def __init__(self, c=0, eps=1e-3):
        self.c = c
        self.eps = eps

    def info(self):
        return CheckInfo()

    def run(self, archive):
        return self.info()
