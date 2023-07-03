import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
import scale.olm.common as common
import json
import sys


class CheckInfo:
    def __init__(self):
        self.test_pass = True


class GridGradient:
    @staticmethod
    def describe_params():
        return {
            "eps0": "minimum value",
            "epsa": "absolute epsilon",
            "epsr": "relative epsilon",
        }

    @staticmethod
    def default_params():
        p = type("", (), {})()
        p.eps0 = 1e-20
        p.epsa = 1e-1
        p.epsr = 1e-1
        return p

    def __init__(self, params):
        p = GridGradient.default_params()
        # Todo: somehow iterate over attributes to
        # set these things.
        self.eps0 = params.get("eps0", p.eps0)
        self.epsa = params.get("epsa", p.epsa)
        self.epsr = params.get("epsr", p.epsr)

    def run(self, archive):
        """Run the calculation and return post-processed results"""

        common.logger.info(
            "Running "
            + __name__
            + " check with params={}".format(json.dumps(self.__dict__))
        )
        self.calc(archive)

        # After calc the self.ahist, rhist, khist, and rel_axes variables are ready to
        # compute metrics.
        info = CheckInfo()
        info.eps0 = self.eps0
        info.epsa = self.epsa
        info.epsr = self.epsr
        info.wa = float(
            np.logical_and((self.ahist > self.epsa), (self.rhist > self.epsr)).sum()
        )
        info.wr = float((self.rhist > self.epsr).sum())
        info.m = float(len(self.ahist))
        info.q1 = 1.0 - info.wr / info.m
        info.q2 = 1.0 - 0.9 * info.wa / info.m - 0.1 * info.wr / info.m

        common.logger.info(
            "Completed "
            + __name__
            + " check with q1={:.2f} and q2={:.2f}".format(info.q1, info.q2)
        )

        return info

    def calc(self, archive):
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
        self.ahist, self.rhist, self.khist = GridGradient.kernel(
            self.rel_axes, self.yreshape, self.eps0
        )
        common.logger.info("Finished computing grid gradients")

    @staticmethod
    def kernel(rel_axes, yreshape, eps0):
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
        p = type("", (), {})()
        p.c = 0
        p.eps = 1e-3
        return p

    def __init__(self, params):
        p = GridGradient.default_params()
        # Todo: somehow iterate over attributes to
        # set these things.
        self.c = params["c"] or p.c
        self.eps = params["eps"] or p.eps

    def run(self, archive):
        return CheckInfo()
