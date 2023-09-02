"""
The scale.olm.core module contains classes that have a core capability that could be
used by any component of OLM and even on their own outside of OLM. Maybe one day
they will become their own package as scale-core instead of scale-olm-core.

There should only be classes in the core, no free functions. 

Each core class is tested in the testing/core_ClassName_test.py file.

Each class in the core has a jupyter notebook, notebooks/ClassName_demo.ipynb.

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
from imohash import hashfile


class ScaleOutFileInfo:
    """
    Extracts and stores basic information from a SCALE output file.

    Args:
    output_file (str): The path to the SCALE output file.

    Attributes:
    output_file (str): The path to the SCALE output file.
    runtime (str): The runtime extracted from the output file.
    sequence (str): The sequence extracted from the output file.
    version (str): The version of the SCALE output file.
    product (str): The product name mapped from the sequence information.
    """

    @staticmethod
    def _map_sequence_to_product(sequence) -> str:
        """
        Maps the sequence information to a product name.

        Args:
        sequence (str): sequence name.

        Returns:
        str: corresponding product name.
        """
        products = {"t-depl-1d": "TRITON", "t-depl": "TRITON"}
        return products.get(sequence, "UNKNOWN")

    def __init__(self, output_file: str):
        """
        Initializes the ScaleOutFileInfo instance.

        Args:
            output_file (str): The path to the SCALE output file.
        """
        self.output_file = Path(output_file)
        self.runtime = None
        self.sequence = None
        self.version = None
        self.product = None

        self._parse_info()
        self.product = _map_sequence_to_product(self.sequence)

    def _parse_info(self) -> None:
        """
        Extracts the runtime and sequence information from the output file.

        Args:
        output_file (str): The path to the SCALE output file.
        """
        with open(self.output_file, "r") as f:
            for line in f.readlines():
                runtime_match = re.search(
                    r"([^\s]+) finished\. used (\d+\.\d+) seconds\.", line
                )
                if runtime_match:
                    self.sequence = runtime_match.group(1)
                    self.runtime = runtime_match.group(2)


class ScaleRte:
    """
    A basic wrapper class around SCALE Runtime Environment (scalerte).

    Example Usage:

        runner = ScaleRte('/my/install/bin/scalerte')
        runner.version
        runner.run('/path/to/scale.inp')

    Attributes:
    - `scalerte_path`: The path to the scalerte executable.
    - `version`: The version of the SCALE Runtime Environment.
    - `data_dir`: The path to the SCALE data directory.
    - `data_size`: The size of the SCALE data directory.
    """

    def __init__(self, scalerte_path):
        """
        Initialize and check the runtime for various key quantities.

        Args:
        scalerte_path (str): the path to the runtime, e.g. /my/install/bin/scalerte
        """
        self.scalerte_path = Path(scalerte_path)
        if not self.scalerte_path.exists():
            raise ValueError(
                f"Path to SCALE Runtime Environment, {self.scalerte_path} does not exist!"
            )
        self.version = self._get_version(self.scalerte_path)
        self.data_dir = self._get_data_dir(self.scalerte_path)
        self.data_size = self._get_data_size(self.data_dir)

    @staticmethod
    def _get_version(scalerte_path):
        """
        Internal method to get the SCALE version.

        Args:
        scalerte_path (str): the path to the runtime, e.g. /my/install/bin/scalerte

        Returns:
        str: the version string as MAJOR.MINOR.PATCH
        """
        version = subprocess.run(
            [scalerte_path, "-V"],
            capture_output=True,
            text=True,
        ).stdout.split(" ")[2]
        return version

    @staticmethod
    def _get_data_dir(scalerte_path):
        """
        Internal method to get the SCALE data directory. If the environmental variable DATA
        or SCALE_DATA_DIR is set then prefer that. Otherwise use the installation convention of
        scalerte being installed to /x/bin and data at /x/data.

        NOTE: This does not verify if the data directory exists or not!

        Args:
        scalerte_path (str): the path to the runtime, e.g. /my/install/bin/scalerte

        Returns:
        pathlib.Path: the data directory
        """
        if "SCALE_DATA_DIR" in os.environ:
            data_dir = os.environ["SCALE_DATA_DIR"]
        elif "DATA" in os.environ:
            data_dir = os.environ["DATA"]
        else:
            data_dir = Path(scalerte_path).parent.parent / "data"

        return Path(data_dir)

    @staticmethod
    def _rerun_cache_name(output_file):
        """
        Internal method to return the cache name corresponding to an output file.

        Args:
        output_file (str): The output file name

        Returns:
        str: The cache name
        """
        return Path(output_file).with_suffix(".run.json")

    @staticmethod
    def _get_data_size(data_dir):
        """
        Internal method to calculate the total file size of the contents of the data directory.

        Args:
        data_dir (str): The path to the SCALE data directory.

        Returns:
        int: size in bytes of the data directory or -1 if the path does not exist.
        """
        if not Path(data_dir).exists():
            return -1

        return sum(
            element.stat().st_size
            for element in pathlib.Path(data_dir).glob("**/*")
            if element.is_file()
        )

    @staticmethod
    def _determine_if_rerun(output_file, input_file, data_size, version):
        """
        Internal method to check that input_file is the same (through hash), as well as
        data_size and version before committing to rerun SCALE. Based on the inernal
        convention that this ScaleRte wrapper drops a file with suffix Scale

        Args:
        output_file (pathlib.Path): The path to the output file.
        input_file (pathlib.Path): The path to the input file.
        data_size (int): The size of the data directory.
        version (str): The version of the SCALE Runtime Environment.

        Returns:
        bool: True if the input file needs to be rerun, False otherwise.
        """
        if output_file.exists():
            sr = self._rerun_cache_name(output_file)
            if sr.exists():
                input_file_hash = hashfile(input_file)
                with open(sr, "r") as f:
                    old = json.load(f)
                    logger.debug(
                        old=old,
                        new={
                            "input_file_hash": input_file_hash,
                            "data_size": data_size,
                            "version": version,
                        },
                    )
                    if (
                        old["input_file_hash"] == input_file_hash
                        and old["data_size"] == data_size
                        and old["version"] == version
                    ):
                        return False
        return True

    def _scrape_errors_from_message_file(message_file):
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

    def run(self, input_file, args=""):
        """
        Run an input file through SCALE, verifying first that the file was not already successfully run.

        Args:
        input_file: path to the input file
        args: arguments

        Raises:
        ValueError: If the input file does not exist.

        Returns:
        dict: of various
        """
        if not Path(input_file).exists():
            raise ValueError(f"SCALE input file, {input_file} does not exist!")

        output_file = Path(input_file).with_suffix(".out")
        rerun = self.determine_if_rerun(
            output_file, input_file, self.data_size, self.version
        )
        command_line = f"{self.scalerte_path} {args} {input_file}"
        message_file = output_file.with_suffix(".msg")

        if rerun:
            if not self.data_dir.exists():
                raise ValueError(
                    f"Path to SCALE Data was not found! Either 1) set the environment variable DATA or 2) symlink the data directory to {data_dir}."
                )
            result = subprocess.run(
                command_line,
                shell=True,
                capture_output=True,
                text=True,
            )
            # If run was not successful, move output to .FAILED
            print(result)
            success = False
            if not success:
                output_file0 = output_file
                output_file += ".FAILED"
                shutil.move(output_file0, output_file)
                errors = self._scrape_errors_from_message_file(message_file)
            data = {
                "rerun": rerun,
                "success": success,
                "errors": errors,
                "command_line": command_line,
                "input_file": input_file,
                "output_file": output_file,
                "message_file": message_file,
                "data_size": self.data_size,
                "data_dir": self.data_dir,
                "scalerte_path": self.scalerte_path,
                "input_file_hash": hashfile(input_file),
                "version": self.version,
            }
            sr = self._rerun_cache_name(output_file)
            with open(sr, "w") as f:
                json.dump(data, f)

        return data
