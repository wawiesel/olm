"""
The scale.olm.core module contains classes that have a core capability that could be
used by any component of OLM and even on their own outside of OLM. Maybe one day
they will become their own package as scale-core instead of scale-olm-core.

There should only be classes in the core, no free functions.

Each core class is tested in the testing/core_ClassName_test.py file.

Each class in the core has a jupyter notebook, notebooks/core_ClassName_demo.ipynb.

Usage:

>>> import scale.olm.core

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
import structlog
import logging
import time

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(
        int(os.environ.get("SCALE_LOG_LEVEL", logging.INFO))
    ),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)

logger = structlog.get_logger(__name__)


class TempDir:
    """
    Creates a temporary directory to write files to. Automatically
    deletes the directory when the class goes out of scope.

    Usage:

    >>> from scale.olm.core import TempDir
    >>> temp_dir = TempDir()
    >>> temp_dir.path.exists()
    True

    Attributes:
    path (pathlib.Path): path to the temporary directory
    """

    def __init__(self):
        """
        Initialize.

        Usage:

        >>> td = TempDir()
        >>> path = td.path
        >>> path.exists() and path.is_dir()
        True

        When the object goes out of scope, the directory is deleted.
        >>> td = None
        >>> path.exists()
        False

        """
        import tempfile

        self._td_obj = tempfile.TemporaryDirectory()
        self.path = Path(self._td_obj.name)

    def write_file(self, text, name):
        """
        Write the string content text to the file name in the temporary directory.

        Usage:

        >>> temp_dir = TempDir()
        >>> file = temp_dir.write_file("CONTENT","my.txt")
        >>> file.name
        'my.txt'
        >>> file.parent == temp_dir.path
        True

        Args:
        text (str): text to write
        name (str): name of the file to write to in the temp directory

        Returns:
        pathlib.Path: filename that was created
        """
        file = self.path / name
        with open(file, "w") as f:
            f.write(text)
        return file


class ThreadPoolExecutor:
    """Executes in parallel using threads.

    Args:
    max_workers (int): number of independent threads/workers to use

    Usage:

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
    max_workers (int): number of independent threads/workers to use
    progress_bar (bool): whether to enable the progress bar output
    """

    def __init__(self, max_workers=2, progress_bar=True):
        """
        Initialize.

        Usage:

        >>> tpe = ThreadPoolExecutor(5)
        >>> tpe.max_workers
        5
        >>> tpe.progress_bar
        True

        Args:
        max_workers (int): number of parallel workers to use
        progress_bar (bool): if progress bar output should be enabled--it can be useful to
            disable for tests and other non-interactive use cases

        """
        self.max_workers = max_workers
        self.progress_bar = progress_bar

    def execute(self, my_func, input_list):
        """
        Run a list of inputs through a function.

        Usage:

        >>> def my_func(input):
        ...     output="x"+input.upper()+"x"
        ...     return input,output
        >>> input_list = ["a","b"]
        >>> results = thread_pool_executor.execute(my_func,input_list)
        >>> results
        {'a': 'xAx', 'b': 'xBx'}

        Args:
        my_func (func[input]): a function that takes a single argument, the elements of
            input_list, one at a time. This function must return things: the original
            input key, and an output data packet, i.e. [str,any]
        input_list: (list[str]): a list of strings that represent the input for each run of my_func

        Returns:
        dict[str,any]: a dictionary of output data returned by the run of the function

        """
        import concurrent.futures
        from tqdm import tqdm

        # We can use a with statement to ensure threads are cleaned up promptly
        results = {}
        with tqdm(total=len(input_list), disable=not self.progress_bar) as pbar:
            logging.info("ThreadPoolExecutor", max_workers=self.max_workers)
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


class ScaleOutFileInfo:
    """
    Extracts and stores basic information from a SCALE .out main output file.

    Usage:

    >>> info = ScaleOutFileInfo(scale_out_file)
    >>> info.out_file == scale_out_file
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

    Args:
    out_file (str): The path to the SCALE output file.

    Attributes:
    out_file (str): The path to the SCALE .out file.
    sequence_list (list[dict]): Information extracted for each sequence in order
       - sequence (str)
       - runtime_seconds (float)
       - product (str)
    """

    @staticmethod
    def get_product_name(sequence) -> str:
        """
        Maps the sequence information to a product name.

        Usage:

        >>> ScaleOutFileInfo.get_product_name('t-depl-1d')
        'TRITON'

        Args:
        sequence (str): sequence name.

        Returns:
        str: corresponding product name.
        """
        products = {"t-depl-1d": "TRITON", "t-depl": "TRITON"}
        return products.get(sequence, "UNKNOWN")

    def __init__(self, out_file: str):
        """
        Initializes the ScaleOutFileInfo instance.

        Usage:

        >>> info = ScaleOutFileInfo(scale_out_file)
        >>> info.version
        '6.3.0'

        Args:
            out_file (str): The path to the SCALE output file.
        """
        self.out_file = Path(out_file)
        self.version = None

        self._parse_info()
        for sequence in self.sequence_list:
            sequence["product"] = ScaleOutFileInfo.get_product_name(
                sequence["sequence"]
            )

    def _parse_info(self) -> None:
        """
        Internal routine to parse the runtime and sequence information from the output file.
        """

        self.sequence_list = []
        with open(self.out_file, "r") as f:
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


class ScaleRte:
    """
    A basic wrapper class around SCALE Runtime Environment (scalerte).

    Example Usage:

        runner = ScaleRte('/my/install/bin/scalerte')
        runner.version
        runner.run('/path/to/scale.inp')

    Attributes:
    - scalerte_path`: The path to the scalerte executable.
    - args: Arguments to use when invoking SCALE (see set_args)
    - version: The version of the SCALE Runtime Environment.
    - data_dir: The path to the SCALE data directory.
    - data_size: The size of the SCALE data directory.
    """

    def __init__(self, scalerte_path, do_not_run=False):
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
        self.args = ""
        self.version = self._get_version(self.scalerte_path, do_not_run)
        self.data_dir = self._get_data_dir(self.scalerte_path)
        self.data_size = self._get_data_size(self.data_dir)

    def __str__(self):
        """
        Emit the string representation of the underlying data.
        """
        p = {}
        for k, v in self.__dict__.items():
            p[k] = str(v)
        return json.dumps(p, indent=4)

    @staticmethod
    def _get_version(scalerte_path, do_not_run=False):
        """
        Internal method to get the SCALE version.

        Args:
        scalerte_path (str): the path to the runtime, e.g. /my/install/bin/scalerte
        do_not_run (bool): do not run scalerte, just return '' (only used for testing)

        Returns:
        str: the version string as MAJOR.MINOR.PATCH
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

        return int(
            sum(
                element.stat().st_size
                for element in pathlib.Path(data_dir).glob("**/*")
                if element.is_file()
            )
        )

    @staticmethod
    def _hash_file(file):
        """
        Internal method to hash a file by name so it is done consistently throughout.

        Args:
        file (str): name of the file which exists

        Returns:
        str: hash in hex of the contents of the file
        """
        from imohash import hashfile

        return hashfile(file, hexdigest=True)

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
        dict: Empty if rerunning needed, what was read off disk otherwise.
        """
        if output_file.exists():
            sr = ScaleRte._rerun_cache_name(output_file)
            if sr.exists():
                input_file_hash = ScaleRte._hash_file(input_file)
                with open(sr, "r") as f:
                    old = json.load(f)
                    logger.debug(
                        __name__,
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
                        return False, old
        return True, {}

    @staticmethod
    def _scrape_errors_from_message_file(message_file):
        """
        Internal method to scrape errors from the SCALE message (.msg) file.

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
        """
        Set the arguments to use for subsequent run calls.

        Args:
        args (str): arguments as a single string
        """
        self.args = args

    def run(self, input_file):
        """
        Run an input file through SCALE, verifying first that the file was not already successfully run.

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

        output_file = Path(input_file).with_suffix(".out")
        rerun, data = self._determine_if_rerun(
            output_file, input_file, self.data_size, self.version
        )
        command_line = f"{self.scalerte_path} {self.args} {input_file}"
        message_file = output_file.with_suffix(".msg")

        start = time.time()
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
            success = result.returncode == 0
            errors = []
            if not success:
                output_file0 = str(output_file)
                output_file = output_file0 + ".FAILED"
                shutil.move(output_file0, output_file)
                errors = self._scrape_errors_from_message_file(message_file)
            data = {
                "returncode": result.returncode,
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
                "input_file_hash": str(ScaleRte._hash_file(input_file)),
                "version": self.version,
            }
            sr = ScaleRte._rerun_cache_name(output_file)
            with open(sr, "w") as f:
                json.dump(data, f)

        runtime_seconds = float(time.time() - start)
        data["runtime_seconds"] = runtime_seconds
        data["rerun"] = rerun

        logger.debug(__file__, **data)

        return str(input_file), data


# This enables succinct doctests for methods by using the existing data in extraglobs.
# However, you can only invoke doctest as `
#           python ./core.py -v
# not as python -m doctest -v ./core.py
if __name__ == "__main__":
    import doctest

    testing_temp_dir = TempDir()
    testing_rte_path = testing_temp_dir.write_file("", "scalerte")
    scale_out_file = testing_temp_dir.write_file(
        """
*                              SCALE 6.3.0                            *
t-depl finished. used 35.2481 seconds.""",
        "example.out",
    )
    doctest.testmod(
        extraglobs={
            "scale_rte": ScaleRte(testing_rte_path, do_not_run=True),
            "thread_pool_executor": ThreadPoolExecutor(
                max_workers=2, progress_bar=False
            ),
            "scale_out_file": scale_out_file,
        },
    )
