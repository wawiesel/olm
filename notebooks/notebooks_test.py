from testbook import testbook
from scale.olm.core import TempDir, ScaleRunner
from pathlib import Path
import os
import sys

this_dir = Path(__file__).parent.resolve()
sys.path.append(str(this_dir))


def notebook_file(filename):
    os.chdir(str(this_dir))
    return filename


@testbook(notebook_file("core_ScaleRunner.ipynb"), execute=False)
def test_core_ScaleRunner(tb):
    """Test that we can execute the notebook.

    Only run these notebook if the environment variable SCALE_DIR is set.

    """
    if "SCALE_DIR" in os.environ:
        tb.execute()


@testbook(notebook_file("core_BurnupHistory.ipynb"), execute=True)
def test_core_BurnupHistory(tb):
    """Test that we can execute the notebook."""


@testbook(notebook_file("core_TemplateManager.ipynb"), execute=True)
def test_core_TemplateManager(tb):
    """Test that we can execute the notebook."""


@testbook(notebook_file("core_CompositionManager.ipynb"), execute=True)
def test_core_CompositionManager(tb):
    """Test that we can execute the notebook."""
