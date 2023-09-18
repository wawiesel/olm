from testbook import testbook
from scale.olm.core import TempDir, ScaleRunner
from pathlib import Path
import os


def notebook_file(filename):
    p = Path(__file__)
    p = p.parent.parent / "notebooks" / filename
    return p


@testbook(notebook_file("core_ScaleRunner.ipynb"), execute=False)
def test_core_ScaleRunner(tb):
    """Test that we can execute the notebook.

    Only run these notebook if the environment variable SCALE is set.

    """
    if "SCALE" in os.environ:
        tb.execute()


@testbook(notebook_file("core_BurnupHistory.ipynb"), execute=True)
def test_core_BurnupHistory(tb):
    """Test that we can execute the notebook."""


@testbook(notebook_file("core_TemplateManager.ipynb"), execute=True)
def test_core_TemplateManager(tb):
    """Test that we can execute the notebook."""
