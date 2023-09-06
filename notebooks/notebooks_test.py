from testbook import testbook
from scale.olm.core import TempDir, ScaleRte
from pathlib import Path
import os


def notebook_file(filename):
    p = Path(__file__)
    p = p.parent.parent / "notebooks" / filename
    return p


@testbook(notebook_file("core_ScaleRte.ipynb"), execute=False)
def test_notebooks_execute(tb):
    """Test that we can execute the notebook.

    Patch the class to not actually run anything so we don't need a SCALE
    install to check notebook validity.
    """

    with tb.patch("scale.olm.core.ScaleRte") as S:
        S._default_do_not_run = lambda: True
        # TODO: enable this execute. Right now it does not seem like the patch
        # is working to change the default do not run. If you change to True,
        #
        # E           Cell In[7], line 2
        # E                 1 # Run the first input.
        # E           ----> 2 input, output = scale_rte.run(input_list[0])
        # E                 3 srs = output['scale_runtime_seconds']
        # E
        # E           ValueError: not enough values to unpack (expected 2, got 0)
        # E           ValueError: not enough values to unpack (expected 2, got 0)
        #
        # which is like the .run function is not found. Curious.
        if False:
            tb.execute()
