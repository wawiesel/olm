# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# For autodoc.
import scale.olm

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "OLM"
copyright = "2023, UT-Batelle, LLC"
author = "W. Wieselquist, S. Skutnik, S. Hart, B. Hiscox, G. Ilas"
release = "0.5.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "myst_nb",
]
# # Suppress sphinx doctest when some deps are not installed.
# doctest_global_setup = '''
# try:
#     import pandas as pd
# except ImportError:
#     pd = None
# '''
plot_rcparams = {"savefig.bbox": "tight"}
plot_apply_rcparams = True  # if context option is used

# Concatenate the class and __init__ docstring.
autoclass_content = "both"
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]
html_theme_path = [
    "_themes",
]

master_doc = "index"

html_sidebars = {"**": ["globaltoc.html", "searchbox.html", "relations.html"]}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "sphinxsetup": "verbatimwithframe=false, VerbatimColor={gray}{0.95}",
    "preamble": r"""\usepackage{tgcursor}""",
}
from sphinx.highlighting import PygmentsBridge
from pygments.formatters.latex import LatexFormatter


class CustomLatexFormatter(LatexFormatter):
    def __init__(self, **options):
        super(CustomLatexFormatter, self).__init__(**options)
        self.verboptions = r"formatcom=\footnotesize"


PygmentsBridge.latex_formatter = CustomLatexFormatter
