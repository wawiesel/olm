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
import tomli
from pathlib import Path

with open(Path(__file__).parent.parent.parent / "pyproject.toml", "rb") as f:
    toml = tomli.load(f)

project = toml["project"]["name"]
copyright = "2023, UT-Batelle, LLC"
author = "W. Wieselquist, S. Skutnik, S. Hart, B. Hiscox, G. Ilas"
release = toml["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add additional style.
sys.path.append(os.path.abspath("_ext"))

extensions = [
    "sphinx.ext.autodoc",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "notfound.extension",
    "sphinx_click.ext",
    "click_extra.sphinx",
    "myst_parser",
    "sphinx_term.termynal",
    "scale_highlighting",
]

#
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
html_static_path = ["_static"]
html_css_files = ["css/termynal.css"]
# html_js_files = ["js/custom.js"]
html_theme_path = [
    "_themes",
]
pygments_style = "default"

master_doc = "index"

html_sidebars = {"**": ["globaltoc.html", "searchbox.html", "relations.html"]}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "sphinxsetup": """verbatimwithframe=true, VerbatimColor={gray}{0.95}""",
    "preamble": r"""
\usepackage{tgcursor}
\catcode8=9 % ignore backspace
""",
}

from sphinx.highlighting import PygmentsBridge
from pygments.formatters.latex import LatexFormatter


class CustomLatexFormatter(LatexFormatter):
    def __init__(self, **options):
        super(CustomLatexFormatter, self).__init__(**options)
        self.verboptions = r"formatcom=\footnotesize"


PygmentsBridge.latex_formatter = CustomLatexFormatter
