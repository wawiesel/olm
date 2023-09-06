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
release = "0.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]

# Concatenate the class and __init__ docstring.
autoclass_content = "both"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_path = [
    "_themes",
]

master_doc = "index"

html_sidebars = {"**": ["globaltoc.html", "searchbox.html", "relations.html"]}
