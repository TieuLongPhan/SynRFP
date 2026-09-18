import os
import sys
from importlib.metadata import version as _get_version, PackageNotFoundError

# -- Path setup --------------------------------------------------------------
# Add project root to sys.path to import the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "synrfp"
author = "Tieu-Long Phan"

try:
    release = _get_version("synrfp")
except PackageNotFoundError:
    release = "0.0.0"
# Use only major.minor for short version
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["refs.bib"]
templates_path = ["_templates"]
exclude_patterns = []
autosectionlabel_prefix_document = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
