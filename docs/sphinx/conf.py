"""Sphinx configuration for stable_ginv API docs (built to GitHub Pages)."""
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "stable-ginv"
author = "MathiasKES"
release = "0.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # NumPy/Google-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = True
autodoc_typehints = "description"

# autodoc must not fail the build if heavy optional imports are unavailable in
# the docs runner.
autodoc_mock_imports = ["torch", "torchvision", "scipy", "skimage", "seaborn",
                        "matplotlib", "numpy", "imageio", "PIL", "tqdm", "pandas"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Files in _extra are copied verbatim to the site root, so thesis.pdf is served
# at <site>/thesis.pdf rather than under _static/.
html_extra_path = ["_extra"]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

# stable_ginv/viz/plot_rank_reconstruction.py reuses its module docstring verbatim
# as argparse --help text (description=__doc__), so the literal "|grad|" in it must
# stay. Define the substitution here so docutils renders it literally instead of
# erroring on an undefined substitution.
rst_prolog = r".. |grad| replace:: \|grad\|"
