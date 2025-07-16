# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib
import inspect
import os
import sys

sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "checkedframe"
copyright = "2025, Cangyuan Li"
author = "Cangyuan Li"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx_copybutton",
    "sphinx_multiversion",
    "sphinx_tabs.tabs",
    "myst_parser",
]


templates_path = ["_templates"]
html_sidebars = {"**": ["search-field.html", "sidebar-nav-bs.html"]}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


def linkcode_resolve(domain, info):
    commit = "main"
    code_url = f"https://github.com/CangyuanLi/checkedframe/blob/{commit}"

    assert domain == "py", "expected only Python objects"

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    file = os.path.relpath(file, os.path.abspath(".."))
    if not file.startswith("src/checkedframe"):
        # e.g. object is a typing.NewType
        return None
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"{code_url}/{file}#L{start}-L{end}"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "show_toc_level": 3,
    "collapse_navigation": False,
    # "switcher": {
    #     "json_url": "https://github.com/CangyuanLi/checkedframe/_static/switcher.json",
    #     "version_match": "latest",
    # },
}
html_static_path = ["_static"]
html_show_sourcelink = False
