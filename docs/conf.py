"""Sphinx configuration for imagetransforms."""

import sys
from importlib.metadata import version as _get_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Project information -----------------------------------------------------

project = 'imagetransforms'
author = 'David G. Grier'
copyright = '2026, David G. Grier'
release = _get_version('imagetransforms')

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False

autodoc_member_order = 'bysource'
autodoc_default_options = {
    'undoc-members': False,
    'show-inheritance': True,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
}

exclude_patterns = ['_build']

# -- HTML output -------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_title = 'imagetransforms'
html_static_path = ['_static']
html_css_files = ['nyu.css']

html_theme_options = {
    'github_url': 'https://github.com/davidgrier/imagetransforms',
    'show_toc_level': 2,
    'navigation_with_keys': True,
    'show_nav_level': 2,
    'navbar_end': ['navbar-icon-links', 'theme-switcher'],
    'footer_start': ['copyright'],
    'footer_end': ['sphinx-version'],
}

html_sidebars = {'**': []}
