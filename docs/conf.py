import os
import sys


# add path for getting autodoc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

project = 'Green MBTools'
html_title = 'Green MBTools'

copyright = '2023, Green-Phys'
author = 'Chia-Nan Yeh, Gaurav Harsha, Sergei Iskakov, Vibin Abraham, Pavel Pokhilko'
release = '0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',       # For Google/NumPy-style docstrings
    'sphinx.ext.viewcode',       # Adds source code links
    'myst_parser',               # for markdown
    'sphinx_autodoc_typehints',  # Better type hint rendering
]

exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'pydata_sphinx_theme'
html_theme = 'furo'
html_logo = "rosetta.png"
html_favicon = "rosetta.png"
html_static_path = ['_static']
html_css_files = ['custom.css']
