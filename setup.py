# setup.py needed in order to have a pip editable install
from setuptools import setup

setup(use_scm_version=True, include_package_data=True)  # config is in setup.cfg
# with `include_package_data`, setuptools_scm will include all Git-tracked files
