"""
Setup module for Classranker
"""

from setuptools import setup, find_packages

import classranker

# Add command line option
setup(
    name="classranker",
    packages=find_packages(),
    entry_points={
        'console_scripts': ['classranker=classranker:main']
    },
    version=classranker.__version__,
)

# Download and install model
classranker.install_and_load_model("v1.0", only_install=True)
