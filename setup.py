"""
Setup module for Classranker
"""

from setuptools import setup, find_packages

import classranker

extras = {
    "classranker": [
        "torch",
        "numpy"
    ]
}

with open("README.md") as f:
    readme = f.read()

# Add command line option
setup(
    name="classranker",
    description="Code for training and running the model for the class ranking functionality of RELION",
    version=classranker.__version__,
    long_description=readme,
    packages=find_packages(),
    entry_points={
        'console_scripts': ['classranker=classranker:main']
    },
    extras_require=extras,
)

# Download, install and load model
classranker.install_and_load_model("v1.0")
