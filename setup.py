#!/usr/classranker/env python

"""
Setup module for Classranker
"""

from setuptools import setup, find_packages

import classranker

setup(
    name="classranker",
    packages=find_packages(),
    entry_points={
        'console_scripts': ['classranker=classranker.command_line:main']
    },
    version=classranker.__version__,
)