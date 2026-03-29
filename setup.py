#!/usr/bin/env python
"""
Minimal setup.py — retained only for Cython extension compilation.
All package metadata, dependencies, and version are defined in pyproject.toml.
"""
import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = cythonize([
    Extension(
        'ecopy.regression.isoFunc',
        ['ecopy/regression/isoFunc.pyx'],
        include_dirs=[numpy.get_include()]
    )
])

setup(ext_modules=ext_modules)
