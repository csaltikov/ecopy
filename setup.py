#!/usr/bin/env python

import logging
import sys
import pprint
from setuptools import setup, find_packages, Extension
# from setuptools.extension import Extension
from Cython.Build import cythonize

# Set up the logging environment
logging.basicConfig()
log = logging.getLogger()

# Handle the -W all flag
if 'all' in sys.warnoptions:
    log.level = logging.DEBUG

# Parse the verison from the ecopy module
with open('ecopy/__init__.py') as f:
    for line in f:
        if line.find('__version__') >= 0:
            version = line.split('=')[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

with open('VERSION.txt', 'w') as f:
    f.write(version)

# Use README.rst as the long description
with open('README.rst') as f:
    readme = f.read()

# Extension options
include_dirs = []
try:
    import numpy
    include_dirs.append(numpy.get_include())
except ImportError:
    log.critical('Numpy and its headers are required to run setup(). Exiting')
    sys.exit(1)

opts = dict(
    include_dirs=include_dirs,
)
log.debug('opts:\n%s', pprint.pformat(opts))

# Build extension modules 
ext_modules = cythonize([
    Extension(
        'ecopy.regression.isoFunc',
        ['ecopy/regression/isoFunc.pyx'], **opts),
])

# Dependencies
install_requires = [
    'cython',
    'numpy',
    'scipy',
    'matplotlib',
    'pandas',
    'patsy'
]

setup_args = dict(
    name='ecopy',
    version=version,
    description='EcoPy: Ecological Data Analysis in Python',
    long_description=readme,
    # Original author's information
    url='https://github.com/Auerilas/ecopy',
    author='Nathan Lemoine',
    author_email='lemoine.nathan@gmail.com',
    project_urls={
        "Original Repository": "https://github.com/Auerilas/ecopy",
        "Forked Repository": "https://github.com/csaltikov/ecopy",
    },
    maintainer="Chad Saltikov",
    maintainer_email="saltikov@ucsc.edu",
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],
    keywords=['ordination', 'ecology', 'multivariate data analysis'],
    ext_modules=ext_modules,
    install_requires=install_requires,
    packages=find_packages(),
    setup_requires=['cython'],  # Add this line
)

setup(**setup_args)
