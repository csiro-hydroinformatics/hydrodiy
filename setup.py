#!/usr/bin/env python

import os, re

import numpy

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from Cython.Distutils import build_ext

import versioneer

def read(fname):
    return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)).read()

# Cython C extensions
ext_modules = [
    Extension(
        name='c_hydrodiy_data',
        sources=[
            'hydrodiy/data/c_hydrodiy_data.pyx',
            'hydrodiy/data/c_dateutils.c',
            'hydrodiy/data/c_dutils.c'
        ],
        include_dirs=[numpy.get_include()]),

    Extension(
        name='c_hydrodiy_stat',
        sources=[
            'hydrodiy/stat/c_hydrodiy_stat.pyx',
            'hydrodiy/stat/c_crps.c',
            'hydrodiy/stat/c_olsleverage.c',
            'hydrodiy/stat/c_ar1.c'
        ],
        include_dirs=[numpy.get_include()]),
    Extension(
        name='c_hydrodiy_gis',
        sources=[
            'hydrodiy/gis/c_hydrodiy_gis.pyx',
            'hydrodiy/gis/c_grid.c',
            'hydrodiy/gis/c_catchment.c'
        ],
        include_dirs=[numpy.get_include()])
]

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

# Setup config
setup(
    name = 'hydrodiy',
    version= versioneer.get_version(),
    description = 'Python tools to support hydrological modelling and data analysis',
    author = 'Julien Lerat',
    author_email = 'julien.lerat@gmail.com',
    url = 'https://bitbucket.org/jlerat/hydrodiy',
    package_data = {
        "hydrodiy": ["gis/data/*.gz"],
    },
    requires= [
        "pandas (>=0.14.0)",
        "scipy (>=0.14.0)",
        "Cython (>=0.20.1)",
        "numpy (>=1.8)",
        "cycler (>=0.10)"
    ],
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    test_suite='nose.collector',
    tests_require=['nose'],
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)'
    ]
)


