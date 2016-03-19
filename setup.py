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
        name='c_hymod',
        sources=[
            'hydrodiy/hymod/c_hymod.pyx',
            'hydrodiy/hymod/c_crps.c'
        ],
        include_dirs=[numpy.get_include()]),

    Extension(
        name='c_hystat',
        sources=[
            'hydrodiy/hystat/c_hystat.pyx',
            'hydrodiy/hystat/c_ar1.c'
        ],
        include_dirs=[numpy.get_include()]),

    Extension(
        name='c_hydata',
        sources=[
            'hydrodiy/hydata/c_hydata.pyx',
            'hydrodiy/hydata/c_baseflow.c'
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
    packages = [
        "hymod",
        "hygis",
        "hyplot",
        "hydata",
        "hyio",
        "hystat"
    ],
    package_dir = {
        "hymod":"hydrodiy/hymod",
        "hygis":"hydrodiy/hygis",
        "hydata":"hydrodiy/hydata",
        "hyio":"hydrodiy/hyio",
        "hyplot":"hydrodiy/hyplot",
        "hystat":"hydrodiy/hystat"
    },
    package_data = {
        "hygis": ["data/*.gz"],
        "hyio": ["*.py.gz"],
        "hyplot": ["*.json"]
    },
    requires= [
        "pandas (>=0.14.0)",
        "scipy (>=0.14.0)",
        "Cython (>=0.20.1)",
        "numpy (>=1.8)"
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
        'License :: OSI Approved :: MIT License'
    ]
)


