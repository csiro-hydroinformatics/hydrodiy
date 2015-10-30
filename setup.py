#!/usr/bin/env python

import json
import numpy
from distutils.core import setup, Extension 
from Cython.Distutils import build_ext

import os, re

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
        name='c_hymod_models_utils', 
        sources=[
            'hydrodiy/hymod/models/c_hymod_models_utils.pyx', 
            'hydrodiy/hymod/models/c_utils.c',
            'hydrodiy/hymod/models/c_uh.c',
            'hydrodiy/hymod/models/c_dummy.c'
        ],
        include_dirs=[numpy.get_include()]),

    Extension(
        name='c_hymod_models_gr4j', 
        sources=[
            'hydrodiy/hymod/models/c_hymod_models_gr4j.pyx', 
            'hydrodiy/hymod/models/c_utils.c',
            'hydrodiy/hymod/models/c_uh.c',
            'hydrodiy/hymod/models/c_gr4j.c'
        ],
        include_dirs=[numpy.get_include()]),

     Extension(
        name='c_hymod_models_gr2m', 
        sources=[
            'hydrodiy/hymod/models/c_hymod_models_gr2m.pyx', 
            'hydrodiy/hymod/models/c_utils.c',
            'hydrodiy/hymod/models/c_gr2m.c'
        ],
        include_dirs=[numpy.get_include()]),

     Extension(
        name='c_hymod_models_lagroute', 
        sources=[
            'hydrodiy/hymod/models/c_hymod_models_lagroute.pyx', 
            'hydrodiy/hymod/models/c_utils.c',
            'hydrodiy/hymod/models/c_uh.c',
            'hydrodiy/hymod/models/c_lagroute.c'
        ],
        include_dirs=[numpy.get_include()]),

    Extension(
        name='c_hystat', 
        sources=[
            'hydrodiy/hystat/c_hystat.pyx', 
            'hydrodiy/hystat/c_ar1.c'
        ],
        include_dirs=[numpy.get_include(), '~/.local/lib/include']),

    Extension(
        name='c_hydata', 
        sources=[
            'hydrodiy/hydata/c_hydata.pyx', 
            'hydrodiy/hydata/c_baseflow.c'
        ],
        include_dirs=[numpy.get_include()])
]
 
# Package config
js = 'package_config.json'
cfg = json.load(open(js, 'r'))
cfg['packages'] = [str(s) for s in cfg['packages']]

# Setup config
setup(
    name = cfg['name'],
    version = cfg['version'],
    description = cfg['description'],
    author = cfg['author'],
    author_email = cfg['author_email'],
    url = cfg['url'],
    package_dir = cfg['package_dir'],
    package_data = cfg['package_data'],
    packages = cfg['packages'],
    requires=cfg['requires'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)


