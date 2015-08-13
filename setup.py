#!/usr/bin/env python

import json
import numpy
from distutils.core import setup, Extension 
from Cython.Distutils import build_ext

import os, re

# Find gnuwin32 for windows compilation
gnuwin32lib = []
gnuwin32inc = []
if 'PATH' in os.environ:
	path = os.environ['PATH']
	gnuwin32lib = [p for p in path.split(';') if re.search('gnuwin32.*lib', p)]
	gnuwin32inc = [p for p in path.split(';') if re.search('gnuwin32.*include', p)]

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
        libraries=['gsl', 'gslcblas'],
        library_dirs=['/usr/local/lib', '~/.local/lib'] + gnuwin32lib,
        include_dirs=[numpy.get_include(), '~/.local/lib/include'] + gnuwin32inc),
    Extension(
        name='c_hydata', 
        sources=[
            'hydrodiy/hydata/c_hydata.pyx', 
            'hydrodiy/hydata/c_lindetect.c',
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


