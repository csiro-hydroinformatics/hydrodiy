#!/usr/bin/env python

import os
import subprocess
import numpy

from setuptools import setup, Extension, find_packages

import distutils.cmd
import distutils.log

from Cython.Distutils import build_ext

import versioneer

def read(fname):
    return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)).read()


# Define Cython C extensions
if os.getenv('HYDRODIY_NO_BUILD') == '1':
    # Not extension
    ext_modules = []
else:
    ext_modules = [
        Extension(
            name='c_hydrodiy_data',
            sources=[
                'hydrodiy/data/c_hydrodiy_data.pyx',
                'hydrodiy/data/c_dateutils.c',
                'hydrodiy/data/c_qualitycontrol.c',
                'hydrodiy/data/c_dutils.c',
                'hydrodiy/data/c_var2h.c',
                'hydrodiy/data/c_baseflow.c'
            ],
            include_dirs=[numpy.get_include()]),

        Extension(
            name='c_hydrodiy_stat',
            sources=[
                'hydrodiy/stat/c_hydrodiy_stat.pyx',
                'hydrodiy/stat/c_crps.c',
                'hydrodiy/stat/c_dscore.c',
                'hydrodiy/stat/c_olsleverage.c',
                'hydrodiy/stat/c_armodels.c',
                'hydrodiy/stat/ADinf.c',
                'hydrodiy/stat/AnDarl.c',
                'hydrodiy/stat/c_andersondarling.c'
            ],
            include_dirs=[numpy.get_include()]),
        Extension(
            name='c_hydrodiy_gis',
            sources=[
                'hydrodiy/gis/c_hydrodiy_gis.pyx',
                'hydrodiy/gis/c_grid.c',
                'hydrodiy/gis/c_catchment.c',
                'hydrodiy/gis/c_points_inside_polygon.c'
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
    packages = find_packages(),
    package_data = {
        "hydrodiy": ["gis/data/*.gz", \
            "gis/data/*.json", \
            "gis/data/*.bil", \
            "gis/data/*.zip", \
            "gis/data/*.hdr", \
            "gis/data/*.dbf", \
            "gis/data/*.shp", \
            "gis/data/*.shx", \
            "gis/data/*.prj", \
            "data/data/*.csv", \
            "data/data/*.zip", \
            "stat/data/*.zip",\
            "stat/data/*.csv"],
    },
    requires= [
        "pandas (>=0.14.0)",
        "scipy (>=0.14.0)",
        "Cython (>=0.20.1)",
        "numpy (>=1.8)",
        "matplotlibb (>=1.3.1)",
        "requests"
    ],
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    test_suite='nose.collector',
    tests_require=[
        'nose',
        'pyproj'
    ],
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: BSD License'
    ]
)


