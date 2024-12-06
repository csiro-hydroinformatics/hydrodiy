#!/usr/bin/env python

import os
import numpy

from setuptools import setup, Extension
from Cython.Build import cythonize

# Define Cython C extensions
if os.getenv("HYDRODIY_NO_BUILD") == "1":
    # Not extension
    extensions = []
else:
    extensions = [
        Extension(
            name="c_hydrodiy_data",
            sources=[
                "src/hydrodiy/data/c_hydrodiy_data.pyx",
                "src/hydrodiy/data/c_dateutils.c",
                "src/hydrodiy/data/c_qualitycontrol.c",
                "src/hydrodiy/data/c_dutils.c",
                "src/hydrodiy/data/c_var2h.c",
                "src/hydrodiy/data/c_baseflow.c"
            ],
            include_dirs=[numpy.get_include()]),

        Extension(
            name="c_hydrodiy_stat",
            sources=[
                "src/hydrodiy/stat/c_hydrodiy_stat.pyx",
                "src/hydrodiy/stat/c_crps.c",
                "src/hydrodiy/stat/c_dscore.c",
                "src/hydrodiy/stat/c_olsleverage.c",
                "src/hydrodiy/stat/c_armodels.c",
                "src/hydrodiy/stat/ADinf.c",
                "src/hydrodiy/stat/AnDarl.c",
                "src/hydrodiy/stat/c_andersondarling.c",
                "src/hydrodiy/stat/c_paretofront.c"
            ],
            include_dirs=[numpy.get_include()]),
        Extension(
            name="c_hydrodiy_gis",
            sources=[
                "src/hydrodiy/gis/c_hydrodiy_gis.pyx",
                "src/hydrodiy/gis/c_grid.c",
                "src/hydrodiy/gis/c_catchment.c",
                "src/hydrodiy/gis/c_points_inside_polygon.c"
            ],
            include_dirs=[numpy.get_include()])
    ]

setup(
    name = "hydrodiy",
    ext_modules = cythonize(extensions,\
                    compiler_directives={"language_level": 3, "profile": False})
)


