#!/usr/bin/env python

import os
import subprocess
import numpy

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

import distutils.cmd
import distutils.log

from Cython.Distutils import build_ext

import versioneer

def read(fname):
    return open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)).read()

# Pylint command
currdir = os.path.abspath(os.path.dirname(__file__))

class PylintCommand(distutils.cmd.Command):
  """A custom command to run Pylint on all Python source files."""

  description = 'run Pylint on Python source files'
  user_options = [
      # The format is (long option, short option, description).
      ('pylint-rcfile=', None, os.path.join(currdir, 'pylintrc')),
  ]

  def initialize_options(self):
    """Set default values for options."""
    # Each user option must be listed here with their default value.
    self.pylint_rcfile = ''

  def finalize_options(self):
    """Post-process options."""
    if self.pylint_rcfile:
      assert os.path.exists(self.pylint_rcfile), (
          'Pylint config file %s does not exist.' % self.pylint_rcfile)

  def run(self):
    """Run command."""

    command = ['pylint', \
        '--rcfile={0}'.format(self.pylint_rcfile), \
        os.path.join(currdir, 'hydrodiy')]

    logfile = os.path.join(currdir, 'pylint.log')
    self.announce('Running command: {0} (log={1})'.format(\
                        command, logfile), \
        level=distutils.log.INFO)

    with open(logfile, 'w') as log:
        subprocess.call(command, stdout=log)


cmdclass = versioneer.get_cmdclass()
cmdclass['pylint'] = PylintCommand

# Cython C extensions
ext_modules = [
    Extension(
        name='c_hydrodiy_data',
        sources=[
            'hydrodiy/data/c_hydrodiy_data.pyx',
            'hydrodiy/data/c_dateutils.c',
            'hydrodiy/data/c_qualitycontrol.c',
            'hydrodiy/data/c_dutils.c'
        ],
        include_dirs=[numpy.get_include()]),

    Extension(
        name='c_hydrodiy_stat',
        sources=[
            'hydrodiy/stat/c_hydrodiy_stat.pyx',
            'hydrodiy/stat/c_crps.c',
            'hydrodiy/stat/c_dscore.c',
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
        "hydrodiy": ["gis/data/*.gz", "stat/data/*.zip"],
    },
    requires= [
        "pandas (>=0.14.0)",
        "scipy (>=0.14.0)",
        "Cython (>=0.20.1)",
        "numpy (>=1.8)",
        "matplotlibb (>=1.3.1)"
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
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)'
    ]
)


