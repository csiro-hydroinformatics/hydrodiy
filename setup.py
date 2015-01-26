#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

import os, re

# Find gnuwin32 for windows compilation
gnuwin32lib = []
gnuwin32inc = []
if 'PATH' in os.environ:
	path = os.environ['PATH']
	gnuwin32lib = [p for p in path.split(';') if re.search('gnuwin32.*lib', p)]
	gnuwin32inc = [p for p in path.split(';') if re.search('gnuwin32.*include', p)]


setup(name='hydrodiy',
      version='0.3',
      description='Utility functions for hydrological modelling',
      author='Julien Lerat',
      author_email='julien.lerat@gmail.com',
      url='http://tuffgong.com/',
      packages=['hymod', 'hygis', 'hyplot', 'hydata',  
                    'hyio', 'hywafari','hystat'],
      package_dir={'hymod':'hydrodiy/hymod', 
            'hygis':'hydrodiy/hygis',
            'hydata':'hydrodiy/hydata',
            'hyio':'hydrodiy/hyio',
            'hyplot':'hydrodiy/hyplot',
            'hywafari':'hydrodiy/hywafari',
            'hystat':'hydrodiy/hystat'},
      package_data={'hygis': ['data/*.csv', 'data/*.png', 'data/*.pngw'],
          'hyio': ['*.py.gz']},
      requires=['pandas (>=11.0)', 'numpy (>=1.7.1)'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension('_metrics', 
                        sources=['hydrodiy/hymod/_metrics.pyx', 
                            'hydrodiy/hymod/c_crps.c'],
                        include_dirs=[numpy.get_include()]),
                    Extension('_match', 
                        sources=['hydrodiy/hyplot/_match.pyx', 
                            'hydrodiy/hyplot/c_match.c'],
                        libraries=['gsl', 'gslcblas'],
                        library_dirs=['/usr/local/lib', '~/.local/lib'] + gnuwin32lib,
                        include_dirs=[numpy.get_include(), '~/.local/lib/include'] + gnuwin32inc),
                    Extension('_sutils', 
                        sources=['hydrodiy/hystat/_sutils.pyx', 
                            'hydrodiy/hystat/c_ar1.c'],
                        libraries=['gsl', 'gslcblas'],
                        library_dirs=['/usr/local/lib', '~/.local/lib'] + gnuwin32lib,
                        include_dirs=[numpy.get_include(), '~/.local/lib/include'] + gnuwin32inc),
                    Extension('_datacheck', 
                        sources=['hydrodiy/hydata/_datacheck.pyx', 
                            'hydrodiy/hydata/c_lindetect.c'],
                        include_dirs=[numpy.get_include()])],
    )


