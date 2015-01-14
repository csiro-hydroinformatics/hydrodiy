#!/usr/bin/env python

import os, sys

if len(sys.argv)>1:
    option = sys.argv[1]
    os.system('python setup.py install %s --record package_files.txt' % option)
else:
    os.system('python setup.py install --record package_files.txt')

os.system('python setup.py sdist --formats=gztar,zip')
