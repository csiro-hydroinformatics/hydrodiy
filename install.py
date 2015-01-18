#!/usr/bin/env python

import os, sys, re

path = os.path.realpath(__file__)

if re.search('jml548', path):
    print('I think I am on NCI. I will use the --user option')
    os.system('python setup.py install --user --record package_files.txt')
else:
    print('Normal install, no special options used')
    os.system('python setup.py install --record package_files.txt')

os.system('python setup.py sdist --formats=gztar,zip')
