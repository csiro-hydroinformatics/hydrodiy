#!/usr/bin/env python

import os
os.system('python setup.py install --user --record package_files.txt')
os.system('python setup.py sdist --formats=gztar,zip')
