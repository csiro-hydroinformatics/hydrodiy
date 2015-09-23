#!/usr/bin/env python

import os, sys, re, socket

host = socket.gethostname()

path = os.path.realpath(__file__)

if re.search('raijin', host):
    print('I think I am on raijin machine from NCI. I will use the --user option')
    os.system('python setup.py install --user --record package_files.txt')
else:
    print('Normal install, no special options used')
    os.system('python setup.py install --record package_files.txt')

os.system('python setup.py sdist --formats=gztar,zip')
