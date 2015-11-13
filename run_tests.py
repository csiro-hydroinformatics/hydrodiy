#!/usr/bin/env python

import sys
import os
import re
from hyio import iutils


# Get args
nargs = len(sys.argv)

folder = 'hydrodiy'
if nargs>1:
    folder = sys.argv[1]

pattern = 'tests.*'
if nargs>2:
    pattern = sys.argv[2]

proceeds = True
if nargs>3:
    proceeds = sys.argv[3] == 'True'

simple = True
if nargs>4:
    simple = sys.argv[3] == 'True'

if not pattern.endswith('py$'):
    pattern += 'py$'

# find test files
test_files = iutils.find_files(folder, pattern)

# Check if we proceed
if len(test_files) == 0:
    raise ValueError('No file found')

# uninstall/install
if proceeds:
    if simple:
        os.system('python setup.py install')
    else:
        os.system('./uninstall.py')
        os.system('./install.py')

# run tests
i = 1
for fn in test_files:
    if proceeds:
        print('\n(%2d) running %s:\n'%(i,fn))
        os.system('python %s'%fn)
    else:
        print('(%2d) found %s'%(i,fn))
    i += 1

