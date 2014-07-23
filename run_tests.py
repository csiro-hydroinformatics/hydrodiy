#!/usr/bin/env python

import sys
import os
import re
from hyio import iutils

folder = '/home/magpie/Dropbox/code/pypackage/hydrodiy'
pattern = 'tests.*'
proceeds = None

# Get args
nargs = len(sys.argv)
if nargs>1:
    folder = sys.argv[1]
if nargs>2:
    pattern = sys.argv[2]
if nargs>3:
    proceeds = sys.argv[3]

if not pattern.endswith('py$'):
    pattern += 'py$'

# find test files
test_files = iutils.find_files(folder, pattern)

# Check if we proceed
nf = len(test_files)
if proceeds is None:
    proceeds = raw_input('\n\npattern: %s\n\t=> %d test files found\n\nproceeds (y/show/n)?\n'%(pattern, nf))

# uninstall/install
if proceeds=='y':
    os.system('/home/magpie/Dropbox/code/pypackage/uninstall.py')
    os.system('/home/magpie/Dropbox/code/pypackage/install.py')

# run tests
i = 1
for fn in test_files:
    if proceeds=='y':
        print('\n(%2d) running %s:\n'%(i,fn))
        os.system('python %s'%fn)
    elif proceeds=='show':
        print('(%2d) found %s'%(i,fn))
    i += 1

