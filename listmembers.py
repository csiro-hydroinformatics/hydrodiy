#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, re
import pandas as pd
import importlib
from inspect import getmembers, isfunction, isclass, getsourcefile, \
                        isbuiltin, getdoc
from hydrodiy.io import iutils, csv

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)
fhydrodiy = os.path.join(froot, 'hydrodiy')

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
lf = iutils.find_files(fhydrodiy, '(?<=[a-z])\.py$')

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

members = pd.DataFrame(columns=['package', 'module', \
                'object_name', 'type', 'doc'])

for f in lf:
    if re.search('version|template|test', f):
        continue

    name = re.sub('.py$', '', os.path.basename(f))
    modname = re.sub('.*[^A-Za-z]', '', os.path.dirname(f))
    import_name = 'hydrodiy.{0}.{1}'.format(modname, name)
    module = importlib.import_module(import_name, package='hydrodiy')

    for obj in getmembers(module):

        # Check object is a class or a function
        if isfunction(obj[1]) or isclass(obj[1]):

            # Check object resides in hydrodiy
            skip = False
            try:
                if not re.search('hydrodiy', getsourcefile(obj[1])):
                    skip = True
            except TypeError:
                skip = True

            if skip:
                continue

            # Get doc
            try:
                doc = re.sub('\\n.*', '', getdoc(obj[1]))
            except TypeError:
                doc = ''

            # Store data
            tp = 'class' if isclass(obj[1]) else 'function'
            dd = {\
                    'package': 'hydrodiy',\
                    'module': import_name,\
                    'object_name': obj[0], \
                    'type': tp, \
                    'doc': doc
            }
            members = members.append(dd, ignore_index=True)


comment = 'List of classes and functions in hydrodiy'
fc = os.path.join(froot, 'listmembers.html')
pd.set_option('display.max_colwidth', 200)
members.to_html(fc)
#csv.write_csv(members, fc, comment, source_file, compress=False)


