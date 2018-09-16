#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : jlerat
## Created : 2018-09-16 21:14:03.674849
## Comment : Run all example scripts
##
## ------------------------------
import sys, os, re
import subprocess

from hydrodiy.io import csv, iutils

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
lf = iutils.find_files(froot, '.*\.py')

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

for f in lf:
    fname = os.path.basename(f)
    if re.search('run_all_examples', f):
        LOGGER.info('Skip '+ fname)
        continue

    LOGGER.info('Running {0}'.format(fname))
    cmd = 'python ' + f
    subprocess.check_call(cmd, shell=True)


LOGGER.info('Process completed')

