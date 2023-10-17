#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : jlerat
## Created : 2018-09-16 21:14:03.674849
## Comment : Run all example scripts
##
## ------------------------------
import sys, re
from pathlib import Path
import subprocess

from hydrodiy.io import csv, iutils

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot = source_file.parent

basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
lf = froot.glob("*.py")

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

for f in lf:
    if re.search('run_all_examples', f.stem):
        LOGGER.info('Skip '+ f.name)
        continue

    LOGGER.info('Running {0}'.format(f.stem))
    cmd = f'python {f}'
    subprocess.check_call(cmd, shell=True)


LOGGER.info('Process completed')

