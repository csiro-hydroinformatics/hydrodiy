#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, re, json, math
import argparse

import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

from hydrodiy.io import csv, iutils

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
parser = argparse.ArgumentParser(\
    description='A script', \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--pattern', help='Site selection pattern', \
                    type=str, default='')
parser.add_argument('-i', '--ibatch', help='Batch process number', \
                    type=int, default=-1)
parser.add_argument('-n', '--nbatch', help='Number of batch process number', \
                    type=int, default=7)
args = parser.parse_args()

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)
#froot = os.path.join(os.path.dirname(source_file), '..')

fdata = froot
#fdata = os.path.join(froot, 'data')

fout = froot # os.path.join(froot, 'outputs')
os.makedirs(fout, exist_ok=True)

basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
fs = os.path.join(fdata, 'sites.csv')
sites_all, _ = csv.read_csv(fs, index_col='siteid')

# Select sites
sites = sites_all
if not args.pattern == '':
    idx = sites_all.index.str.findall(args.pattern).astype(bool)
    sites = sites_all.loc[idx, :]
else:
    if args.ibatch >=0:
        idx = iutils.get_ibatch(sites_all.shape[0], args.nbatch, \
                            args.ibatch)
        sites = sites_all.iloc[idx, :]

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

for i, (siteid, row) in enumerate(sites.iterrows()):

    LOGGER.info('dealing with {0} ({1}/{2})'.format( \
        siteid, i, len(sites)))



LOGGER.info('Process completed')

