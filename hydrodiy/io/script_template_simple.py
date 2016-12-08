#!/usr/bin/env python

import sys, os, re, json, math
import numpy as np
import pandas as pd

import zipfile
from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

from hydrodiy.io import csv, iutils

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

# Define config options
basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
fdata = froot #os.path.join('data')

fout = os.path.join('outputs')
if not os.path.exists(fout): os.mkdir(fout)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------
fs = os.path.join(fdata, 'sites.csv')
sites, _ = csv.read_csv(fs, index_col='siteid')

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

for i, (siteid, row) in enumerate(sites.iterrows()):

    LOGGER.info('dealing with {0} ({1}/{2})'.format( \
        siteid, i, len(sites)))




LOGGER.info('Process completed')

