#!/usr/bin/env python




from datetime import datetime
time_now = datetime.now
print('\n\n## Script run started at {0} ##\n\n'.format(time_now()))

import sys, os, re, json, math

from dateutil.relativedelta import relativedelta as delta

from itertools import product as prod

import numpy as np
import pandas as pd

from hydrodiy.io import csv, iutils

#------------------------------------------------------------
# Options
#------------------------------------------------------------

#nargs = len(sys.argv)
#if nargs > 0:
#    arg1 = sys.argv[1]

#------------------------------------------------------------
# Functions
#------------------------------------------------------------

def fun(x):
    return x

#------------------------------------------------------------
# Folders
#------------------------------------------------------------

source_file = os.path.abspath(__file__)

FROOT = os.path.dirname(source_file)

FDATA = os.path.join(FROOT, 'data')
if not os.path.exists(FDATA): os.mkdir(FDATA)

#------------------------------------------------------------
# Logging
#------------------------------------------------------------
flog = source_file + '.log'
if os.path.exists(flog): os.remove(flog)
logger = iutils.get_logger(os.path.basename(source_file),
    level='INFO', console=True, flog=flog)

#------------------------------------------------------------
# Get data
#------------------------------------------------------------

#fd = os.path.join(FDATA, 'data.csv')
#data, comment = csv.read_csv(fd)

sites = pd.DataFrame(np.random.uniform(size=(30, 5)))

#------------------------------------------------------------
# Process
#------------------------------------------------------------

ns = sites.shape[0]
count = 0

for idx, row in sites.iterrows():

    count += 1
    info = '.. dealing with site {0:3d} / {1:3d} ..'.format(count, ns)
    logger.info(info)

print('\n\n## Script run completed at {0} ##\n\n'.format(time_now()))
