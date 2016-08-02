#!/usr/bin/env python




from datetime import datetime
time_now = datetime.now

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

ibatch = 0
nbatch = 5

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
logger = iutils.get_logger(re.sub('\\..*', '', os.path.basename(source_file)),
    level='INFO', console=True, flog=flog)

info = '#### Script run started at {0} ####'.format(time_now())
logger.info(info)

#------------------------------------------------------------
# Get data
#------------------------------------------------------------

#fd = os.path.join(FDATA, 'data.csv')
#data, comment = csv.read_csv(fd)

sites = pd.DataFrame(np.random.uniform(size=(30, 5)))

# Extract batch of sites
idx = iutils.get_ibatch(len(sites), nbatch, ibatch)
sites = sites.iloc[idx]

#------------------------------------------------------------
# Process
#------------------------------------------------------------

ns = sites.shape[0]
count = 0

for idx, row in sites.iterrows():

    siteid = idx
    count += 1
    info = '.. dealing with site {0} ({1:3d}/{2:3d}) ..'.format(siteid, count, ns)
    logger.info(info)

info = '#### Script run completed at {0} ####'.format(time_now())
logger.info(info)
