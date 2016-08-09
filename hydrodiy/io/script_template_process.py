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
log_name = re.sub('\\..*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(log_name, level='INFO',
    fmt = '%(asctime)s - %(message)s',
    console=True, flog=flog)

LOGGER.critical('Script {0} started'.format(source_file))

#------------------------------------------------------------
# Get data
#------------------------------------------------------------

#fd = os.path.join(FDATA, 'data.csv')
#data, comment = csv.read_csv(fd)

sites = pd.DataFrame(np.random.uniform(size=(30, 5)))

# Extract batch of sites
idx = iutils.get_ibatch(len(sites), nbatch, ibatch)
sites = sites.iloc[idx]

mess = '{0} sites found'.format(sites.shape[0])
LOGGER.critical(mess)

#------------------------------------------------------------
# Process
#------------------------------------------------------------

ns = sites.shape[0]
count = 0

for idx, row in sites.iterrows():

    siteid = idx
    count += 1

    mess = 'Dealing with site {0} ({1:3d}/{2:3d})'.format(siteid, count, ns)
    LOGGER.critical(mess)


LOGGER.critical('Script {0} completed'.format(source_file))
