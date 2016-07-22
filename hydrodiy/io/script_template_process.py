#!/usr/bin/env python




from datetime import datetime
time_now = datetime.now
print('\n\n## Script run started at {0} ##\n\n'.format(time_now()))

import sys, os, re, json, math, logging

from dateutil.relativedelta import relativedelta as delta

from itertools import product as prod

import numpy as np
import pandas as pd

from hydrodiy.io import csv

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
logger = logging.getLogger(os.path.basename(source_file))
logger.setLevel(logging.INFO)

# log format
ft = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# log to console
sh = logging.StreamHandler()
sh.setFormatter(ft)
logger.addHandler(sh)

# log to file
flog = re.sub('py.*', 'log', source_file)
if os.path.exists(flog): os.remove(flog)
fh = logging.FileHandler(flog)
fh.setFormatter(ft)
logger.addHandler(fh)


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
