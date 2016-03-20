#!/usr/bin/env python




from datetime import datetime
time_now = datetime.now
print('\n\n## Script run started at {0} ##\n\n'.format(time_now()))

import sys, os, re, json, math

import itertools

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
    print('.. dealing with site {0:3d} / {1:3d} ..'.format(count, ns))

print('\n\n## Script run completed at {0} ##\n\n'.format(time_now()))
