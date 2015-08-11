#!/usr/bin/env python




import datetime
time_now = datetime.datetime.now
print(' ## Script run started at %s ##' % time_now())

import sys, os, re, json, math

import itertools
#import requests

import numpy as np
import pandas as pd

from hyio import csv

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

FDATA = '%s/data' % FROOT
if not os.path.exists(FDATA): os.mkdir(FDATA)

#------------------------------------------------------------
# Get data
#------------------------------------------------------------

#fd = '%s/data.csv' % FDATA
#data, comment = csv.read_csv(fd)

sites = pd.DataFrame(np.random.uniform(size=(30, 5)))

#------------------------------------------------------------
# Process
#------------------------------------------------------------

ns = sites.shape[0]
count = 0

for idx, row in sites.iterrows():
    
    count += 1
    print('.. dealing with site %3d / %3d ..' % (count, ns))

print(' ## Script run completed at %s ##' % time_now())
