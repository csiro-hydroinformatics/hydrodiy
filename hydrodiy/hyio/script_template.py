#!/usr/bin/env python




import datetime
now = datetime.datetime.now()
print(' ## Script run started at %s ##' % now)

import sys, os, re, json, math

#import itertools
#import requests

#from string import ascii_lowercase as letters
#from calendar import month_abbr as months

import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt

#from hyio import csv

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

FIMG = '%s/images' % FROOT
if not os.path.exists(FIMG): os.mkdir(FIMG)

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


print(' ## Script run completed at %s ##' % datetime.datetime.now())
