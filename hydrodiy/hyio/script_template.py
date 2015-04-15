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

FROOT = os.path.dirname(os.path.abspath(__file__))

FIMG = '%s/images' % FROOT
if not os.path.exists(FIMG): os.mkdir(FIMG)

FDATA = '%s/data' % FROOT
if not os.path.exists(FDATA): os.mkdir(FDATA)

#------------------------------------------------------------
# Get data
#------------------------------------------------------------

#fd = '%s/data.csv' % FDATA
#data, comment = csv.read_csv(fd)

#------------------------------------------------------------
# Process
#------------------------------------------------------------


print(' ## Script run completed at %s ##' % datetime.datetime.now())
