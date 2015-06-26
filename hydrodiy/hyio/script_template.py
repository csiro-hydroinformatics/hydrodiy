#!/usr/bin/env python




import datetime
time_now = datetime.datetime.now
print(' ## Script run started at %s ##' % time_now())

import sys, os, re, json, math

import itertools
#import requests

#from string import ascii_lowercase as letters
#from calendar import month_abbr as months

import numpy as np
import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


#------------------------------------------------------------
# Plot
#------------------------------------------------------------

plt.close('all')

fig = plt.figure()

gs = gridspec.GridSpec(3,3, 
        width_ratios=[1]*3,
        height_ratios=[3, 3, 1])

nval = 100

for i, j in itertools.product(range(3), range(3)):
    ax = fig.add_subplot(gs[i, j])

    xx = np.random.uniform(size=(nval, 2))
    x = xx[:,0]
    y = xx[:,1]

    # Scatter plot
    ax.plot(x, y, 'o',
        markersize=10,
        mec='black',
        mfc='pink',
        alpha=0.5,
        label='points')

    # Decoration
    ax.legend(frameon=True, 
        shadow=True,
        fancybox=True,
        framealpha=0.7,
        numpoints=1)

    ax.set_title('Title')
    ax.set_xlabel('X label')
    ax.set_xlabel('Y label')

fig.suptitle('Overall title')

dpi = 100
width = 1000
height = 1000
fig.set_size_inches(float(width)/dpi, float(height)/dpi)

gs.tight_layout(fig)

fp = '%s/image.png' % FIMG
fig.savefig(fp, dpi=dpi)

print(' ## Script run completed at %s ##' % time_now())
