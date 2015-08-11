#!/usr/bin/env python




import datetime
time_now = datetime.datetime.now
print(' ## Script run started at %s ##' % time_now())

import sys, os, re, json, math

import itertools

from string import ascii_lowercase as letters
from calendar import month_abbr as months

import numpy as np
import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from hyio import csv

#------------------------------------------------------------
# Options
#------------------------------------------------------------

#nargs = len(sys.argv)
#if nargs > 0:
#    arg1 = sys.argv[1]

# Plotting options
fig_dpi = 100
fig_nrows = 2
fig_ncols = 3
ax_width = 1000
ax_height = 1000

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
# Plot
#------------------------------------------------------------

plt.close('all')

fig = plt.figure()

gs = gridspec.GridSpec(fig_nrows, fig_ncols, 
        width_ratios=[1] * fig_ncols,
        height_ratios=[1] * fig_nrows)

nval = 100

for i, j in itertools.product(range(fig_nrows), 
                            range(fig_ncols)):

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

fig.set_size_inches(float(fig_ncols * ax_width)/fig_dpi, 
                float(fig_nrows * ax_height)/fig_dpi)

gs.tight_layout(fig)

fp = '%s/image.png' % FIMG
fig.savefig(fp, dpi=fig_dpi)


print(' ## Script run completed at %s ##' % time_now())
