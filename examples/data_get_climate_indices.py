#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : julien
## Created : 2018-01-15 10:14:42.857717
## Comment : Download BOM climate grids and plot them
##
## ------------------------------
import sys, os, re, json, math
import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from hydrodiy.data import hyclimind
from hydrodiy.io import iutils, csv

from hydrodiy.plot import putils

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

indices = hyclimind.INDEX_NAMES

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------

# Define folders
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

fdata = os.path.join(froot, 'data', 'climate_indices')
os.makedirs(fdata, exist_ok=True)

fimg = os.path.join(froot, 'images', 'climate_indices')
os.makedirs(fimg, exist_ok=True)

# Create logger to follow script execution
basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)


#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

data = []

# Loop through the indices
for indn in indices:
    LOGGER.info('Downloading {0}'.format(indn))

    #  --- Download the data ---
    series, url = hyclimind.get_data(indn)
    series = series['1900-01-01':]
    series.name = indn
    data.append(series)

    # --- Plot the data --
    plt.close('all')
    fig, ax = plt.subplots()
    series.plot(ax=ax)
    ax.set_title(indn.upper())
    fig.savefig(os.path.join(fimg, '{0}.png'.format(indn)))

# Save all indices to disk
data = pd.concat(data, axis=1)
fd = os.path.join(fdata, 'climate_indices.csv')
comments = {'comment': 'Monthly time series of climate indices'}
csv.write_csv(data, fd, comments, source_file, compress=False, \
            write_index=True)

LOGGER.info('Process completed')

