#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : julien
## Created : 2018-01-15 10:15:13.785549
## Comment : Download BOM Water Data Online data
##
## ------------------------------
import sys, os, re, json, math
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta as delta

import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.data import hykiwis
from hydrodiy.plot import putils

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

# Data start
start = datetime(2010, 1, 1)

# Upper Murray basin
siteids = ['401012', '401203', '401549', '401211', '401204', '401201', \
                '402205']

storages = ['hume', 'dartmouth']

# Configure matplotlib
putils.set_mpl()

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------

# Define folders
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

fdata = os.path.join(froot, 'data', 'kiwis')
os.makedirs(fdata, exist_ok=True)

fimg = os.path.join(froot, 'images', 'kiwis')
os.makedirs(fimg, exist_ok=True)

# Create logger to follow script execution
basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# Get storages info
storages_all = hykiwis.get_storages()

# Download flow data
data = {}
for siteid in siteids+storages:
    LOGGER.info('downloading data for '+siteid)

    # Set time series name
    if siteid in siteids:
        ts_name = 'daily_9am'
        symbol = 'ML/d'
        type = 'flow'
        ksiteid = siteid
        dh = -9

    else:
        ts_name = 'daily_12pm'
        symbol = 'ML'
        type = 'storage'
        idx = storages_all.name.str.lower().str.findall(siteid)
        idx = idx.astype(bool).values
        ksiteid = storages_all.index[idx][0]
        dh = 0

    # Download time series attributes
    attrs, url = hykiwis.get_tsattrs(ksiteid, ts_name, \
                        external=False)

    # Use the first one
    attrs = attrs[0]

    # Download data for the first attribute
    ts, url = hykiwis.get_data(attrs, start=start, external=True)

    # Set same time for all series
    ts = ts.shift(periods=dh, freq='H')

    # Convert flow to ML/d if needed
    conversion = 1.
    if attrs['ts_unitsymbol'] == 'cumec':
        conversion = 86.4

    data['{0}_{1}[{2}]'.format(type, siteid, symbol)] = ts * conversion


# Store as csv
LOGGER.info('Writing data to disk')
data = pd.DataFrame(data)
fd = os.path.join(fdata, 'kiwis_data.csv')
comments = {'comment': 'Flow and storage data extracted from '+\
                            'BOM-Water Data Online'}
csv.write_csv(data, fd, comments, source_file, compress=False, \
                write_index=True)

# plot data data
plt.close('all')
fig, axs = putils.get_fig_axs(nrows=2)

ax = axs[0]
data.filter(regex='flow', axis=1).plot(ax=ax)
ax.set_title('Flow data')

ax = axs[1]
data.filter(regex='storage', axis=1).plot(ax=ax)
ax.set_title('Storage data')

# Save plot to image
fig.set_size_inches((15, 12))
fig.tight_layout()
fp = os.path.join(fimg, 'kiwis_data.png')
fig.savefig(fp)


LOGGER.info('Process completed')

