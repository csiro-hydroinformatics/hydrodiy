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

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from hydrodiy.io import csv, iutils
from hydrodiy.data import hykiwis
from hydrodiy.plot import putils

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

# Data start
start = datetime(2015, 1, 1)

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
        type = 'flow'
        ksiteid = siteid
        dh = -9

    else:
        ts_name = 'daily_0am'
        type = 'storage'
        idx = storages_all.name.str.lower().str.findall(siteid)
        idx = idx.astype(bool).values
        ksiteid = storages_all.index[idx][0]
        dh = 0

    # Download time series attributes
    try:
        attrs, url = hykiwis.get_tsattrs(ksiteid, ts_name, external=True)

        # Use the first one
        attrs = attrs[0]

        # Download data for the first attribute
        ts, url = hykiwis.get_data(attrs, start=start, external=True)
    except:
        continue

    # Set same time for all series
    ts = ts.shift(periods=dh, freq='H')

    # Convert flow to ML/d if needed
    conversion = 1.
    if attrs['ts_unitsymbol'] == 'cumec':
        symbol = 'ML/d'
        conversion = 86.4
    else:
        symbol = attrs['ts_unitsymbol']

    data['{0}_{1}[{2}]'.format(type, siteid, symbol)] = ts * conversion


if len(data)>0:
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
    fig, axs = plt.subplots(nrows=2)
    axs = axs.flat

    ax = axs[0]
    data.filter(regex='flow', axis=1).plot(ax=ax)
    ax.set_title('Flow data')

    ax = axs[1]
    df = data.filter(regex='storage', axis=1)
    df = df-df.min()
    df.plot(ax=ax)
    ax.set_title('Storage data')

    # Save plot to image
    fig.set_size_inches((15, 12))
    fig.tight_layout()
    fp = os.path.join(fimg, 'kiwis_data.png')
    fig.savefig(fp)


LOGGER.info('Process completed')

