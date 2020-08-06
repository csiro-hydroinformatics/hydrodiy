#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : julien
## Created : 2020-08-06 Thu 04:30 PM
## Comment : Plot data in Australia
##
## ------------------------------
import sys, os, re, json, math
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial import cKDTree as KDTree

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from hydrodiy.io import iutils
from hydrodiy.plot import putils
from hydrodiy.gis.grid import get_grid
from hydrodiy.gis.oz import ozlayer

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

# Regions of Australia to focus on
regions = ['AWRAL_STATE_NSW', 'AWRAL_DRAINAGE_MURRAY_DARLING', \
                'AWRAL_RIVER_MURRUMBIDGEE', 'AWRAL_RIVER_CONDAMINE_CULGOA']

putils.set_mpl(font_size=10)

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------

# Define folders
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

fimg = os.path.join(froot, 'images', 'ozmaps')
os.makedirs(fimg, exist_ok=True)

# Create logger to follow script execution
basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

for region in regions:
    # Get region grid
    gr = get_grid(region)

    # Sample random data
    d = gr.clone(dtype=np.float64)
    d.data = np.random.uniform(size=(d.nrows, d.ncols))

    # Remove data outside region
    d.data[gr.data == 0] = np.nan

    # Create matplotlib objects
    plt.close('all')

    fig, ax = plt.subplots()

    im = d.plot(ax=ax)
    plt.colorbar(im)

    # Add Australian coastline
    ozlayer(ax, 'ozcoast50m', color='k', lw=2)

    # Add Australian state boundariese
    ozlayer(ax, 'states50m', color='grey', linestyle='-.', lw=0.8)

    # Set axis range
    ax.set_xlim(d.xlim)
    ax.set_ylim(d.ylim)

    # Figure title
    title = re.sub('_', ' ', re.sub('^[^_]+_', '', region)).title()
    ax.set_title(title)

    # Save figure
    fig.tight_layout()
    fp = os.path.join(fimg, 'ozmap_{}.png'.format(region))
    fig.savefig(fp)


LOGGER.info('Process completed')

