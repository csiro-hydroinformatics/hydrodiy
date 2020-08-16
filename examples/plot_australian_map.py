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

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from hydrodiy.io import iutils
from hydrodiy.plot import putils, gridplot
from hydrodiy.gis.grid import get_grid
from hydrodiy.gis.oz import ozlayer

import pyproj

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

# Regions of Australia to focus on
regions = ['AWRAL_STATE_NSW', 'AWRAL_DRAINAGE_MURRAY_DARLING', \
                'AWRAL_RIVER_MURRUMBIDGEE', 'AWRAL_RIVER_CONDAMINE_CULGOA']

putils.set_mpl(font_size=10)

# Map projection
proj = pyproj.Proj('+init=EPSG:3112')

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
# Get gridplot config
cfg = gridplot.GridplotConfig('rainfall')

for region in regions:
    # Get region grid
    gr = get_grid(region)

    # Sample random data
    toplot = gr.clone(dtype=np.float64)
    toplot.data = np.random.uniform(cfg.clevs[0], cfg.clevs[-1], \
                                        size=(toplot.nrows, toplot.ncols))

    # Remove data outside region
    toplot.data[gr.data == 0] = np.nan

    # Create matplotlib objects
    plt.close('all')

    fig, ax = plt.subplots()

    # Plot
    cont_gr, cont_lines, _, _ = gridplot.gplot(toplot, cfg, ax, proj=proj)

    # Add Australian state boundariese
    ozlayer(ax, 'states50m', color='grey', lw=1, proj=proj)

    # Add Australian coastline
    ozlayer(ax, 'ozcoast50m', color='k', lw=3, proj=proj)

    # Set axis range
    lims = [proj(xx, yy) for xx, yy in zip(toplot.xlim, toplot.ylim)]
    lims = np.array(lims)
    a = np.mean(lims, axis=0)
    lims = (lims-a)*1.3+a

    ax.set_xlim(lims[:, 0])
    ax.set_ylim(lims[:, 1])
    ax.axis('off')

    # Figure title
    title = re.sub('_', ' ', re.sub('^[^_]+_', '', region)).title()
    ax.set_title(title)

    # Save figure
    fact = (lims[1, 1]-lims[0, 1])/(lims[1, 0]-lims[0, 0])
    fig.set_size_inches((10, 9*fact))
    fp = os.path.join(fimg, 'ozmap_{}.png'.format(region))
    fig.savefig(fp)


LOGGER.info('Process completed')

