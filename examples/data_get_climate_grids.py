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

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from hydrodiy.data import hywap
from hydrodiy.io import iutils

from hydrodiy.gis.grid import get_mask
from hydrodiy.gis.oz import Oz

from hydrodiy.plot.gridplot import gplot, gsmooth, GridplotConfig
from hydrodiy.plot.gridplot import gbar

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

# Variable to download
varnames = ['rainfall', 'temperature']

# Start / end of the period to download
# 5 days ending on day before yesterday
now = datetime.now()
dt = now-delta(days=2)
end = datetime(dt.year, dt.month, dt.day)

start = end-delta(days=5)

timestep = 'day'

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------

# Define folders
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

fdata = os.path.join(froot, 'data', 'climate_grids')
os.makedirs(fdata, exist_ok=True)

fimg = os.path.join(froot, 'images', 'climate_grids')
os.makedirs(fimg, exist_ok=True)

# Create logger to follow script execution
basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)


#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# Create a series of dates
days = pd.date_range(start, end)

# Loop through the variable names
for varn in varnames:

    # Extract the type of data that can be downloaded
    obj = hywap.VARIABLES[varn]
    vartypes = [vt['type'] for vt in obj]

    # Loop through the variable types
    for vart in vartypes:
        for day in days:
            LOGGER.info('Downloading {0}-{1} for {2}'.format(\
                    varn, vart, day))

            #  --- Download the data ---
            grd = hywap.get_data(varn, vart, timestep, day)

            # --- Save file to disk ---
            basename = '{0}_{1}_{2}_{3}.bil'.format(\
                    varn, vart, timestep, day.date())
            fg = os.path.join(fdata, basename)
            grd.save(fg)

            # --- Plot the grid - basic ---
            plt.close('all')
            fig, ax = plt.subplots()
            grd.plot(ax, cmap='Blues')
            ax.set_title('{0} - {1} - {2}'.format(varn, vart, day.date()))
            fig.savefig(os.path.join(fimg, re.sub('\.bil', \
                                '_basic.png', basename)))

            # --- Plot the grid - advanced ---
            fig = plt.figure()

            # .. Get the grid spec object
            gs = GridSpec(nrows=3, ncols=3, \
                height_ratios=[1, 4, 1], \
                width_ratios=[6, 1, 1])

            # .. get map of australia
            ax = plt.subplot(gs[:,0])
            om = Oz(ax=ax)
            bm = om.map

            # .. smooth data
            mask = get_mask('AWAP')
            grd_smooth = gsmooth(grd, mask)

            # .. get plot config
            cfg = GridplotConfig(varn)

            # .. draw grid data
            contf = gplot(grd_smooth, om.map, cfg)

            # .. axis title
            ax.set_title('{0} - {1} - {2}'.format(varn, vart, day.date()))

            # .. state boundaries
            om.drawstates()

            # .. draw color bar
            cbar_ax = plt.subplot(gs[1, 2])
            gbar(cbar_ax, cfg, contf)

            fig.tight_layout()
            fig.savefig(os.path.join(fimg, re.sub('\.bil', \
                                '_advanced.png', basename)))


LOGGER.info('Process completed')

