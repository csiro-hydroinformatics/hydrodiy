#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : julien
## Created : 2018-01-15 10:16:35.712913
## Comment : Create fancy boxplots
##
## ------------------------------
import sys, os, re, json, math
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt

from hydrodiy.io import iutils
from hydrodiy.plot import putils, ensplot

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

# Configure obs generation
nvar = 5
nval = 1000

# Configure times series obs generation
nens = 1000
nval = 50

putils.set_mpl(font_size=10)

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------

# Define folders
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

fimg = os.path.join(froot, 'images', 'ensemble')
os.makedirs(fimg, exist_ok=True)

# Create logger to follow script execution
basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get obs
#----------------------------------------------------------------------


# Create a random set of forecasts
fcst = np.random.normal(loc=1, size=(nval, nens))
fcst = fcst + 0.5*np.sin(np.linspace(-math.pi, math.pi, nval))[:, None]
fcst = np.maximum(fcst, 0.)

days = pd.date_range('2010-01-01', periods=nval)
fcst = pd.DataFrame(fcst, columns=['E{0}'.format(i) for i in range(nens)])

# Create a random set of obs
obs = fcst.mean(axis=1) + np.random.normal(size=nval)
obs = np.maximum(obs, 0.)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

for stat in ['mean', 'median']:

    # Create matplotlib objects
    plt.close('all')

    fig, ax = plt.subplots()

    # Draw ensemble time series and get performance metrics
    x, alpha, crps_ss, R2 = ensplot.tsplot(obs, fcst, ax, \
                show_pit=True, show_scatter=True, \
                line=stat, random_pit=True)

    # Set x ticks
    xticks = np.where(pd.Series(days.day).isin([1, 10, 20]))[0]
    xticklabels = [datetime.strftime(d, format='%d %b\n%Y') \
                                        for d in days[xticks]]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # Save figure
    fig.tight_layout()
    fp = os.path.join(fimg, 'ensemble_plot_{0}.png'.format(stat))
    fig.savefig(fp)


LOGGER.info('Process completed')

