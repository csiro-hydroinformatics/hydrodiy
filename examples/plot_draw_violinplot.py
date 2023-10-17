#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : julien
## Created : 2018-01-15 10:16:35.712913
## Comment : Create fancy violinplots
##
## ------------------------------
from pathlib import Path
import sys, os, re, json, math
import numpy as np
import pandas as pd
from calendar import month_abbr as months

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from hydrodiy.io import iutils
from hydrodiy.plot import putils, violinplot

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

# Configure data generation
nvar = 5
nval = 1000

# Configure times series data generation
nens = 1000
nts = 50

putils.set_mpl(font_size=10)

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot  = source_file.parent

fimg = froot / "images" / "violinplots"
fimg.mkdir(exist_ok=True, parents=True)

#----------------------------------------------------------------------
# Logging
#----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------

# Create a random set of data
data = np.random.normal(size=(nval, nvar))
data = data + np.linspace(0, 3, nvar)[None, :]
data = pd.DataFrame(data, columns=['V{0}'.format(i) for i in range(nvar)])

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------
# Instanciate the violinplot object
vl = violinplot.Violin(data)

# Show violinplot statistics
print("median values:")
print(vl.stat_median)

# Plot it
plt.close('all')

fig, ax = plt.subplots(layout="tight")
vl.draw(ax)

fp = os.path.join(fimg, 'violinplot_default.png')
fig.savefig(fp)


# Same violinplot with different colors
vl = violinplot.Violin(data, crm="brown", cro="tab:orange")
fig, ax = plt.subplots(layout="tight")
vl.draw(ax)
fp = os.path.join(fimg, 'violinplot_colors.png')
fig.savefig(fp)


LOGGER.info('Process completed')

