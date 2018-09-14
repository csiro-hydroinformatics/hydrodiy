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
from calendar import month_abbr as months

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt

from hydrodiy.io import iutils
from hydrodiy.plot import putils, boxplot

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

# Define folders
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

fimg = os.path.join(froot, 'images', 'boxplots')
os.makedirs(fimg, exist_ok=True)

# Create logger to follow script execution
basename = re.sub('\\.py.*', '', os.path.basename(source_file))
LOGGER = iutils.get_logger(basename)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------

# Create a random set of data
data = np.random.normal(size=(nval, nvar))
data = data + np.linspace(0, 3, nvar)[None, :]
data = pd.DataFrame(data, columns=['V{0}'.format(i) for i in range(nvar)])

# Create a random set of time series data
tsdata = np.random.normal(size=(nens, nts))
tsdata = tsdata + 3*np.sin(np.linspace(-math.pi, math.pi, nts))[None, :]
days = pd.date_range('2010-01-01', periods=nts)
tsdata = pd.DataFrame(tsdata, columns=days)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------


# Instanciate the boxplot object
bp = boxplot.Boxplot(data)

# Show boxplot statistics
print(bp.stats)

# Plot it
plt.close('all')

fig, ax = plt.subplots()
bp.draw(ax)

fig.tight_layout()
fp = os.path.join(fimg, 'boxplot_default.png')
fig.savefig(fp)


# Same boxplot with centered labels and sample count
bp = boxplot.Boxplot(data, centertext=True)
fig, ax = plt.subplots()
bp.draw(ax)
bp.show_count()
fig.tight_layout()
fp = os.path.join(fimg, 'boxplot_centertext.png')
fig.savefig(fp)


# Boxplot with large number of variables (e.g time series)
bp = boxplot.Boxplot(tsdata, style='narrow')
fig, ax = plt.subplots()
bp.draw(ax)

# .. reformat x axis to help readability
cc = {icn:'{0}\n{1}\n{2}'.format(cn.day, months[cn.month], cn.year) \
                for icn, cn in enumerate(tsdata.columns) \
                    if cn.day in [1, 10, 20]}
cc = pd.Series(cc)
ax.set_xticks(cc.index)
ax.set_xticklabels(cc.values)
ax.grid(axis='x')

fig.tight_layout()
fp = os.path.join(fimg, 'boxplot_narrow.png')
fig.savefig(fp)


LOGGER.info('Process completed')

