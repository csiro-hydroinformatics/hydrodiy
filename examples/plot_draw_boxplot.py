#!/usr/bin/env python
# -*- coding: utf-8 -*-

## -- Script Meta Data --
## Author  : julien
## Created : 2018-01-15 10:16:35.712913
## Comment : Create fancy boxplots
##
## ------------------------------
import sys, re, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from calendar import month_abbr as months

import matplotlib as mpl
mpl.use("Agg")

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
source_file = Path(__file__).resolve()
froot  = source_file.parent

fimg = froot / "images" / "boxplots"
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
data = pd.DataFrame(data, columns=["V{0}".format(i) for i in range(nvar)])

# Create a random set of time series data
tsdata = np.random.normal(size=(nens, nts))
tsdata = tsdata + 3*np.sin(np.linspace(-math.pi, math.pi, nts))[None, :]
days = pd.date_range("2010-01-01", periods=nts)
tsdata = pd.DataFrame(tsdata, columns=days)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# Instanciate the boxplot object
bp = boxplot.Boxplot(data)

# Show boxplot statistics
print(bp.stats)

# Plot it
plt.close("all")

fig, ax = plt.subplots(layout="tight")
bp.draw(ax)

fp = fimg / "boxplot_default.png"
fig.savefig(fp)


# Same boxplot with centered labels and sample count
bp = boxplot.Boxplot(data, center_text=True)
fig, ax = plt.subplots(layout="tight")
bp.draw(ax)
bp.show_count()
fp = fimg / "boxplot_centertext.png"
fig.savefig(fp)


# Boxplot with large number of variables (e.g time series)
bp = boxplot.Boxplot(tsdata, style="narrow")
fig, ax = plt.subplots(layout="tight")
bp.draw(ax)

# .. reformat x axis to help readability
cc = {icn:"{0}\n{1}\n{2}".format(cn.day, months[cn.month], cn.year) \
                for icn, cn in enumerate(tsdata.columns) \
                    if cn.day in [1, 10, 20]}
cc = pd.Series(cc)
ax.set(xticks=cc.index, xticklabels=cc.values)
ax.grid(axis="x")

fp = fimg / "boxplot_narrow.png"
fig.savefig(fp)

LOGGER.info("Process completed")

