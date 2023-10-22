#!/usr/bin/env python
# -*- coding: utf-8 -*-

[COMMENT]

import sys, os, re, json, math
import argparse
from pathlib import Path

#import warnings
#warnings.filterwarnings("ignore")

from datetime import datetime

from dateutil.relativedelta import relativedelta as delta
from string import ascii_lowercase as letters
from calendar import month_abbr as months

import numpy as np
import pandas as pd

import matplotlib as mpl

# Select backend
mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from matplotlib.backends.backend_pdf import PdfPages

from hydrodiy.io import csv, iutils
from hydrodiy.plot import putils

# Package to plot spatial data
import pyproj
from hydrodiy.gis.oz import ozlayer

import tqdm

# Code to facilitate import of a "utils" package
#import utils
#import importlib
#importlib.reload(utils)

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

parser = argparse.ArgumentParser(\
    description="[DESCRIPTION]", \
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-v", "--version", \
                    help="Version number", \
                    type=int, required=True)
parser.add_argument("-e", "--extension", help="Image file extension", \
                    type=str, default="png")
#parser.add_argument("-p", "--projection", \
#                    help="Spatial projection (GDA94=3112, WGS84=4326)", \
#                    type=int, default=3112)
args = parser.parse_args()

version = args.version

# Image file extension
imgext = args.extension

# Plot dimensions
fnrows = 2
fncols = 2
fdpi = 120
awidth = 8
aheight = 8

# Figure transparency
ftransparent = False

# Set matplotlib options
#mpl.rcdefaults() # to reset
putils.set_mpl()

# Manage projection
#proj = pyproj.Proj("+init=EPSG:{0}".format(args.projection))
#
#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()

froot = [FROOT]
fdata = [FDATA]
fout = [FOUT]
fimg = [FIMG]

#------------------------------------------------------------
# Logging
#------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

#------------------------------------------------------------
# Get data
#------------------------------------------------------------
#fd = "%s/data.csv" % FDATA
#data, comment = csv.read_csv(fd)

sites = pd.DataFrame(np.random.uniform(size=(30, 5)))

mess = "{0} sites found".format(sites.shape[0])
LOGGER.info(mess)

#------------------------------------------------------------
# Plot
#------------------------------------------------------------

# To use multipage pdf
#fpdf = fimg / "images.pdf"
#pdf = PdfPages(fpdf)

plt.close("all")

# Create figure
figsize = (awidth*fncols, aheight*fnrows)
fig = plt.figure(constrained_layout=True, figsize=figsize)

# Create mosaic with named axes
mosaic = [[f"F{fncols*i+j}" for j in range(fncols)] for i in range(fnrows)]
gw = dict(height_ratios=[1]*fnrows, width_ratios=[1]*fncols)
axs = fig.subplot_mosaic(mosaic, gridspec_kw=gw)

nval = 10
LOGGER.info("Drawing plot")

for name, ax in axs.items():
    # Retrieve column and row numbers
    iplot = int(name[1:])
    icol = iplot % fncols
    irow = iplot // fncols

    # Get data
    x = pd.date_range("2001-01-01", freq="MS", periods=nval)
    x = x.to_pydatetime()
    y = np.random.uniform(size=nval)

    # Scatter plot
    ax.plot(x, y, "o",
        markersize=10,
        mec="black",
        mfc="pink",
        alpha=0.5,
        label="points")

    # Spatial
    #ozlayer(ax, "ozcoast50m")

    # Decoration
    ax.legend(shadow=True, framealpha=0.7)

    # Axis
    title = f"{name}: Row {irow} / Column {icol}"
    ax.set_title(title)
    ax.set_xlabel("X label")
    ax.set_xlabel("Y label")

fig.suptitle("Overall title")

# Footer
label = "Generated: %s" % datetime.now().strftime("%H:%M %d/%m/%Y")
fig.text(0.05, 0.010, label, color="#595959", ha="left", fontsize=9)

# Save file
fp = fimg / f"image.{imgext}"
fig.savefig(fp, dpi=fdpi, transparent=ftransparent)
putils.blackwhite(fp)

# To save to pdf
#pdf.savefig(fig)
#pdf.close()

LOGGER.info("Plotting completed")
