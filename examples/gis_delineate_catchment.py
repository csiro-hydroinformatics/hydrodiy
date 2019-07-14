import os, re, math
import warnings
import unittest
import numpy as np
import pandas as pd
import warnings

import zipfile

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from hydrodiy.gis.grid import Grid, Catchment
from hydrodiy.gis.grid import accumulate, voronoi, delineate_river

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

# Define coordinates of catchment outlet
# used to delineate catchment boundaries
outletxy = [145.93375, -17.99375]

# Define coordinate of upstream point
# used to delineate the river path
upstreamxy = [145.7, -17.8]

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = os.path.abspath(__file__)
froot = os.path.dirname(source_file)

fgrid = os.path.join(froot, '..', 'hydrodiy', 'gis', 'tests', 'fdtest.bil')

fimg = os.path.join(froot, 'images', 'catchments')
os.makedirs(fimg, exist_ok=True)

#----------------------------------------------------------------------
# Get data
#----------------------------------------------------------------------

# Open flow direction grid
flowdir = Grid.from_header(fgrid)

#----------------------------------------------------------------------
# Process
#----------------------------------------------------------------------

# Create a catchment instance
ca = Catchment('test', flowdir)

# identify the cell number of the catchment outlet
idxcell = flowdir.coord2cell(outletxy)

# Delineate catchment corresponding to the defined outlet
ca.delineate_area(idxcell)
ca.delineate_boundary()

# Delineate river path
idxcell = flowdir.coord2cell(upstreamxy)
datariver = delineate_river(flowdir, idxcell, nval=160)

#----------------------------------------------------------------------
# Plots
#----------------------------------------------------------------------

plt.close('all')
fig, ax = plt.subplots()

# plot flow dir
flowdir.dtype = np.float64
data = flowdir.data
data[data>128] = np.nan
data = np.log(data)/math.log(2)
flowdir.data = data
flowdir.plot(ax, interpolation='nearest', cmap='Blues')

# plot catchment
ca.plot_area(ax, '+', markersize=2)

# plot boundary
ca.plot_boundary(ax, color='green', lw=4)

# plot river
ax.plot(datariver['x'], datariver['y'], 'r', lw=3)

fig.set_size_inches((15, 15))
fig.tight_layout()
fp = os.path.join(fimg, re.sub('\\.bil', '_plot.png', os.path.basename(fgrid)))
fig.savefig(fp)

