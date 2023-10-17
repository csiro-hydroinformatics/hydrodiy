import re, math
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import warnings

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from hydrodiy.gis.grid import Grid, Catchment
from hydrodiy.gis.grid import accumulate, voronoi, delineate_river

from hydrodiy.io import iutils

#----------------------------------------------------------------------
# Config
#----------------------------------------------------------------------

# Define coordinates of catchment outlet
# used to delineate catchment boundaries
outletxy = [145.9335, -17.9935]

# Define coordinate of upstream point
# used to delineate the river path
upstreamxy = [145.7, -17.8]

#----------------------------------------------------------------------
# Folders
#----------------------------------------------------------------------
source_file = Path(__file__).resolve()
froot  = source_file.parent

fgrid = froot.parent / "hydrodiy" / "gis" / "tests" / "fdtest.bil"

fimg = froot / "images" / "catchments"
fimg.mkdir(exist_ok=True, parents=True)

#----------------------------------------------------------------------
# Logging
#----------------------------------------------------------------------
basename = source_file.stem
LOGGER = iutils.get_logger(basename)

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

# Intersect with a coarser grid
coarse = Grid('coarse', nrows=9, ncols=9, \
                xllcorner=flowdir.xllcorner, \
                yllcorner=flowdir.yllcorner, \
                cellsize=0.1)
gri, _, _ = ca.intersect(coarse)

#----------------------------------------------------------------------
# Plots
#----------------------------------------------------------------------

plt.close('all')
fig, ax = plt.subplots(figsize=(15, 15), layout="tight")

# plot flow dir
flowdir.dtype = np.float64
data = flowdir.data
data[data>128] = np.nan
data = np.log(data)/math.log(2)
flowdir.data = data
flowdir.plot(ax, interpolation='nearest', cmap='Blues')

# Plot intersect
gri.plot(ax=ax, cmap='Reds', alpha=0.3)
gri.plot_values(ax=ax, fontsize=20, color='w', fontweight='bold')

# plot catchment
ca.plot_area(ax, '+', markersize=2)

# plot boundary
ca.plot_boundary(ax, color='green', lw=4)

# plot river
ax.plot(datariver['x'], datariver['y'], 'r', lw=3)

# Save image
fp = fimg / f"{fgrid.stem}_plot.png"
fig.savefig(fp)

