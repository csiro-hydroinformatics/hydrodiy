''' Objects used to download data from  AWAP '''

import re
import os
import datetime

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from mpl_toolkits.basemap import cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hydrodiy.data import hywap
from  hydrodiy.plot import putils


VARNAMES =  hywap.VARIABLES.keys() + ['effective-rainfall', \
    'decile-rain', 'decile-temp', 'evapotranspiration', \
    'soil-moisture']


class GridplotConfig:
    ''' Class containing gridplot configuration data '''

    def __init__(self, varname=None):
        # Set default values
        self.cmap = plt.cm.RdBu
        self._clevs = None
        self._clevs_ticks = None
        self._clevs_tick_labels = None
        self.norm = None
        self.linewidth = 0.8
        self.linecolor = '#%02x%02x%02x' % (150, 150, 150)
        self.varname = varname

        # Refine default values based on variable name
        self._default_values(varname)


    @property
    def clevs(self):
        return self._clevs

    @clevs.setter
    def clevs(self, value):
        self._clevs = np.atleast_1d(value).astype(np.float64)
        self.clevs_ticks = self._clevs


    @property
    def clevs_ticks(self):
        return self._clevs_ticks

    @clevs_ticks.setter
    def clevs_ticks(self, value):
        self._clevs_ticks = np.atleast_1d(value).astype(np.float64)
        self.clevs_tick_labels = self._clevs_ticks


    @property
    def clevs_tick_labels(self):
        return self._clevs_tick_labels

    @clevs_tick_labels.setter
    def clevs_tick_labels(self, value):
        labels = np.atleast_1d(value)
        if not len(labels) == len(self._clevs_ticks):
            raise ValueError(('Number of labels({0}) different ' + \
                'from number of ticks ({1})').format(len(labels), \
                    len(self._clevs_ticks)))

        self._clevs_tick_labels = labels


    def is_valid(self):
        ''' Check configuration is properly set '''
        if self._clevs is None or \
            self._clevs_ticks is None or \
            self._clevs_tick_labels is None:
            raise ValueError('clevs, or clevs_ticks or clevs_tick_labels is None')

        if len(self.clevs_tick_labels) != len(self.clevs_ticks):
            raise ValueError('Not the same number of tick levels and tick' + \
                    ' labels')

    def _default_values(self, varname):

        if varname is None:
            return

        if not varname in VARNAMES:
            raise ValueError(('Variable {0} not in '+\
                '{1}').format(varname, '/'.join(VARNAMES)))

        if varname == 'decile-rain':
            self.clevs = [0., 0.1, 0.3, 0.7, 0.9, 1.0]
            self.clevs_ticks = [0.05, 0.2, 0.5, 0.8, 0.95]
            self.clevs_tick_labels = ['Very Much\nBelow Average', 'Below Average', 'Average',
                        'Above Average', 'Very Much\nAbove Average']

            cols = {1.:'#%02x%02x%02x' % (102, 102, 255),
                    0.5:'#%02x%02x%02x' % (255, 255, 255),
                    0.:'#%02x%02x%02x' % (255, 102, 102)}
            self.cmap = putils.col2cmap(cols)
            self.norm = mpl.colors.Normalize(vmin=self.clevs[0], vmax=self.clevs[-1])

        if varname == 'decile-temp':
            self._default_values('decile-rain')

            cols = {1.:'#%02x%02x%02x' % (255, 153, 0),
                    0.5:'#%02x%02x%02x' % (255, 255, 255),
                    0.:'#%02x%02x%02x' % (0, 153, 204)}
            self.cmap = putils.col2cmap(cols)

        elif varname == 'evapotranspiration':
            clevs = [0, 10, 50, 80, 100, 120, 160, 200, 250, 300, 350]
            self.clevs = clevs
            self.clevs_tick_labels = clevs[:-1] + ['']

            cols = {0.:'#%02x%02x%02x' % (255, 229, 204),
                    1.:'#%02x%02x%02x' % (153, 76, 0)}
            self.cmap = putils.col2cmap(cols)
            self.norm = mpl.colors.Normalize(vmin=clevs[0], vmax=clevs[-1])

        elif varname == 'soil-moisture':
            self.clevs = np.arange(0, 1.05, 0.05)
            self.clevs_ticks = np.arange(0, 1.2, 0.2)
            self.clevs_tick_labels = ['{0:3.0f}%'.format(l*100) \
                                                for l in self.clevs_ticks]
            self.cmap = plt.cm.Blues
            self.linewidth = 0.
            self.norm = mpl.colors.Normalize(vmin=self.clevs[0], vmax=self.clevs[-1])

        elif varname == 'effective-rainfall':
            clevs = [-200, -100, -75, -50, -25, -10, -5, 0, \
                5, 10, 25, 50, 75, 100, 200]
            self.clevs = clevs
            self.clevs_tick_labels = [''] + clevs[1:-1] + ['']

            cols = {0.:'#%02x%02x%02x' % (255, 76, 0), \
                    0.5:'#%02x%02x%02x' % (255, 255, 255), \
                    1.:'#%02x%02x%02x' % (40, 178, 157)}
            self.cmap = putils.col2cmap(cols)
            self.linewidth = 0
            self.norm = mpl.colors.SymLogNorm(10., vmin=clevs[0], vmax=clevs[-1])

        elif varname == 'rainfall':
            clevs = [0, 1, 5, 10, 25, 50, 100, 200, 300, 400, 600, 800, 1000]
            self.clevs = clevs
            self.clevs_tick_labels = clevs[:-1] + ['']

            self.linewidth = 0
            self.norm = mpl.colors.SymLogNorm(10., vmin=clevs[0], vmax=clevs[-1])

        elif varname == 'temperature':
            clevs = [-20] + range(-9, 51, 3) + [60]
            self.clevs = clevs
            self.clevs_tick_labels = clevs[:-1] + ['']

            self.linewidth = 0
            self.cmap = plt.get_cmap('gist_rainbow_r')
            self.norm = mpl.colors.Normalize(vmin=clevs[0], vmax=clevs[-1])

        elif varname == 'vprp':
            self._default_values('temperature')
            clevs = range(0, 42, 2)
            self.clevs = clevs
            self.clevs_tick_labels = clevs[:-1] + ['']

        elif varname == 'solar':
            self._default_values('temperature')
            clevs = range(0, 43, 3)
            self.clevs = clevs
            self.clevs_tick_labels = clevs[:-1] + ['']



def gsmooth(grid, mask=None, sigma=5., minval=-np.inf, eps=1e-6):
    ''' Smooth gridded value to improve look of map '''

    smooth = grid.clone(dtype=np.float64)
    z0 = smooth.data

    # Gapfill with nearest neighbour
    ixm, iym = np.where(np.isnan(z0) | (z0<minval-eps))
    ixnm, iynm = np.where(~np.isnan(z0) & (z0>=minval-eps))
    z0[ixm, iym] = -np.inf
    z1 = maximum_filter(z0, size=50, mode='nearest')
    z1[ixnm, iynm] = z0[ixnm, iynm]

    # Smooth
    z2 = gaussian_filter(z1, sigma=sigma, mode='nearest')

    # Cut mask
    if not mask is None:
        # Check grid and mask have the same geometry
        if not grid.same_geometry(mask):
            raise ValueError('Mask does not have the same '+
                'geometry than input grid')

        ixm, iym = np.where(mask.data == 0)
        z2[ixm, iym] = np.nan

    smooth.data = z2

    return smooth



def gplot(grid, basemap_object, config):
    ''' Plot gridded data '''

    # Check config is proper
    config.is_valid()

    # Get cell coordinates
    ncells = grid.nrows*grid.ncols
    xycoords = grid.cell2coord(np.arange(ncells))
    llongs = xycoords[:, 0]
    llats = xycoords[:, 1]

    # Project to basemap
    bmap = basemap_object
    xcoord, ycoord = bmap(llongs, llats)
    zval = grid.data.copy()

    xcoord = xcoord.reshape(zval.shape)
    ycoord = ycoord.reshape(zval.shape)

    # Clip data to level range
    clevs = config['clevs']
    zval = np.clip(zval,clevs[0], clevs[-1])

    # draw contour
    contf = bmap.contourf(xcoord, ycoord, zval, config.clevs, \
                cmap=config.cmap, \
                norm=config.norm)

    if config.linewidth > 0.:
        bmap.contour(xcoord, ycoord, zval, config.clevs, \
                linewidths=config.linewidth, \
                colors=config.linecolor)

    return contf


def gbar(fig, ax, config, contf, \
    vertical_alignment='center', aspect='auto', \
    legend=None):
    ''' Add color bar to plot '''

    # Check config is proper
    config.is_valid()

    # Create colorbar axe
    div = make_axes_locatable(ax)
    cbar_ax = div.append_axes('right', size='4%', pad=0.)
    cbar_ax.set_aspect(aspect)
    colorb = fig.colorbar(contf, cax=cbar_ax)

    # Ticks and tick labels
    colorb.set_ticks(config.clevs_ticks)
    colorb.ax.set_yticklabels(config.clevs_tick_labels, \
        fontsize=8, \
        va=vertical_alignment)

    # Legend text
    if not legend is None:
        colorb.ax.text(0.0, 1.07, legend,
                size=12, fontsize=8)

    return colorb


