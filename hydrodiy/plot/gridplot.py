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


VARNAMES =  list(hywap.VARIABLES.keys()) + ['effective-rainfall', \
    'decile-rainfall', 'decile-temperature', 'evapotranspiration', \
    'decile-effective-rainfall', 'soil-moisture', 'relative-metric',\
    'bias']


class GridplotConfig(object):
    ''' Class containing gridplot configuration data '''

    def __init__(self, varname=None):
        # Set default values
        self.cmap = plt.cm.RdBu
        self._clevs = None
        self._clevs_ticks = None
        self._clevs_tick_labels = None
        self.norm = None
        self.contour_format = None
        self.contour_fontsize = 8
        self.linewidth = 0.8
        self.linecolor = '#%02x%02x%02x' % (150, 150, 150)
        self.varname = varname
        self.show_ticks = True
        self.legend_title = ''
        self.legend_fontsize = 8

        # Refine default values based on variable name
        self._default_values(varname)

    # setters and getters so that
    # if one defines clevs, clevs_ticks and clevs_tick_labels and norm
    # are defined automatically
    @property
    def clevs(self):
        self.is_valid()
        return self._clevs

    @clevs.setter
    def clevs(self, value):
        self._clevs = np.atleast_1d(value).astype(np.float64)

        # Side effects
        self.clevs_ticks = self._clevs
        self.norm = mpl.colors.Normalize(vmin=self._clevs[0],
                                                vmax=self._clevs[-1])

    @property
    def clevs_ticks(self):
        self.is_valid()
        return self._clevs_ticks

    @clevs_ticks.setter
    def clevs_ticks(self, value):
        self._clevs_ticks = np.atleast_1d(value).astype(np.float64)
        self.clevs_tick_labels = ['{}'.format(lev) for lev \
                                                in self._clevs_ticks]

    @property
    def clevs_tick_labels(self):
        self.is_valid()
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
            raise ValueError('clevs, or clevs_ticks or '+\
                                    'clevs_tick_labels is None')

        if len(self._clevs_tick_labels) != len(self._clevs_ticks):
            raise ValueError('Not the same number of tick levels'+\
                        ' and tick labels')

    def _default_values(self, varname):
        ''' Set default configuration values for specific variables '''

        if varname is None:
            return

        if not varname in VARNAMES:
            raise ValueError(('Variable {0} not in '+\
                '{1}').format(varname, '/'.join(VARNAMES)))

        if varname == 'decile-rainfall':
            self.clevs = [0., 0.01, 0.1, 0.3, 0.7, 0.9, 0.99, 1.0]
            self.clevs_ticks = [0.005, 0.05, 0.2, 0.5, 0.8, 0.95, 0.995]
            self.clevs_tick_labels = ['Lowest\non record', \
                'Very Much\nBelow Average', \
                'Below Average', 'Average', \
                'Above Average', 'Very Much\nAbove Average', \
                'Highest\non record']

            cols = {1.:'#%02x%02x%02x' % (0, 0, 255),
                    0.5:'#%02x%02x%02x' % (255, 255, 255),
                    0.:'#%02x%02x%02x' % (255, 25, 25)}
            self.cmap = putils.colors2cmap(cols)
            self.legend_title = 'Rainfall\ndeciles'
            self.show_ticks = False

        if varname == 'decile-temperature':
            self._default_values('decile-rainfall')

            cols = {1.:'#%02x%02x%02x' % (255, 153, 0),
                    0.5:'#%02x%02x%02x' % (255, 255, 255),
                    0.:'#%02x%02x%02x' % (0, 153, 204)}
            self.cmap = putils.colors2cmap(cols)
            self.legend_title = 'Temperature deciles'

        if varname == 'decile-effective-rainfall':
            self._default_values('decile-rainfall')

            cols = {1.:'#%02x%02x%02x' % (1, 126, 123),
                    0.5:'#%02x%02x%02x' % (254, 254, 228),
                    0.:'#%02x%02x%02x' % (254, 118, 37)}
            self.cmap = putils.colors2cmap(cols)
            self.legend_title = 'Temperature\ndeciles'

        elif varname == 'evapotranspiration':
            clevs = [0, 10, 50, 80, 100, 120, 160, 200, 250, 300, 350]
            self.clevs = clevs
            self.clevs_tick_labels = clevs[:-1] + ['']

            cols = {0.:'#%02x%02x%02x' % (255, 229, 204),
                    1.:'#%02x%02x%02x' % (153, 76, 0)}
            self.cmap = putils.colors2cmap(cols)
            self.legend_title = 'Evapo-\ntranspiration\n[mm]'

        elif varname == 'soil-moisture':
            self.clevs = np.arange(0, 1.05, 0.05)
            self.clevs_ticks = np.arange(0, 1.2, 0.2)
            self.clevs_tick_labels = ['{0:3.0f}'.format(l*100) \
                                                for l in self.clevs_ticks]
            self.cmap = plt.cm.Blues
            self.linewidth = 0.
            self.norm = mpl.colors.Normalize(vmin=self.clevs[0], \
                                vmax=self.clevs[-1])
            self.legend_title = 'Soil Moisture [%]'

        elif varname == 'effective-rainfall':
            clevs = [-200, -100, -75, -50, -25, -10, -5, 0, \
                5, 10, 25, 50, 75, 100, 200]
            self.clevs = clevs
            self.clevs_tick_labels = [''] + clevs[1:-1] + ['']

            cols = {0.:'#%02x%02x%02x' % (255, 76, 0), \
                    0.5:'#%02x%02x%02x' % (255, 255, 255), \
                    1.:'#%02x%02x%02x' % (40, 178, 157)}
            self.cmap = putils.colors2cmap(cols)
            self.linewidth = 0
            self.norm = mpl.colors.SymLogNorm(10., \
                                vmin=clevs[0], vmax=clevs[-1])
            self.legend_title = 'Effective\nRainfall [mm]'

        elif varname == 'rainfall':
            clevs = [0, 1, 5, 10, 25, 50, 100, 200, 300, 400, 600, 800, 1000]
            self.clevs = clevs
            self.cmap = 'Blues'
            self.clevs_tick_labels = clevs[:-1] + ['']

            self.linewidth = 0
            self.norm = mpl.colors.SymLogNorm(10., \
                                vmin=clevs[0], vmax=clevs[-1])
            self.legend_title = 'Rainfall\nTotals [mm]'

        elif varname == 'temperature':
            clevs = [-20] + list(range(-9, 51, 3)) + [60]
            self.clevs = clevs
            self.clevs_tick_labels = clevs[:-1] + ['']

            self.linewidth = 0
            self.cmap = plt.get_cmap('gist_rainbow_r')
            self.legend_title = 'Temperature [C]'

        elif varname == 'vprp':
            self._default_values('temperature')
            clevs = list(range(0, 42, 2))
            self.clevs = clevs
            self.clevs_tick_labels = clevs[:-1] + ['']
            self.legend_title = 'Vapour\nPressure'

        elif varname == 'solar':
            self._default_values('temperature')
            clevs = list(range(0, 43, 3))
            self.clevs = clevs
            self.clevs_tick_labels = clevs[:-1] + ['']
            self.legend_title = 'Solar\nRadiation'

        elif varname == 'relative-metric':
            self.clevs = [-1, -0.5, -0.05, 0.05, 0.5, 1.]
            self.clevs_ticks = [-0.75, -0.255, 0, 0.255, 0.75]
            self.clevs_tick_labels = ['Much lower\n'+\
                            'than benchmark\n(-1, -0.5)', \
                'Lower than\nbenchmark\n(-0.5, -0.05)', \
                'Same than\nbenchmark\n(-0.05, +0.05)', \
                'Higher than\nbenchmark\n(+0.05, +0.5)', \
                'Much higher than\nbenchmark\n(+0.5, +1)']

            self.cmap = 'PiYG'
            self.legend_title = 'Relative\nmetric [-]'
            self.show_ticks = False

        elif varname == 'bias':
            self._default_values('relative-metric')
            self.legend_title = 'Bias [%]'
            self.clevs = [-1, -0.5, -0.1, 0.1, 0.5, 1.]
            self.clevs_ticks = [-0.75, -0.3, 0, 0.3, 0.75]
            self.clevs_tick_labels = ['Large under\n'+\
                            'prediction\n(-100%, -50%)', \
                'Under-prediction\n(-50%, -10%)', \
                'Insignificant\nbias\n(-10%, +10%)', \
                'Over-prediction\n(+10%, +50%)', \
                'Large\nOver-prediction\n(+50%, +100%)']


def gsmooth(grid, mask=None, coastwin=50, sigma=5., \
                                    minval=-np.inf, eps=1e-6):
    ''' Smooth gridded value to improve look of map '''

    smooth = grid.clone(dtype=np.float64)
    z0 = smooth.data

    # Gapfill with nearest neighbour
    ixm, iym = np.where(np.isnan(z0) | (z0<minval-eps))
    ixnm, iynm = np.where(~np.isnan(z0) & (z0>=minval-eps))
    z0[ixm, iym] = -np.inf
    z1 = maximum_filter(z0, size=coastwin, mode='nearest')
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
    ''' Plot gridded data on a basemap object '''

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
    zval = np.clip(zval, config.clevs[0], config.clevs[-1])

    # draw contour plot
    contour_grid = bmap.contourf(xcoord, ycoord, zval, config.clevs, \
                cmap=config.cmap, \
                norm=config.norm)

    # draw contour lines
    contour_lines = None
    if config.linewidth > 0.:
        contour_lines = bmap.contour(xcoord, ycoord, zval, config.clevs, \
                linewidths=config.linewidth, \
                colors=config.linecolor)

        # Set continuous line style
        for line in contour_lines.collections:
            line.set_linestyle('-')

        # Show levels
        if config.contour_format is not None:
            contour_labs = bmap.ax.clabel(contour_lines, config.clevs, \
                            fmt=config.contour_format, \
                            colors='k', \
                            fontsize=config.contour_fontsize)

    return contour_grid, contour_lines


def gbar(cbar_ax, config, contour_grid, **kwargs):
    ''' Draw a color bar associated with a gridplot

        kwargs are passed to matplotlib.colorbar.make_axes
        default values used are:
            fraction = 1.
            pad = 0.1
            aspect = 10
    '''
    # Default kwargs args
    kw = {'fraction': 1., 'pad': 0.1, 'aspect': 10}
    for key, value in kwargs.items():
        kw[key] = kwargs[key]

    # Create colorbar axes within the prescribed axes
    # to allow the use of kwargs
    cbar_ax_inside, kw = mpl.colorbar.make_axes(cbar_ax, **kw)
    colorb = plt.colorbar(contour_grid, cax=cbar_ax_inside)
    cbar_ax.axis('off')

    # Ticks and tick labels
    colorb.set_ticks(config.clevs_ticks)
    colorb.ax.set_yticklabels(config.clevs_tick_labels, \
        fontsize=config.legend_fontsize, \
        va='center')

    # Remove tick marks if needed
    if not config.show_ticks:
        colorb.ax.tick_params(axis='y', which='both', length=0)

    # Legend text
    colorb.ax.text(0.0, 1.07, config.legend_title,
                size=12, fontsize=config.legend_fontsize+1)

    return colorb


