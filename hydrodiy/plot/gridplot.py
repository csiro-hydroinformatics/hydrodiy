''' Objects used to download data from  AWAP '''

import re
import os
import datetime
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

HAS_BASEMAP = False
try:
    from mpl_toolkits import basemap
    HAS_BASEMAP = True
except (ImportError, FileNotFoundError) as err:
    pass

from hydrodiy.data.hywap import VARIABLES
from  hydrodiy.plot import putils

class HYPlotGridplotError(Exception):
    pass


VARNAMES =  list(VARIABLES.keys()) + ['effective-rainfall', \
    'decile-rainfall', 'decile-temperature', 'evapotranspiration', \
    'decile-effective-rainfall', 'soil-moisture', 'relative-metric',\
    'relative-bias']

def vect2txt(x):
    text = ', '.join(['{:0.2f}'.format(v) for v in x])
    return text


class GridplotConfig(object):
    ''' Class containing gridplot configuration data '''

    def __init__(self, varname=None):
        # Set default values
        self.cmap = plt.cm.RdBu
        self._clevs = None
        self._clevs_contour = None
        self._clevs_ticks = None
        self._clevs_tick_labels = None
        self.norm = None
        self.contour_text_format = '%0.1f'
        self.contour_text_color = 'k'
        self.contour_fontsize = 8
        self.contour_linewidth = 0.8
        self.contour_linecolor = '#%02x%02x%02x' % (150, 150, 150)
        self.show_ticks = True
        self.legend_title = ''
        self.legend_fontsize = 8

        # Refine default values based on variable name
        self._default_values(varname)


    def __str__(self):
        ''' Pretty print of config object '''
        text =  ' clevs: {}\n'.format(vect2txt(self.clevs))
        text += ' clevs_contour: {}\n'.format(vect2txt(self.clevs_contour))
        text += ' clevs_ticks: {}\n'.format(vect2txt(self.clevs_ticks))
        text += ' norm: {}\n'.format(self.norm.__class__.__name__)

        for attr in ['cmap', 'contour_text_format', 'contour_fontsize', \
                        'contour_linewidth', 'contour_linecolor', \
                        'show_ticks', 'legend_title', 'legend_fontsize']:
            t = re.sub('\n', ' ', '{}'.format(getattr(self, attr)))
            text += ' {}: {}\n'.format(attr, t)

        return text

    # setters and getters so that
    # if one defines clevs, clevs_ticks and clevs_tick_labels and norm
    # are defined automatically
    @property
    def clevs(self):
        self.is_valid()
        return self._clevs

    @clevs.setter
    def clevs(self, value):
        self._clevs = np.sort(np.atleast_1d(value).astype(np.float64))

        # Side effects
        self.clevs_ticks = self._clevs
        self.clevs_contour = self._clevs

        if self.norm is None:
            self.norm = mpl.colors.Normalize(vmin=self._clevs[0],
                                                vmax=self._clevs[-1])
        else:
            self.norm.vmin = self._clevs[0]
            self.norm.vmax = self._clevs[-1]


    @property
    def clevs_contour(self):
        self.is_valid()
        return self._clevs_contour

    @clevs_contour.setter
    def clevs_contour(self, value):
        # No clevs
        self._clevs_contour = None

        # Check object has len
        try:
            if len(value) > 0:
                value = np.sort(np.atleast_1d(value).astype(np.float64))

                # Check range
                if value[0] < self.clevs[0]:
                    raise ValueError('Expected min(clevs_contour) '+\
                        '>= {}, got {}'.format(self.clevs[0], value[0]))

                if value[-1] > self.clevs[-1]:
                    raise ValueError('Expected max(clevs_contour) '+\
                        '<= {}, got {}'.format(self.clevs[-1], value[-1]))

                self._clevs_contour = value
        except TypeError:
            pass

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
            self.clevs_contour = [0.1, 0.9]

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
            self.legend_title = 'Effective rainfall\ndeciles'

        elif varname == 'evapotranspiration':
            clevs = [0, 10, 50, 80, 100, 120, 160, 200, 250, 300, 350]
            self.clevs = clevs
            self.clevs_contour = [50, 100, 200]
            self.contour_text_format = '%0.0f'
            self.clevs_tick_labels = clevs[:-1] + ['']

            cols = {0.:'#%02x%02x%02x' % (255, 229, 204),
                    1.:'#%02x%02x%02x' % (153, 76, 0)}
            self.cmap = putils.colors2cmap(cols)
            self.legend_title = 'Evapo-\ntranspiration\n[mm]'

        elif varname == 'soil-moisture':
            self.clevs = np.arange(0, 1.05, 0.05)
            self.clevs_contour = [0.1, 0.9]
            self.clevs_ticks = np.arange(0, 1.2, 0.2)
            self.clevs_tick_labels = ['{0:3.0f}'.format(l*100) \
                                                for l in self.clevs_ticks]
            self.cmap = plt.cm.Blues
            self.contour_linewidth = 0.
            self.norm = mpl.colors.Normalize(vmin=self.clevs[0], \
                                vmax=self.clevs[-1])
            self.legend_title = 'Soil Moisture\n[%]'

        elif varname == 'effective-rainfall':
            clevs = [-200, -100, -75, -50, -25, -10, -5, \
                5, 10, 25, 50, 75, 100, 200]
            self.clevs = clevs
            self.clevs_contour = [-100, -10, 10, 100]
            self.contour_text_format = '%0.0f'
            self.clevs_tick_labels = [''] + clevs[1:-1] + ['']

            cols = {0.:'#%02x%02x%02x' % (255, 76, 0), \
                    0.5:'#%02x%02x%02x' % (255, 255, 255), \
                    1.:'#%02x%02x%02x' % (40, 178, 157)}
            self.cmap = putils.colors2cmap(cols)
            self.contour_linewidth = 0
            self.norm = mpl.colors.SymLogNorm(10., \
                                vmin=clevs[0], vmax=clevs[-1])
            self.legend_title = 'Effective\nRainfall [mm]'

        elif varname == 'rainfall':
            clevs = [0, 1, 5, 10, 25, 50, 100, 200, 300, 400, 600, 800, 1000]
            self.clevs = clevs
            self.clevs_contour = [1, 10, 100, 200]
            self.contour_text_format = '%0.0f'
            self.cmap = 'Blues'
            self.clevs_tick_labels = clevs[:-1] + ['']

            self.contour_linewidth = 0
            self.norm = mpl.colors.SymLogNorm(10., \
                                vmin=clevs[0], vmax=clevs[-1])
            self.legend_title = 'Rainfall\nTotals [mm]'

        elif varname == 'temperature':
            clevs = [-20] + list(range(-9, 51, 3)) + [60]
            self.clevs = clevs
            self.clevs_contour = [0, 10, 20, 30]
            self.clevs_tick_labels = clevs[:-1] + ['']

            self.contour_linewidth = 0
            self.cmap = plt.get_cmap('gist_rainbow_r')
            self.legend_title = 'Temperature [C]'

        elif varname == 'vprp':
            self._default_values('temperature')
            clevs = list(range(0, 42, 2))
            self.clevs = clevs
            self.clevs_contour = [0, 10, 20, 30]
            self.clevs_tick_labels = clevs[:-1] + ['']
            self.legend_title = 'Vapour\nPressure'

        elif varname == 'solar':
            self._default_values('temperature')
            clevs = list(range(0, 43, 3))
            self.clevs = clevs
            self.clevs_contour = [0, 10, 20, 30]
            self.clevs_tick_labels = clevs[:-1] + ['']
            self.legend_title = 'Solar\nRadiation'

        elif varname == 'relative-metric':
            self.clevs = [-1, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1.]
            self.clevs_contour = [-0.2, 0, 0.2]
            self.clevs_ticks = [-0.75, -0.35, -0.125, 0, 0.125, 0.35, 0.75]
            self.clevs_tick_labels = [\
                'Very much lower\n than benchmark\n(-1, -0.5)', \
                'Much lower\n than benchmark\n(-0.5, -0.2)', \
                'Lower than\nbenchmark\n(-0.2, -0.05)', \
                'Same than\nbenchmark\n(-0.05, +0.05)', \
                'Higher than\nbenchmark\n(+0.05, +0.2)', \
                'Much higher than\nbenchmark\n(+0.2, +0.5)',\
                'Very much higher than\nbenchmark\n(+0.5, +1)']

            self.cmap = 'PiYG'
            self.legend_title = 'Relative\nmetric [-]'
            self.norm = mpl.colors.SymLogNorm(linthresh=0.1, \
                                vmin=self.clevs[0], vmax=self.clevs[-1])
            self.show_ticks = False

        elif varname == 'relative-bias':
            self._default_values('relative-metric')
            self.legend_title = 'Relative\nBias [%]'
            self.clevs = [-1, -0.5, -0.1, 0.1, 0.5, 1.]
            self.clevs_contour = [-0.5, 0, 0.5]
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
    # Check coastwin param
    if coastwin < 10*sigma:
        warnings.warn('Expected coastwin > 10 x sigma, got '+
                'coastwin={:0.1f} sigma={:0.1f}'.format(coastwin, \
                    sigma))

    # Set up smooth grid
    smooth = grid.clone(dtype=np.float64)
    z0 = grid.data.copy()

    # Cut mask
    if not mask is None:
        # Check grid and mask have the same geometry
        if not grid.same_geometry(mask):
            raise ValueError('Mask does not have the same '+
                'geometry than input grid')

        # Check mask as integer dtype
        if not issubclass(mask.data.dtype.type, np.integer):
            raise ValueError('Expected integer dtype for mask, '+\
                        'got {}'.format(mask.dtype))

        # Set mask
        z0[mask.data == 0] = np.nan

    # Gapfill with nearest neighbour
    idx = np.isnan(z0) | (z0<minval-eps)
    z0[idx] = -np.inf

    z1 = maximum_filter(z0, size=coastwin, mode='nearest')
    z1[~idx] = z0[~idx]

    # Smooth
    z2 = gaussian_filter(z1, sigma=sigma, mode='nearest')

    # Final mask cut
    if not mask is None:
        z2[mask.data == 0] = np.nan

    # Store
    smooth.data = z2

    return smooth



def gplot(grid, config, plotting_object, proj=None):
    ''' Plot gridded data on a basemap object

    Plot data on Australia map

    Parameters
    -----------
    grid : hydrodiy.gis.grid.Grid
        Grid data to be plotted
    config : hydrodiy.plot.gridplot.GridplotConfig
        Plotting configuration.
    plotting_obect : mpl_toolkits.basemap.map  or
                        matplotlib.axes.Axes
        Plotting object to be used. If an axes is used, data
        is plotted without projection.
    proj : pyproj.projProj
        Map projection. Example with transform to GDA94:
        proj = pyproj.Proj('+init=EPSG:3112')

    Returns
    -----------
    contour_grid : matplotlib.contour.QuadContourSet
        Contour grid.
    contour_lines : matplotlib.contour.QuadContourSet
        Contour lines.
    xcoord : numpy.ndarray
        Projected X coordinates.

    ycoord : numpy.ndarray
        Projected Y coordinates.
    '''

    # Get cell coordinates
    ncells = grid.nrows*grid.ncols
    xycoords = grid.cell2coord(np.arange(ncells))
    llongs = xycoords[:, 0]
    llats = xycoords[:, 1]

    # Setup plotting objects
    if not isinstance(plotting_object, mpl.axes.Axes):
        plotobj = plotting_object.map
        ax = plotobj.ax
        # Project to basemap
        xcoord, ycoord = plotobj(llongs, llats)
    else:
        # If not basemap object provided, then use raw coordinates
        plotobj = plotting_object
        ax = plotobj
        if not proj is None:
            # Convert coordinates to projection
            coords = [proj(xx, yy) for xx, yy in zip(llongs, llats)]
            xcoord, ycoord = np.array(coords).T
        else:
            xcoord, ycoord = llongs, llats

    zval = grid.data.copy()
    xcoord = xcoord.reshape(zval.shape)
    ycoord = ycoord.reshape(zval.shape)

    # Clip data to level range
    zval = np.clip(zval, config.clevs[0], config.clevs[-1])

    # draw contour plot
    contour_grid = plotobj.contourf(xcoord, ycoord, zval, config.clevs, \
                cmap=config.cmap, \
                norm=config.norm)

    # draw contour lines
    contour_lines = None
    if config.contour_linewidth > 0.:
        contour_lines = plotobj.contour(xcoord, ycoord, zval, \
                config.clevs_contour, \
                contour_linewidths=config.contour_linewidth, \
                colors=config.contour_linecolor)

        # Set continuous line style
        for line in contour_lines.collections:
            line.set_linestyle('-')

        # Show levels
        if (config.contour_text_format is not None) and \
                    (config.clevs_contour is not None):
            try:
                contour_labs = ax.clabel(contour_lines, \
                        config.clevs_contour, \
                        fmt=config.contour_text_format, \
                        colors=config.contour_text_color, \
                        fontsize=config.contour_fontsize)
            except ValueError as err:
                warnings.warn('Cannot draw contour lines: {}'.format(\
                                    str(err)))

    return contour_grid, contour_lines, xcoord, ycoord


def gbar(cbar_ax, config, contour_grid, rect=[0, 0, 0.6, 0.95], \
                delta=0.01, draw_ticks=True):
    ''' Draw a color bar associated with a gridplot

    '''
    # Check argument
    config.is_valid()

    clevs = config.clevs
    clevs_ticks = config.clevs_ticks

    # Plotting variables
    nlevs = len(clevs)
    xx = np.repeat(np.array([rect[0], rect[2]])[None, :], \
                        nlevs, axis=0)
    ylevs = np.linspace(rect[1], rect[3], nlevs)
    yylevs = np.column_stack([ylevs, ylevs])
    zzlevs = (clevs[1:]+clevs[:-1])/2

    # Plot color bar
    cbar_ax.pcolor(xx, yylevs, zzlevs[:, None], cmap=config.cmap, \
                        norm=config.norm)

    cbar_ax.plot([rect[0], rect[0], rect[2], rect[2], rect[0]], \
                [rect[1], rect[3], rect[3], rect[1], rect[1]], 'k-', lw=0.5)

    # Set labels
    yticks = np.interp(clevs_ticks, clevs, ylevs)
    dx = (rect[2]-rect[0])/10
    for lab, pos in zip(config.clevs_tick_labels, yticks):
        # Tick label
        cbar_ax.text(rect[2]+dx, pos, lab, \
                    ha='left', va='center', \
                    fontsize=config.legend_fontsize)
        # Tick mark
        if draw_ticks:
            cbar_ax.plot([rect[2]-dx, rect[2]], [pos]*2, 'k-', \
                            lw=0.5)

    # Legend
    title = cbar_ax.text(rect[0]+delta, \
                rect[3]+delta, \
                config.legend_title, \
                va='bottom', ha='left', \
                fontsize=config.legend_fontsize)

    # Final
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks([])
    cbar_ax.set_xlim([-delta/2, 1+delta/2])
    cbar_ax.set_ylim([-delta/2, 1+delta/2])
    cbar_ax.axis('off')
