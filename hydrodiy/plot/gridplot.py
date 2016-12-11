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


def get_gconfig(varname):
    ''' Generate grid plotting configuration '''

    cfg = {'cmap':None, \
            'clevs':None, \
            'clevs_ticks':None, \
            'clevs_tick_labels': None, \
            'norm':None, \
            'linewidth':0.8, \
            'linecolor':'#%02x%02x%02x' % (150, 150, 150)}

    if not varname in VARNAMES:
        raise ValueError(('Variable {0} not in '+\
            '{1}').format(varname, '/'.join(VARNAMES)))

    if varname == 'decile-rain':
        clevs = [0., 0.1, 0.3, 0.7, 0.9, 1.0]
        clevs_tick_labels = ['', '\nVery Much\nBelow Average', '\nBelow Average', '\nAverage',
                    '\nAbove Average', '\nVery Much\nAbove Average']
        cfg['clevs'] = clevs
        cfg['clevs_ticks'] = clevs
        cfg['clevs_tick_labels'] = clevs_tick_labels

        cols = {1.:'#%02x%02x%02x' % (102, 102, 255),
                0.5:'#%02x%02x%02x' % (255, 255, 255),
                0.:'#%02x%02x%02x' % (255, 102, 102)}
        cfg['cmap'] = putils.col2cmap(cols)
        cfg['norm'] = mpl.colors.Normalize(vmin=clevs[0], vmax=clevs[-1])

    if varname == 'decile-temp':
        cfg = get_gconfig('decile-rain')

        cols = {1.:'#%02x%02x%02x' % (255, 153, 0),
                0.5:'#%02x%02x%02x' % (255, 255, 255),
                0.:'#%02x%02x%02x' % (0, 153, 204)}
        cfg['cmap'] = putils.col2cmap(cols)

    elif varname == 'evapotranspiration':
        clevs = [0, 10, 50, 80, 100, 120, 160, 200, 250, 300]
        cfg['clevs'] = clevs
        cfg['clevs_ticks'] = clevs
        cfg['clevs_tick_labels'] = clevs

        cols = {0.:'#%02x%02x%02x' % (255, 229, 204),
                1.:'#%02x%02x%02x' % (153, 76, 0)}
        cfg['cmap'] = putils.col2cmap(cols)
        cfg['norm'] = mpl.colors.Normalize(vmin=cfg['clevs'][0], vmax=cfg['clevs'][-1])

    elif varname == 'soil-moisture':
        clevs = np.arange(0, 1.05, 0.05)
        clevs_ticks = np.arange(0, 1.2, 0.2)
        cfg['clevs'] = clevs
        cfg['clevs_ticks'] = clevs_ticks
        cfg['clevs_tick_labels'] = ['{0:3.0f}%'.format(l*100) \
                                            for l in clevs_ticks]
        cfg['cmap'] = plt.cm.Blues
        cfg['linewidth'] = 0.
        cfg['norm'] = mpl.colors.Normalize(vmin=clevs[0], vmax=clevs[-1])

    elif varname == 'effective-rainfall':
        clevs = [-100, -75, -50, -25, -10, -5, 0, \
            5, 10, 25, 50, 75, 100]
        cfg['clevs'] = clevs
        cfg['clevs_ticks'] = clevs
        cfg['clevs_tick_labels'] = clevs

        cols = {0.:'#%02x%02x%02x' % (255, 76, 0), \
                0.5:'#%02x%02x%02x' % (255, 255, 255), \
                1.:'#%02x%02x%02x' % (40, 178, 157)}
        cmap = putils.col2cmap(cols)
        cfg['cmap'] = cmap
        cfg['linewidth'] = 0
        cfg['norm'] = mpl.colors.SymLogNorm(10., vmin=clevs[0], vmax=clevs[-1])

    elif varname == 'rainfall':
        clevs = [0, 1, 5, 10, 25, 50, 100, 200, 300, 400, 600, 800]
        cfg['clevs'] = clevs
        cfg['clevs_ticks'] = clevs
        cfg['clevs_tick_labels'] = clevs

        cfg['cmap'] = 'RdBu'
        cfg['linewidth'] = 0
        cfg['norm'] = mpl.colors.SymLogNorm(10., vmin=clevs[0], vmax=clevs[-1])

    elif varname == 'temperature':
        clevs = range(-9, 51, 3)
        cfg['clevs'] = clevs
        cfg['clevs_ticks'] = clevs
        cfg['clevs_tick_labels'] = clevs

        cfg['cmap'] = plt.get_cmap('gist_rainbow_r')
        cfg['linewidth'] = 0
        cfg['norm'] = mpl.colors.SymLogNorm(10., vmin=clevs[0], vmax=clevs[-1])

    elif varname == 'vprp':
        cfg = get_gconfig('temperature')
        clevs = range(0, 40, 2)
        cfg['clevs'] = clevs
        cfg['clevs_ticks'] = clevs
        cfg['clevs_tick_labels'] = clevs

    elif varname == 'solar':
        cfg = get_gconfig('temperature')
        clevs = range(0, 40, 3)
        cfg['clevs'] = clevs
        cfg['clevs_ticks'] = clevs
        cfg['clevs_tick_labels'] = clevs

    return cfg


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
        # Check grid and mas have the same geometry
        if not grid.same_geometry(mask):
            raise ValueError('Mask does not have the same '+
                'geometry than input grid')

        ixm, iym = np.where(mask.data == 0)
        z2[ixm, iym] = np.nan
        #ixnm, iynm = np.where(mask.data == 1)

        #z2[ixm, iym] = -np.inf
        #z3 = maximum_filter(z2, size=50, mode='nearest')

        #z3[ixnm, iynm] = 0.0
        #z2[ixm, iym] = 0.0

        #z4 = z2 + z3

    smooth.data = z2

    return smooth



def gplot(grid, basemap_object, config):
    ''' Plot gridded data '''

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

    # Filter data
    clevs = config['clevs']
    idx_x, idx_y = np.where(zval < clevs[0])
    zval[idx_x, idx_y] = np.nan

    idx_x, idx_y = np.where(zval > clevs[-1])
    zval[idx_x, idx_y] = np.nan

    # Refine levels
    if np.nanmax(zval) < np.max(clevs):
        idx_z = np.min(np.where(np.nanmax(zval) < np.sort(clevs))[0])
        clevs = clevs[:idx_z+1]

    # draw contour
    contf = bmap.contourf(xcoord, ycoord, zval, config['clevs'], \
                cmap=config['cmap'], \
                norm=config['norm'])

    if config['linewidth'] > 0.:
        bmap.contour(xcoord, ycoord, zval, config['clevs'], \
                linewidths=config['linewidth'], \
                colors=config['linecolor'])

    return contf


def gplot_colorbar(fig, ax, cfg, contf, \
    vertical_alignment='center', aspect='auto', \
    legend=None):
    ''' Add color bar to plot '''

    # Create colorbar axe
    div = make_axes_locatable(ax)
    cbar_ax = div.append_axes('right', size='4%', pad=0.)
    cbar_ax.set_aspect(aspect)
    colorb = fig.colorbar(contf, cax=cbar_ax)

    # Ticks and tick labels
    clevs_ticks = cfg['clevs_ticks']
    clevs_tick_labels = cfg['clevs_tick_labels']
    colorb.set_ticks(clevs_ticks)
    colorb.ax.set_yticklabels(clevs_tick_labels, fontsize=8, \
        va=vertical_aligmnent)

    # Legend text
    if not legend is None:
        cb.ax.text(0.0, 1.07, legend,
                size=12, fontsize=8)
    return cb


