''' Objects used to download data from  AWAP '''

import re
import os
import datetime

import urllib2

from dateutil.relativedelta import relativedelta as delta

import tempfile

import itertools

from subprocess import Popen, PIPE

import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

HAS_BASEMAP = True

try:
    from mpl_toolkits.basemap import cm as cm
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mpl.rcParams['contour.negative_linestyle'] = 'solid'

except ImportError:
    HAS_BASEMAP = False

from hydrodiy.io import csv

VARIABLES = {
    'rainfall':[{'type':'totals', 'unit':'mm/d'}],
    'temperature':[{'type':'maxave', 'unit':'celsius'},
                   {'type':'minave', 'unit':'celsius'}],
    'vprp':[{'type':'vprph09', 'unit':'Pa'}],
    'solar':[{'type':'solarave', 'unit':'MJ/m2'}]
}

TIMESTEPS = ['day', 'month']

AWAP_URL='http://www.bom.gov.au/web03/ncc/www/awap'


def get_cellcoords(header):
    ''' Get coordinates and cell number of gridded data '''

    nrows = header['nrows']
    ncols = header['ncols']
    xll = header['xllcorner']
    yll = header['yllcorner']
    csz = header['cellsize']

    longs = xll + csz * np.arange(0, ncols)
    lats = yll + csz * np.arange(0, nrows)

    # We have to flip the lats
    llongs, llats = np.meshgrid(longs, lats[::-1])

    cellids = np.array(['%0.2f_%0.2f' % (x, y) \
                    for x, y in zip(llongs.flat[:], \
                        llats.flat[:])]).reshape(llongs.shape)

    return cellids, llongs, llats


def get_plotconfig(cfg, varname):
    ''' Generate default plotting configuration '''

    if cfg is None:
        cfg = {'cmap':None, \
            'clevs':None, \
            'norm':None, \
            'linewidth':1., \
            'linecolor':'#%02x%02x%02x' % (60, 60, 60)}

    if varname == 'rainfall':
        if cfg['clevs'] is None:
            cfg['clevs'] = [0, 1, 5, 10, 15, 25, 50, \
                                    100, 150, 200, 300, 400]

        if cfg['cmap'] is None:
            cfg['cmap'] = cm.s3pcpn

    if varname == 'temperature':
        if cfg['clevs'] is None:
            cfg['clevs'] = range(-9, 51, 3)

        if cfg['cmap'] is None:
            cfg['cmap'] = plt.get_cmap('gist_rainbow_r')

    if varname == 'vprp':
        if cfg['clevs'] is None:
            cfg['clevs'] = range(0, 40, 2)

        if cfg['cmap'] is None:
            cfg['cmap'] = plt.get_cmap('gist_rainbow_r')

    if varname == 'solar':
        if cfg['clevs'] is None:
            cfg['clevs'] = range(0, 40, 3)

        if cfg['cmap'] is None:
            cfg['cmap'] = plt.get_cmap('jet_r')

    if cfg['norm'] is None:
        cfg['norm'] = plt.cm.colors.Normalize( \
                vmin=np.min(cfg['clevs']), \
                vmax=np.max(cfg['clevs']))

    return cfg


class HyWap(object):
    ''' Class to download daily awap grids '''

    def __init__(self):

        self.current_url = None


    def get_data(self, varname, vartype, timestep, date):
        ''' Download gridded awap daily data '''

        # Check variable
        if not varname in VARIABLES:
            raise ValueError(('varname(%s) not'+ \
                ' recognised (should be %s)') % (varname, \
                    ', '.join(VARIABLES.keys())))

        vartypes = [v['type'] for v in VARIABLES[varname]]
        if not vartype in vartypes:
            raise ValueError(('vartype(%s) not'+ \
                ' recognised (should be %s)') % (vartype, \
                    ', '.join(vartypes)))

        if not timestep in TIMESTEPS:
            raise ValueError(('timestep(%s) not'+ \
                ' recognised (should be %s)') % (varname, \
                    ', '.join(TIMESTEPS)))

        # Define start and end date of period
        dt1 = datetime.datetime.strptime(date, '%Y-%m-%d')

        if (timestep == 'month') & (dt1.day != 1):
            raise ValueError(('Invalide date(%s). '+ \
                'Should be on day 1 of the month') % dt1.date())

        if timestep == 'day':
            timestep = 'daily'
            dt2 = dt1

        if timestep == 'month':
            dt2 = dt1 + delta(months=1) - delta(days=1)

        # Download data
        self.current_url = ('%s/%s/%s/%s/grid/0.05/history/nat/'+ \
                '%4d%2.2d%2.2d%4d%2.2d%2.2d.grid.Z') % (AWAP_URL, \
                    varname, vartype, timestep, \
                    dt1.year, dt1.month, dt1.day, \
                    dt2.year, dt2.month, dt2.day)

        try:
            resp = urllib2.urlopen(self.current_url)

        except urllib2.HTTPError as ehttp:
            print('Cannot download %s: HTTP Error = %s' % (self.current_url, \
                            ehttp))
            raise ehttp

        # Read data from pipe and write it to disk
        zdata = resp.read()

        adir = tempfile.gettempdir()
        ftmp = os.path.join(adir, 'tmp.Z')
        with open(ftmp, 'wb') as fobj:
            fobj.write(zdata)

        # Extract data from compressed file
        # (Unix compress format produced with the 'compress' unix command
        compressedfile = Popen(['zcat', ftmp], stdout=PIPE).stdout
        txt = compressedfile.readlines()
        compressedfile.close()
        try:
            os.remove(ftmp)
            os.remove(re.sub('\\.Z$', '', ftmp))
        except OSError:
            pass

        # Spot header / comments
        tmp = [bool(re.search('([a-zA-Z]|\\[)', l[0])) for l in txt]
        iheader = np.argmin(tmp)
        icomment = np.argmin(tmp[::-1])

        # Process grid
        header = {k:float(v) \
            for k, v in [re.split(' +', s.strip()) for s in txt[:iheader]]}

        header['varname'] = varname
        header['vartype'] = vartype
        header['date'] = date
        header['url'] = self.current_url

        # Reformat header
        header['ncols'] = int(header['ncols'])
        header['nrows'] = int(header['nrows'])
        header['cellsize'] = float(header['cellsize'])

        header['xllcorner'] = float(header['xllcenter'])
        header.pop('xllcenter')
        header['yllcorner'] = float(header['yllcenter'])
        header.pop('yllcenter')

        # Get meta
        meta = [s.strip() for s in txt[-icomment:]]

        # Get data
        data = [np.array(re.split(' +', s.strip())) \
                    for s in txt[iheader:-icomment]]
        data = np.array(data).astype(np.float)
        data[data == header['nodata_value']] = np.nan

        # Check dimensions of dataset
        ncols = header['ncols']
        nrows = header['nrows']

        if data.shape != (nrows, ncols):
            raise IOError(('Dataset dimensions (%d,%d)'+ \
                ' do not match header (%d,%d)') % (data.shape[0], \
                    data.shape[1], nrows, ncols))

        # Build comments
        comment = ['AWAP Data set downloaded from ' + AWAP_URL, '', '',]
        comment += meta
        comment += ['']

        return data, comment, header

    def plot(self, data, header, basemap_object, config=None):
        ''' Plot gridded data '''

        if not HAS_BASEMAP:
            raise ImportError('basemap is not available')

        cfg = get_plotconfig(config, header['varname'])

        cellnum, llongs, llats, = get_cellcoords(header)

        m = basemap_object.get_map()
        x, y = m(llongs, llats)
        z = data.copy()

        # Filter data
        clevs = cfg['clevs']
        ix, iy = np.where(z < clevs[0])
        z[ix, iy] = np.nan

        ix, iy = np.where(z > clevs[-1])
        z[ix, iy] = np.nan

        if 'sigma' in cfg:
            z = gaussian_filter(z, sigma=cfg['sigma'], mode='nearest')

        # Refine levels
        if np.nanmax(z) < np.max(clevs):
            iw = np.min(np.where(np.nanmax(z) < np.sort(clevs))[0])
            clevs = clevs[:iw+1]

        # draw contour
        cs = m.contourf(x, y, z, cfg['clevs'], \
                    cmap=cfg['cmap'], \
                    norm=cfg['norm'])

        if cfg['linewidth'] > 0.:
            m.contour(x, y, z, cfg['clevs'], \
                    linewidths=cfg['linewidth'], \
                    colors=cfg['linecolor'])

        return cs


def plot_cbar(fig, ax, cs, *args, **kwargs):

    div = make_axes_locatable(ax)

    cbar_ax = div.append_axes(*args, **kwargs)

    cb = fig.colorbar(cs, cax=cbar_ax)


