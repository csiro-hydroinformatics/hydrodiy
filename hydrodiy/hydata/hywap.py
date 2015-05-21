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

has_basemap = True
try:
    from mpl_toolkits.basemap import cm as cm
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from hygis import oz

except ImportError:
    has_basemap = False

from hyio import csv

class HyWap():
    ''' Class to download daily awap grids '''

    def __init__(self, awap_url ='http://www.bom.gov.au/web03/ncc/www/awap'):

        self.awap_url = awap_url

        self.awap_dir = None

        self.variables = {
            'rainfall':[{'type':'totals', 'unit':'mm/d'}],
            'temperature':[{'type':'maxave','unit':'celsius'}, 
                           {'type':'minave','unit':'celsius'}], 
            'vprp':[{'type':'vprph09', 'unit':'Pa'}],
            'solar':[{'type':'solarave','unit':'MJ/m2'}]
           }

        self.timesteps = ['daily', 'month']

        self.current_url = None

    def set_awapdir(self, awap_dir):
        ''' Set AWAP output directory ''' 

        self.awap_dir = awap_dir

        for varname in self.variables.keys():

            # Create output directories
            F = os.path.join(awap_dir, varname)

            if not os.path.exists(F):
                os.mkdir(F)

            # Create output directories for each timestep
            for timestep in ['daily', 'month']:
                F = os.path.join(awap_dir, varname, timestep)
                if not os.path.exists(F):
                    os.mkdir(F)

    def getgriddata(self, varname, vartype, timestep, date):
        ''' Download gridded awap daily data '''

        # Check variable
        if not (varname in self.variables):
            raise ValueError('varname(%s) not recognised (should be %s)' % (varname,
                ', '.join(self.variables.keys())))
           
        vt = [v['type'] for v in self.variables[varname]]
        if not (vartype in vt):
            raise ValueError('vartype(%s) not recognised (should be %s)' % (vartype,
                ', '.join(vt)))
           
        if not (timestep in self.timesteps):
            raise ValueError('timestep(%s) not recognised (should be %s)' % (varname,
                ', '.join(self.timesteps)))

        # Define start and end date of period
        dt1 = datetime.datetime.strptime(date, '%Y-%m-%d')

        if (timestep == 'month') & (dt1.day != 1):
            raise ValueError(('Invalide date(%s). '
                'Should be on day 1 of the month') % dt1.date())

        dt2 = dt1
        if timestep == 'month':
            dt2 = dt1 + delta(months=1) - delta(days=1)

        # Download data
        self.current_url = ('%s/%s/%s/%s/grid/0.05/history/nat/'
                '%4d%2.2d%2.2d%4d%2.2d%2.2d.grid.Z') % (self.awap_url, 
                    varname, vartype, timestep, 
                    dt1.year, dt1.month, dt1.day,
                    dt2.year, dt2.month, dt2.day)
        
        try:
            resp = urllib2.urlopen(self.current_url)

        except urllib2.HTTPError, e:
            print('Cannot download %s: HTTP Error = %s' % (self.current_url, e))
            raise e

        # Read data from pipe and write it to disk
        zdata = resp.read()

        F = self.awap_dir
        if F is None:
            F = tempfile.gettempdir()

        ftmp = os.path.join(F, 'tmp.Z')
        with open(ftmp, 'wb') as fobj:
            fobj.write(zdata)

        # Extract data from compressed file
        # (Unix compress format produced with the 'compress' unix command
        f = Popen(['zcat', ftmp], stdout=PIPE).stdout
        txt = f.readlines()
        f.close()
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
        header = {k:float(v) 
            for k,v in [re.split(' +', s.strip()) for s in txt[:iheader]]}

        header['varname'] = varname
        header['vartype'] = vartype
        header['date'] = date
        header['url'] = self.current_url

        meta = [s.strip() for s in txt[-icomment:]]

        data = [np.array(re.split(' +', s.strip())) for s in txt[iheader:-icomment]]
        data = np.array(data).astype(np.float)
        data[data == header['nodata_value']] = np.nan

        # Check dimensions of dataset
        nc = int(header['ncols'])
        nr = int(header['nrows'])

        if data.shape != (nr, nc):
            import pdb; pdb.set_trace()

            raise IOError(('Dataset dimensions (%d,%d)'
                ' do not match header (%d,%d)' % (data.shape[0], 
                data.shape[1], nr, nc)))

        # Build comments
        comment = ['AWAP Data set downloaded from %s' % self.awap_url, '', '',]
        comment += meta
        comment += ['']

        return data, comment, header

    def getcoords(self, header):
        ''' Get coordinates and cell number of gridded data '''

        nrows = int(header['nrows'])
        ncols = int(header['ncols'])
        xll = float(header['xllcenter'])
        yll = float(header['yllcenter'])
        sz = float(header['cellsize'])

        longs = xll + sz * np.arange(0, ncols)
        lats = yll + sz * np.arange(0, nrows)

        # We have to flip the lats
        llongs, llats = np.meshgrid(longs, lats[::-1])

        cellids = np.array(['%0.2f_%0.2f' % (x,y) for x,y in zip(llongs.flat[:],
                            llats.flat[:])]).reshape(llongs.shape)

        return cellids, llongs, llats
        
    
    def savegriddata(self, varname, vartype, timestep, dt):
        ''' Download gridded data and save it to disk '''

        data, comment, header = self.getgriddata(varname, vartype, timestep, dt)

        data = pd.DataFrame(data)
        data.columns = ['c%3.3d' % i for i in range(data.shape[1])]

        F = self.awap_dir
        if F is None:
            raise ValueError('Cannot write data, awap dir is not setup')

        fout = os.path.join(F, varname, timestep, '%s_%s_%s_%s.csv' % (varname, 
                                timestep, vartype, dt))

        co = comment + [''] + ['%s:%s' % (k, header[k]) for k in header] + ['']
        csv.write_csv(data, fout, co)

        return '%s.gz' % fout

    def default_plotconfig(self, cfg, varname):
        ''' Generate default plotting configuration '''

        if cfg is None:
            cfg = {'cmap':None, 'clevs':None, 'norm':None}

        if varname == 'rainfall':
            if cfg['clevs'] is None:
                cfg['clevs'] = [0, 1, 5, 10, 15, 25, 50, 
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
            cfg['norm'] = plt.cm.colors.Normalize(
                    vmin=np.min(cfg['clevs']), 
                    vmax=np.max(cfg['clevs']))

        return cfg 

    def plot(self, data, header, ax, 
            config = None,
            coast={'linestyle':'-'}, 
            states={'linestyle':'--'}):
        ''' Plot gridded data '''

        if not has_basemap:
            raise ImportError('basemap is not available')

        cfg = self.default_plotconfig(config, header['varname'])

        cellnum, llongs, llats, = self.getcoords(header)

        om = oz.Oz(ax = ax)

        if not coast is None:
            om.drawcoast(**coast)

        if not states is None:
            om.drawstates(**states)

        m = om.get_map()
        x, y = m(llongs, llats)
        z = data

        # Filter data
        clevs = cfg['clevs']
        z[z<clevs[0]] = np.nan
        z[z>clevs[-1]] = np.nan

        # Refine levels
        if np.nanmax(z)<np.max(clevs):
            iw = np.min(np.where(np.nanmax(z) < np.sort(clevs))[0])
            clevs = clevs[:iw+1]

        # draw contour
        cs = m.contourf(x, y, z, cfg['clevs'], 
                    cmap=cfg['cmap'],
                    norm=cfg['norm'])

        return cs

    def plot_cbar(self, fig, ax, cs, *args, **kwargs):

        div = make_axes_locatable(ax)

        cbar_ax = div.append_axes(*args, **kwargs)

        cb = fig.colorbar(cs, cax=cbar_ax)


