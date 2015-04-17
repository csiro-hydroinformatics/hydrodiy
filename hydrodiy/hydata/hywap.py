import re
import os
import datetime

import urllib2

import tempfile

from subprocess import Popen, PIPE

import numpy as np
import pandas as pd

from mpl_toolkits.basemap import cm

from hyio import csv
from hygis import oz

class HyWap():
    ''' Class to download daily awap grids '''

    def __init__(self, 
            awap_url ='http://www.bom.gov.au/web03/ncc/www/awap'):

        self.awap_url = awap_url

        self.awap_dirs = {'rainfall':None, 'temperature':None}

    def set_awapdir(self, awap_dir):
        ''' Set AWAP output directory ''' 

        # Create output directories
        F = os.path.join(awap_dir, 'rainfall')
        if not os.path.exists(F):
            os.mkdir(F)
           
        self.awap_dirs['rainfall'] = F

        F = os.path.join(awap_dir, 'temperature')
        if not os.path.exists(F):
            os.mkdir(F)

        self.awap_dirs['temperature'] = F


    def getgriddata(self, varname, vartype, date):
        ''' Download gridded awap daily data '''

        if (varname == 'rainfall') & (vartype != 'totals'):
            raise ValueError('when varname=%s, vartype must be "totals"' % varname)
            
        if (varname == 'temperature') & ~(vartype in ['maxave', 'minave']):
            raise ValueError('when varname=%s, vartype must be "maxave" or "minave"' % varname)

        if not varname in ['rainfall', 'temperature']:
            raise ValueError('varname should be rainfall or temperature, not %s' % varname)

        dt = datetime.datetime.strptime(date, '%Y-%m-%d')

        # Download data
        url = ('%s/%s/%s/daily/grid/0.05/history/nat/'
                '%4d%2.2d%2.2d%4d%2.2d%2.2d.grid.Z') % (self.awap_url, 
                    varname, vartype, dt.year, dt.month, dt.day,
                    dt.year, dt.month, dt.day)
        
        try:
            resp = urllib2.urlopen(url)
        except urllib2.HTTPError, e:
            print('Cannot download %s: HTTP Error = %s' % (url, e))
            raise e

        zdata = resp.read()

        F = self.awap_dirs[varname]
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

        # Process grid
        header = {k:float(v) 
            for k,v in [re.split(' ', s.strip()) for s in txt[:6]]}

        meta = [s.strip() for s in txt[-18:]]

        data = [np.array(re.split(' ', s.strip())) for s in txt[6:-18]]
        data = np.array(data).astype(np.float)
        data[data == header['nodata_value']] = np.nan

        data = pd.DataFrame(data)
        data.columns = ['c%3.3d' % i for i in range(data.shape[1])]

        comment = ['AWAP Data set downloaded from %s' % self.awap_url, '', '',]
        comment += meta
        comment += ['']

        return data, comment, header

    def getcoords(self, header):

        nrows = int(header['nrows'])
        ncols = int(header['ncols'])
        xll = header['xllcenter']
        yll = header['yllcenter']
        sz = header['cellsize']

        cellnum = np.arange(1, nrows*ncols+1).reshape((nrows, ncols))

        longs = np.linspace(xll, xll+sz*ncols, ncols)
        lats = np.linspace(yll, yll+sz*nrows, nrows)[::-1]

        llongs, llats = np.meshgrid(longs, lats)

        return cellnum, llongs, llats
        
    
    def writegriddata(self, varname, vartype, date):

        data, comment, header = self.getgriddata(varname, vartype, date)

        F = self.awap_dirs[varname]
        if F is None:
            raise ValueError('Cannot write data, awap dir is not setup')

        fout = os.path.join(F, '%s_%s_%s.csv' % (varname, vartype, date))
        csv.write_csv(data, fout, comment)

        return '%s.gz' % fout

    def plotdata(self, data, header, ax,
            clevs = [0, 1, 5, 10, 15, 25, 50, 100, 150, 200, 300, 400],
            cmap = cm.s3pcpn):

        cellnum, llongs, llats, = self.getcoords(header)

        om = oz.Oz(ax = ax)

        om.drawcoast()
        om.drawstates()

        m = om.get_map()
        x, y = m(llongs, llats)
        cs = m.contourf(x, y, data.values, clevs, cmap=cmap)
        cbar = m.colorbar(cs, location = 'bottom', pad='5%')

