''' Objects used to download data from  AWAP '''

import re
import os
import datetime

try:
    import urllib2 as urllib
except ImportError:
    import urllib3 as urllib

from dateutil.relativedelta import relativedelta as delta

import tempfile
from subprocess import Popen, PIPE

import numpy as np

from hydrodiy.io import iutils
from hydrodiy.gis.grid import Grid

# Constants
VARIABLES = {
    'rainfall':[{'type':'totals', 'unit':'mm/d'}],
    'temperature':[{'type':'maxave', 'unit':'celsius'},
                   {'type':'minave', 'unit':'celsius'}],
    'vprp':[{'type':'vprph09', 'unit':'Pa'}],
    'solar':[{'type':'solarave', 'unit':'MJ/m2'}]
}

TIMESTEPS = ['day', 'month']

AWAP_URL = 'http://www.bom.gov.au/web03/ncc/www/awap'


def get_data(varname, vartype, timestep, date):
    ''' Download gridded awap data

        Parameters
        -----------
        varname : str
            Variable name [rainfall|temperature|vprp|solar]
        vartype : str
            Variable type (see hydrodiy.grid.VARIABLES)
        timestep : str
            Variable time step [daily|monthly]
        date : datetime.datetime
            Desired date

        Returns
        -----------
        grd : hydrodiy.grid.Grid
            Gridded data
    '''

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

    if not isinstance(date, datetime.datetime):
        raise ValueError(('date {0} is not a' + \
            ' datetime.datetime object').format(date))

    # Define start and end date of period
    start = date

    if (timestep == 'month') & (start.day != 1):
        raise ValueError(('Invalide date(%s). '+ \
            'Should be on day 1 of the month') % start.date())

    if timestep == 'day':
        timestep = 'daily'
        end = start

    if timestep == 'month':
        end = start + delta(months=1) - delta(days=1)

    # Download data
    url = ('%s/%s/%s/%s/grid/0.05/history/nat/'+ \
            '%4d%2.2d%2.2d%4d%2.2d%2.2d.grid.Z') % (AWAP_URL, \
                varname, vartype, timestep, \
                start.year, start.month, start.day, \
                end.year, end.month, end.day)

    try:
        resp = urllib.urlopen(url)

    except urllib.HTTPError as ehttp:
        print('Cannot download %s: HTTP Error = %s' % (url, ehttp))
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

    # Process grid header
    header = {k:float(v) \
        for k, v in [re.split(' +', s.strip()) for s in txt[:iheader]]}

    # Reformat header
    header['ncols'] = int(header['ncols'])
    header['nrows'] = int(header['nrows'])
    header['cellsize'] = float(header['cellsize'])

    header['xllcorner'] = float(header['xllcenter'])
    header.pop('xllcenter')
    header['yllcorner'] = float(header['yllcenter'])
    header.pop('yllcenter')

    header['name'] = iutils.dict2str({ \
        'varname': varname, \
        'vartype': vartype, \
        'date': date.date(), \
        })

    header['comment'] = iutils.dict2str({ \
        'url' : url \
        })

    # Get meta
    #meta = [s.strip() for s in txt[-icomment:]]

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

    # Create grid
    grd = Grid(**header)
    grd.data = data

    return grd

