''' Objects used to download data from  AWAP '''

import re
import os
import datetime

from dateutil.relativedelta import relativedelta as delta
from subprocess import Popen, PIPE
import tempfile
import warnings

import numpy as np

from hydrodiy.io import iutils
from hydrodiy.gis.grid import Grid

from hydrodiy import PYVERSION

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

    # File name to be downloaded
    filename = '%4d%2.2d%2.2d%4d%2.2d%2.2d.grid' % (\
                    start.year, start.month, start.day, \
                    end.year, end.month, end.day)

    # File url
    url = '%s/%s/%s/%s/grid/0.05/history/nat/%s.Z' % (AWAP_URL, \
                varname, vartype, timestep, filename)

    # Temporary storage folder
    tmpdir = os.path.join(tempfile.gettempdir(), varname)
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)

    # Download data
    ftmp = os.path.join(tmpdir, filename+'.Z')
    try:
        iutils.download(url, filename=ftmp)
    except Exception as err:
        print('Cannot download {0}: Error = {1}'.format(url, err))
        raise err

    # Extract data from compressed file
    # (Unix compress format produced with the 'compress' unix command
    try:
        compressedfile = Popen(['zcat', ftmp], stdout=PIPE).stdout
        fdata = ftmp
    except FileNotFoundError as err:
        warnings.warn('zcat decompression utility is not available. Trying 7z')

        # Try 7z if the previous one does not work
        fdata = os.path.join(tmpdir, filename)
        try:
            Popen(['7z', 'e', ftmp, '-o'+tmpdir])
            compressedfile = open(fdata, 'r')

        except FileNotFoundError as err:
            raise ValueError('Problem with decompression of {0}: {1}'.format(\
                    url, str(err)))

    txt = compressedfile.readlines()
    if PYVERSION == 3:
        txt = [line.encode('utf-8') for line in txt]

    compressedfile.close()
    try:
        os.remove(ftmp)
        if os.path.exists(fdata):
            os.remove(fdata)
        os.remove(os.path.dirname(ftmp))
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

