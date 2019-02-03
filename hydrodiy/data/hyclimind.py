import re, os, sys
import datetime
from calendar import month_abbr as months
import numpy as np
import pandas as pd
import json

from ftplib import FTP

from hydrodiy.io import iutils
from hydrodiy import PYVERSION

from io import BytesIO
if PYVERSION == 2:
    from StringIO import StringIO
elif PYVERSION == 3:
    from io import StringIO

NOAA_URL1 = 'http://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Data'
NOAA_URL2 = 'http://www.ncdc.noaa.gov/teleconnections'

BOM_FTP = 'ftp.bom.gov.au'
BOM_SOI_DIR = 'anon/home/ncc/www/sco/soi'
BOM_SOI_FILE = 'soiplaintext.html'

INDEX_NAMES = ['nao', 'pdo', 'soi', 'pna', \
                'nino12', 'nino34', 'nino4', 'ao']

def get_data(index, timeout=300):
    ''' Download climate indices time series from NOAA and BoM

    Parameters
    -----------
    index : str
        Climate index name : nao, pdo, soi, pna,
        nino12, nino34m nino4, ao
    timeout : int
        Timeout is seconds

    Returns
    -----------
    data : pandas.core.series.Series
        Index monthly data
    url : str
        Download URL

    Example
    -----------
    >>> nao = hyclimind('nao')
    '''
    # Build url
    url = '{0}/{1}.long.data'.format(NOAA_URL1, index)
    sep = ' +'

    if index == 'soi':
        url = '/'.join([BOM_FTP, BOM_SOI_DIR, BOM_SOI_FILE])
        sep = '\t'

    elif index in ['nao', 'pdo', 'pna', 'ao']:
        url = '{0}/{1}/data.json'.format(NOAA_URL2, index)

    elif re.search('nino', index):
        url = re.sub('long', 'long.anom', url)

    elif not index in INDEX_NAMES:
        raise ValueError('Index index({0}) is not in {1}'.format(index, \
                ','.join(INDEX_NAMES)))

    # Download data
    if index == 'soi':
        ftp = FTP(BOM_FTP)
        ftp.login()
        ftp.cwd(BOM_SOI_DIR)
        stream = BytesIO()
        ftp.retrbinary('RETR {0}'.format(BOM_SOI_FILE), stream.write)
        stream.seek(0)
        ftp.close()

    else:
        stream = iutils.download(url, timeout=timeout)

    txt = stream.read().decode('cp437')

    if index in ['nao', 'pdo', 'pna', 'ao']:
        data = json.loads(txt)
        series = pd.Series({pd.to_datetime(k, format='%Y%m'):
                    data['data'][k] for k in data['data']})

    else:
        # Convert to dataframe (tried read_csv but failed)
        stream = StringIO(txt)
        if re.search('nino', index):
            colspecs = [[0, 5], [5, 14]]+\
                            [[14+8*m, 14+8*(m+1)] for m in range(11)]
            data = pd.read_fwf(stream, colspecs)

            # Remove last lines
            data = data.iloc[:-6, :]

        else:
            data = pd.read_csv(stream, skiprows=11, \
                                        sep='   ', engine='python')

        # Name columns
        data.columns = ['year'] + [months[i] for i in range(1, 13)]

        # Remove negative years
        idx = data.year.str.findall('^(1|2)').astype(bool)
        data = data.loc[idx, :]

        series = pd.melt(data, id_vars='year')
        series = series[pd.notnull(series['value'])]

        def fun(x):
            return '1 {0} {1}'.format(x['variable'], x['year'])

        day = series.loc[:, ['year', 'variable']].apply(fun, axis=1)
        series['month'] = pd.to_datetime(day)
        series = series.sort_values(by='month')
        series = series.set_index('month')
        series = series['value']
        series.name = '{0}[{1}]'.format(index, url)

    # Make sure series is float
    series = series.astype(float)

    # Check index
    if not series.index.is_unique:
        raise ValueError('Time index is not unique')

    # Set missing values to missing
    if re.search('nino', index):
        idx = np.abs(series+99.99)<1e-8
        series[idx] = np.nan

    return series, url

