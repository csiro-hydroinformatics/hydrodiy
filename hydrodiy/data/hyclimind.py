import re, os, sys
import datetime

from calendar import month_abbr as months

import json

# Deal with Python 2 and 3 related to StringIO
try:
    from cStringIO import cStringIO
except ImportError:
    from io import StringIO

# Deal with Python 2 and 3 related to UNICODE
try:
    UNICODE = unicode
except NameError:
    UNICODE = str


import urllib2

import numpy as np
import pandas as pd

NOAA_URL1 = 'http://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Data'
NOAA_URL2 = 'http://www.ncdc.noaa.gov/teleconnections'
BOM_SOI_URL = ('ftp://ftp.bom.gov.au/anon/home/ncc/' + \
                                'www/sco/soi/soiplaintext.html')

INDEX_NAMES = ['nao', 'pdo', 'soi', 'pna', \
                'nino12', 'nino34', 'nino4', 'ao', 'amo']


def get_data(index):
    ''' Download climate indices time series

    Parameters
    -----------
    index : str
        Climate index name : nao, pdo, soi, pna,
        nino12, nino34m nino4, ao, amo

    Returns
    -----------
    data : pandas.Series
        Index monthly data
    url : str
        Download URL

    Example
    -----------
    >>> nao = hyclimind('nao')
    '''

    if not index in INDEX_NAMES:
        raise ValueError('Index index({0}) is not in {1}'.format(index, \
                ','.join(INDEX_NAMES)))

    # Build url
    url = '{0}/{1}.long.data'.format(NOAA_URL1, index)
    sep = ' +'

    if index == 'soi':
        url = BOM_SOI_URL
        sep = '\t'

    if index in ['nao', 'pdo', 'pna', 'ao']:
        url = '{0}/{1}/data.json'.format(NOAA_URL2, index)

    if re.search('nino', index):
        url = re.sub('long', 'long.anom', url)

    # Download data
    req = urllib2.urlopen(url)
    txt = req.read()
    req.close()

    if index in ['nao', 'pdo', 'pna', 'ao']:
        data = json.loads(''.join(txt))
        series = pd.Series({pd.to_datetime(k, format='%Y%m'):
                    data['data'][k] for k in data['data']})

    else:
        # Convert to dataframe (tried read_csv but failed)
        iotxt = StringIO(UNICODE(txt))
        data = pd.read_csv(iotxt, skiprows=11, sep='   ', engine='python')

        # Build time series
        data.columns = ['year'] + [months[i] for i in range(1, 13)]
        series = pd.melt(data, id_vars='year')
        series = series[pd.notnull(series['value'])]

        def fun(x):
            return '1 {0} {1}'.format(x['variable'], x['year'])

        series['month'] = pd.to_datetime(
                            series[['year', 'variable']].apply(fun, axis=1))
        series = series.sort_values(by='month')
        series = series.set_index('month')
        series = series['value']
        series.name = '{0}[{1}]'.format(index, url)

    return series, url

