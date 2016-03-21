import re, os, sys
import datetime

import json

# Import of StringIO depends on python version
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import urllib2

import numpy as np
import pandas as pd


class HyClimInd():
    ''' Class to download climate indices '''

    def __init__(self):

        self.noaa_url1 = 'http://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Data'

        self.noaa_url2 = 'http://www.ncdc.noaa.gov/teleconnections'

        self.bom_soi_url = ('ftp://ftp.bom.gov.au/anon/home/ncc/'
                                'www/sco/soi/soiplaintext.html')

        self.index_names = ['nao', 'pdo', 'soi', 'pna',
                            'nino12', 'nino34', 'nino4',
                            'ao', 'amo']

    def get_data(self, name):
        ''' Download climate indices time series '''

        if not name in self.index_names:
            raise ValueError('wron index name(%s), not in %s' % (name,
                    ','.join(self.index_names)))

        # Download data
        url = '%s/%s.long.data' % (self.noaa_url1, name)
        sep = ' +'

        if name == 'soi':
            url = self.bom_soi_url
            sep = '\t'

        if name in ['nao', 'pdo', 'pna', 'ao']:
            url = '%s/%s/data.json' % (self.noaa_url2, name)

        if re.search('nino', name):
            url = re.sub('long', 'long.anom', url)

        # download data
        req = urllib2.urlopen(url)
        txt = req.read()
        req.close()

        if name in ['nao', 'pdo', 'pna', 'ao']:
            data = json.loads(''.join(txt))
            data = pd.Series(
                    {pd.to_datetime(k, format='%Y%m'):
                        data['data'][k] for k in data['data']})

        else:
            # Convert to dataframe (tried read_csv but failed)
            iotxt = StringIO(unicode(txt))
            data = pd.read_csv(iotxt, skiprows=11, sep='   ', engine='python')

            # Build time series
            cn = data.columns[0]
            data = pd.melt(data, id_vars=cn)
            data = data[pd.notnull(data['value'])]

            def fun(x):
                return '1 {0} {1}'.format(x['variable'], x[cn])

            data['month'] = pd.to_datetime(
                            data[[cn, 'variable']].apply(fun, axis=1))

            data = data.sort('month')
            data = data.set_index('month')
            data = data['value']
            data.name = '%s[%s]' % (name, url)

        return data

