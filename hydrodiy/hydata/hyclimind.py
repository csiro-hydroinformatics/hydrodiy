import re
import os
import datetime

import json

import urllib2

import numpy as np
import pandas as pd

def tofloat(x):
    try:
        f = float('%s' % x)
    except ValueError:
        f = np.nan

    return f

def tofloats(s):
    return pd.Series([tofloat(x) for x in s], index=s.index)

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
        txt = re.split('\n', req.read())
        req.close()

        if name in ['nao', 'pdo', 'pna', 'ao']:
            
            data = json.loads(''.join(txt))
            data = pd.Series({pd.to_datetime(k, format='%Y%m'):data['data'][k] 
                        for k in data['data']})
            
        else:
            # Convert to dataframe (tried read_csv but failed)
            data = pd.DataFrame([re.split(sep, l) for l in txt])
            data = data.apply(tofloats)

            # Check not superfulous columns
            nmiss = data.apply(lambda x: np.sum(pd.notnull(x)))
            data = data[nmiss.index[nmiss>0]]
            data.columns = range(0, 13)

            # Remove first line if required
            if data.iloc[0,0] == data.iloc[1,0]:
                data = data.iloc[1:,:]

            # Remove superfluous lines
            nmiss = data.T.apply(lambda x: np.sum(pd.notnull(x)))
            data = data[nmiss > 1]

            # Build time series
            data = pd.melt(data, id_vars=0)
            data = data[pd.notnull(data[0])]

            data['month'] = data[[0, 'variable']].T.apply(lambda x: 
                                    datetime.datetime(int(x[0].squeeze()), 
                                        int(x['variable'].squeeze()), 1))

            data = data.sort('month')
            data = data.set_index('month')
            data = data['value']
            data.name = '%s[%s]' % (name, url)

        return data

