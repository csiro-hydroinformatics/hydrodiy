import re
import os
import datetime

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
        self.noaa_url = 'http://www.esrl.noaa.gov/psd/gcos_wgsp/Timeseries/Data'
        
        self.bom_soi_url = ('ftp://ftp.bom.gov.au/anon/home/ncc/'
                                'www/sco/soi/soiplaintext.html')
        
        self.index_names = ['soi', 'nino12', 'nino34', 'nino4',
                            'ao', 'amo']

    def get_data(self, name):
        ''' Download climate indices time series '''

        if not name in self.index_names:
            raise ValueError('wron index name(%s), not in %s' % (name, 
                    ','.join(self.index_names)))

        # Download data
        url = '%s/%s.long.data' % (self.noaa_url, name)
        sep = ' +'

        if name == 'soi':
            url = self.bom_soi_url
            sep = '\t'
        
        # download data
        req = urllib2.urlopen(url)
        txt = re.split('\n', req.read())
        req.close()

        # Convert to dataframe (tried read_csv but failed)
        data = pd.DataFrame([re.split(sep, l) for l in txt])
        data = data.apply(tofloats)

        # Check not superfulous columns
        nmiss = data.apply(lambda x: np.sum(pd.notnull(x)))
        data = data[nmiss.index[nmiss>0]]
        data.columns = range(0, 13)

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

