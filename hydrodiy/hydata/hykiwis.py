import re
import requests
import numpy as np
import pandas as pd

def testjson(req):
    ''' Test validity of json conversion '''
    
    try:
        out = req.json()
        return out

    except ValueError, thrown:
        return None


class HyKiwis():

    def __init__(self, 
            kiwis_url ='http://www.bom.gov.au/waterdata/KiWIS/KiWIS'):

        self.kiwis_url = kiwis_url

        self.base_params = {'service':'kisters',
                'type':'queryServices',
                'datasource':0,
                'format':'json'}

    def getattrs(self, id, ts_name = 'PR01QaQc.Merged.DailyMean.09HR'):
        ''' Retrieve time series id from station id '''

        params = dict(self.base_params)
        params.update({
                'request': 'getTimeseriesList',
                'ts_name': ts_name,
                'station_no':id,
                'returnfields':'ts_id,ts_unitname,ts_unitsymbol'
            })
        
        req = requests.get(self.kiwis_url, params=params)
        # req.url gives url

        attrs = testjson(req)
        if attrs is None or re.search('No matches', ' '.join(attrs[0])):
            raise ValueError('No data')

        return {'tsid': attrs[1][0], 'unit':attrs[1][2]}


    def getdata(self, attrs, from_dt='1900-01-01', to_dt='2100-12-31'):

        params = {'request':'getTimeseriesValues',
            'ts_id':attrs['tsid'],
            'from': from_dt,
            'to': to_dt
        }
        params.update(self.base_params)
        params['format'] = 'dajson'

        req = requests.get(self.kiwis_url, params=params)
        js = testjson(req)

        if js is None or len(js[0]['data']) == 0:
            ts = pd.Series([])

        else:
            d = np.array(js[0]['data'])
            ts = pd.Series(d[:,1], 
                    index=pd.to_datetime(d[:,0]), dtype='f')

            ts.name = '%s[%s]' % (attrs['tsid'], attrs['unit'])

        return ts
    
