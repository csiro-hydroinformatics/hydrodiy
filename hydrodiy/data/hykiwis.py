import re
import requests
import datetime
import numpy as np
import pandas as pd

#
# This code is largely pasted from the kiwis_py package
# developed by Andrew McDonald, Bureau of Meteorology, EHP
# 

def testjson(req):
    ''' Test validity of json conversion '''
    
    try:
        out = req.json()
        return out

    except ValueError, thrown:
        return None

url_external = 'http://www.bom.gov.au/waterdata/KiWIS/KiWIS'

url_internal ='http://ccfvp-wadiapp04:8080/KiWIS/KiWIS'


class HyKiwis():

    def __init__(self, bom_internal = True):
        
        if bom_internal:

            # Test internal is ok
            req = requests.get(re.sub('/KiWIS$', '', url_internal))

            if req.status_code == 200:
                self.kiwis_url = url_internal

            else:
                raise ValueError('Cannot access bom internal WISKI server, '
                        'try external instead')

        else:
            self.kiwis_url = url_external

        self.base_params = {'service':'kisters',
                'type':'queryServices',
                'datasource':0,
                'format':'json'}

        self.current_url = ''

    def get_sites(self):
        ''' Get list of stations '''

        params = dict(self.base_params)
        params.update({
                'request': 'getStationList',
                'returnfields':'station_no,station_id,'
                    'station_latitude,station_longitude'
            })
        
        req = requests.get(self.kiwis_url, params=params)

        self.current_url = req.url

        sites = testjson(req)
        if sites is None or re.search('No matches', ' '.join(sites[0])):
            raise ValueError('No data')

        sites = pd.DataFrame(sites[1:], columns = sites[0])

        return sites 

    def get_tsattrs(self, id, ts_name = 'PR01QaQc.Merged.DailyMean.09HR'):
        ''' Retrieve time series id from station id '''

        params = dict(self.base_params)
        params.update({
                'request': 'getTimeseriesList',
                'ts_name': ts_name,
                'station_no':id,
                'returnfields':'ts_id,ts_unitname,ts_unitsymbol,'
                    'station_no,station_name,coverage'
            })
        
        req = requests.get(self.kiwis_url, params=params)

        self.current_url = req.url

        tsattrs = testjson(req)
        if tsattrs is None or re.search('No matches', ''.join(tsattrs[0])):
            raise ValueError('No data')

        tsattrs = {k:v for k, v in zip(tsattrs[0], tsattrs[1])}

        tsattrs['ts_name'] = ts_name

        tsattrs['to'] = datetime.datetime.strptime(tsattrs['to'][:10], 
                            "%Y-%m-%d")

        tsattrs['from'] = datetime.datetime.strptime(tsattrs['from'][:10], 
                            "%Y-%m-%d")

        return tsattrs

    def get_tsattrs_all(self, sites, ts_name = 'PR01QaQc.Merged.DailyMean.09HR'):
        ''' Retrieve time series attributes for all stations '''

        tsattrs = {}

        count = 1
        ns = sites.shape[0]

        for idx, row in sites.iterrows():

            if count % 10 == 0:
                print('.. retrieving ts attributes for site %5d/%5d ..' %(count, ns))
            count += 1

            id = row['station_no']
            
            try:
                a = self.get_tsattrs(id, ts_name)
            
            except ValueError:
                a = {}

            tsattrs[id] = a

        tsattrs = pd.DataFrame(tsattrs).T

        return tsattrs


    def get_data(self, tsattrs, from_dt='1900-01-01', to_dt='2100-12-31'):

        params = {'request':'getTimeseriesValues',
            'ts_id':tsattrs['ts_id'],
            'from': from_dt,
            'to': to_dt
        }
        params.update(self.base_params)
        params['format'] = 'dajson'

        req = requests.get(self.kiwis_url, params=params)

        self.current_url = req.url

        js = testjson(req)

        if js is None or len(js[0]['data']) == 0:
            ts = pd.Series([])

        else:
            d = np.array(js[0]['data'])
            ts = pd.Series(d[:,1], 
                    index=pd.to_datetime(d[:,0]), dtype='f')

            ts.name = '%s[%s]' % (tsattrs['ts_id'], tsattrs['ts_unitsymbol'])

        return ts
    
