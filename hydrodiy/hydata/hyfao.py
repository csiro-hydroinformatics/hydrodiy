import re
import requests
import datetime
import numpy as np
import pandas as pd

class HttpFAOError(Exception):
        pass

faostat_url = 'http://faostat3.fao.org/faostat-api/rest'

object_columns = {
        'countries':['country_code', 'country_label'], 
        'regions':['region_code', 'region_label'], 
        'specialgroups':['specialgroup_code', 
            'specialgroup_label'], 
        'items':['item_code', 'item_label'],
        'elements':['element_code', 'element_label'],
        'itemsaggregated':['itemagg_code', 'itemagg_label']
    }

class HyFAO():

    def __init__(self):

        self.current_url = ''

    def get_domain_codes(self):

        # Run query
        base_url = ('%s/groupsanddomains/faostat/E') % (
                        faostat_url)

        req = requests.get(base_url)

        js = req.json()

        # Convert json object to dataframe
        df = pd.DataFrame(js)
        
        df.columns = ['group_code', 'group_label',
                    'domain_code', 'domain_label',
                    'order']

        return df


    def get_object_codes(self, object_name, domain_code):

        if not object_name in object_columns:
            raise ValueError(('object_name(%s) not in ['
                '%s]') % (object_name, 
                    '/'.join(object_columns.keys())))

        # Run query
        base_url = ('%s/procedures/%s/faostat/%s/E') % (
                        faostat_url, object_name, domain_code)

        req = requests.get(base_url)

        self.current_url = req.url

        # Convert json object to dataframe
        js = req.json()

        df = pd.DataFrame(js)

        cc = object_columns[object_name]

        df.columns = cc + ['X%d' % (i+1) 
                            for i in range(df.shape[1]-len(cc))]

        df['object_name'] = object_name
        df['domain_code'] = domain_code

        # Remove useless columns
        cc = [cn for cn in df.columns if re.search('^X', cn)]
        df = df.drop(cc, axis=1)

        return df

    def search_object_codes(self, object_name, domain_code, 
                pattern):

        object_codes = self.get_object_codes(object_name, 
                                    domain_code)

        cc = [cn for cn in object_codes.columns 
                    if re.search('_label', cn)]
        
        labels = object_codes[cc].squeeze()

        idx = [bool(re.search(pattern, x)) for x in labels]

        return object_codes.loc[idx,:]


    def get_data(self, 
            domain_code,
            item_codes,
            element_codes,
            years=None,
            areaCodes=None):

        # Run query
        params = {
            'areaCodes':areaCodes,
            'years':years,
            'domainCode':domain_code,
            'itemCodes':item_codes,
            'elementListCodes':element_codes,
            'decimalPlaces':2,
            'units':'true',
            'codes':'true',
            'flags':'true',
            'nullValue':'false'
        }

        base_url = '%s/procedures/data' % faostat_url 

        req = requests.get(base_url, params=params)

        if req.status_code != 200:
            raise HttpFAOError('URL : %s\nRequest status : %d' % 
                    (req.url, req.status_code))

        self.current_url = req.url

        # Convert json object to dataframe
        js = req.json()

        df = pd.DataFrame(js)

        import pdb; pdb.set_trace()

        cc = ['X0', 'domain_label', 'country_label',
                'country_code', 'item_label', 'item_code',
                'element_label', 'element_code', 'year',
                'unit', 'value', 'flag']


        return df
    
